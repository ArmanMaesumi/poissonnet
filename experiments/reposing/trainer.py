import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from networks.PoissonNet import PoissonNet
from .dataset import MOYOBakedDataset
from geometry.operators import construct_mesh_operators
from utils.helpers import cycle, seed_everything, count_parameters, MSE_loss, to_np
from viz.helpers import render_mesh, render_overlayed_meshes, add_text, image_grid

seed_everything(31415)
device = torch.device('cuda:0')

# === Config ===
with open('./experiments/reposing/config.json', 'r') as f:
    config = json.load(f)

config['exp_name'] = 'repose'
batch_size = config['batch_size']
grad_accum = config['grad_accum']
lr = config['lr']
clip_grad_norm = config['clip_grad_norm']
train_steps = config['train_steps']
mass_mse = config['mass_mse']
viz_steps = config['viz_steps']

lambda_v = config.get('lambda_v', 1.0)
lambda_g = config.get('lambda_g', 1.0)

exp_name = config['exp_name']
os.makedirs(os.path.join('results', exp_name), exist_ok=True)
outfile = lambda x: os.path.join('results', exp_name, x)

# === Data ===
data_dir='./smplx_data'
train_dataset = MOYOBakedDataset(data_dir=data_dir, train=True, config=config)
test_dataset = MOYOBakedDataset(data_dir=data_dir, train=False, config=config)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = cycle(train_loader)
test_loader = cycle(test_loader)

pose_ndim = train_dataset.num_pose_params
model = PoissonNet( C_in=3,
                    C_out=3,
                    C_width=config['width'],
                    n_blocks=config['nblocks'],
                    head='njf',
                    extra_features=pose_ndim,
                    config=config,)
    
print('Model parameters:', count_parameters(model))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

@torch.no_grad()
def form_batch(data_loader, augment=False):
    verts_src, verts_tar, faces, pose_params = next(data_loader)

    if augment: 
        # augment src/target global scale before computing operators
        scale_xyz = torch.rand(verts_src.shape[0], 1, 1) * 0.6 + 0.7
        verts_src = verts_src * scale_xyz
        verts_tar = verts_tar * scale_xyz

        # augment mesh position:
        shift_xyz = torch.randn(verts_src.shape[0], 1, 3, device=verts_src.device) * 0.15
        verts_src = verts_src + shift_xyz
        verts_tar = verts_tar + shift_xyz

    verts_src = verts_src.to(device)
    verts_tar = verts_tar.to(device)
    faces = faces.to(device)
    pose_params = pose_params.to(device)

    mass_src, solver_src, G_src, M_src = construct_mesh_operators(verts_src, faces, high_precision=True)
    return verts_src, verts_tar, faces, mass_src, solver_src, G_src, M_src, pose_params

def compute_loss(pred_v, pred_grad, tar_v, G, v_mass, f_mass):
    tar_grad = torch.bmm(G, tar_v)
    loss_v = MSE_loss(pred_v, tar_v, v_mass, mass_weighted=True)
    loss_g = MSE_loss(pred_grad, tar_grad, f_mass, mass_weighted=True)
    return loss_v, loss_g

def train_batch(batch_i):
    model.train()

    batch_loss_v = 0
    batch_loss_g = 0
    batch_loss = 0
    batch_samples = 0

    while batch_samples < grad_accum:
        verts_src, verts_tar, faces, mass_src, solver_src, G_src, M_src, pose_params = form_batch(train_loader, augment=True)

        preds, preds_grad = model(
            x_in=verts_src,
            M=M_src, 
            G=G_src, 
            solver=solver_src, 
            faces=faces, 
            vertex_mass=mass_src,
            extra_features=pose_params
        )

        # Align pred centroid with target centroid to stabilize training:
        preds_mean = preds.mean(dim=1, keepdim=True) # (B, 1, 3)
        tar_mean = verts_tar.mean(dim=1, keepdim=True) # (B, 1, 3)
        verts_tar = verts_tar - tar_mean
        preds = preds - preds_mean

        # L = λ_v * ‖v_tar - v_pred‖^2 + λ_g * ‖∇_src v_tar - ∇_src v_pred‖^2
        loss_v, loss_g = compute_loss(preds, preds_grad, verts_tar, G_src, mass_src, M_src, mass_weighted=mass_mse)
        loss_v = loss_v * lambda_v
        loss_g = loss_g * lambda_g
        loss = (loss_v + loss_g) / grad_accum
        loss.backward() 

        batch_loss += loss.item()
        batch_loss_v += loss_v.item() / grad_accum
        batch_loss_g += loss_g.item() / grad_accum
        batch_samples += preds.shape[0] # batch size

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    optimizer.step()
    optimizer.zero_grad()

    if batch_i % viz_steps == 0:
        vsrc_np = to_np(verts_src[0])
        f_np = to_np(faces[0])
        vtar_np = to_np(verts_tar[0])
        preds_np = to_np(preds[0])
        
        render_src = add_text(render_mesh(vsrc_np, f_np), caption='source')
        render_tar = add_text(render_mesh(vtar_np, f_np), caption='target')
        render_pred = add_text(render_mesh(preds_np, f_np), caption='output')
        render_overlayed = add_text(render_overlayed_meshes([vtar_np, preds_np], [f_np, f_np]), caption='overlayed')
        render = torch.cat([render_src, render_tar, render_pred, render_overlayed], dim=-1)
        save_image(render, outfile('viz_train.png'))

    return batch_loss, batch_loss_v, batch_loss_g

@torch.no_grad()
def test():
    model.eval()
    
    total_loss = 0
    total_loss_v = 0
    total_loss_g = 0
    total_samples = 0

    renders = []
    render_idxs = random.sample(range(len(test_dataset)), 9) if len(test_dataset) > 9 else range(len(test_dataset))
    for i in range(len(test_dataset)):
        verts_src, verts_tar, faces, mass_src, solver_src, G_src, M_src, pose_params = form_batch(test_loader)

        preds, preds_grad = model(
            x_in=verts_src,
            M=M_src, 
            G=G_src, 
            solver=solver_src, 
            faces=faces, 
            vertex_mass=mass_src,
            extra_features=pose_params
        )

        preds_mean = preds.mean(dim=1, keepdim=True)
        tar_mean = verts_tar.mean(dim=1, keepdim=True)
        verts_tar = verts_tar - tar_mean
        preds = preds - preds_mean

        loss_v, loss_g = compute_loss(preds, preds_grad, verts_tar, G_src, mass_src, M_src, mass_weighted=mass_mse)
        loss_v = loss_v * lambda_v
        loss_g = loss_g * lambda_g
        loss = loss_v + loss_g

        total_loss += loss.item()
        total_loss_v += loss_v.item()
        total_loss_g += loss_g.item()
        total_samples += 1

        if i in render_idxs:
            vsrc_np = to_np(verts_src[0])
            f_np = to_np(faces[0])
            vtar_np = to_np(verts_tar[0])
            preds_np = to_np(preds[0])

            render_src = add_text(render_mesh(vsrc_np, f_np), caption='source')
            render_tar = add_text(render_mesh(vtar_np, f_np), caption='target')
            render_pred = add_text(render_mesh(preds_np, f_np), caption='output')
            render_overlayed = add_text(render_overlayed_meshes([vtar_np, preds_np], [f_np, f_np]), caption='overlayed')
            render = torch.cat([render_src, render_tar, render_pred, render_overlayed], dim=-1)
            renders += [render]

    renders = image_grid(renders)
    save_image(renders, outfile('viz_test.png'))

    total_loss /= total_samples
    total_loss_v /= total_samples
    total_loss_g /= total_samples

    return total_loss


train_losses = []
train_losses_v = []
train_losses_g = []
test_losses = []
test_steps = []
pbar = tqdm(range(train_steps), dynamic_ncols=True)
for step_i in pbar:
    train_loss, train_loss_v, train_loss_g = train_batch(step_i)
    
    train_losses += [train_loss]
    train_losses_v += [train_loss_v]
    train_losses_g += [train_loss_g]

    if step_i % viz_steps == 0:
        test_loss = test()
        test_losses += [test_loss]
        test_steps += [step_i]
        torch.save(model.state_dict(), outfile(f'poissonnet_repose_{step_i}_{test_loss:.4f}.pt'))

    if step_i % 500 == 0:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 20))
        ax[0].plot(train_losses, label='Train')
        ax[0].plot(train_losses_v, label='vertex')
        ax[0].plot(train_losses_g, label='gradient')
        ax[0].set_ylim(0, 0.8)
        ax[0].legend()
        ax[0].set_title('Train loss')
        ax[1].plot(test_steps, test_losses, label='Test')
        ax[1].set_title('Test loss')
        plt.tight_layout()
        plt.savefig(outfile('loss.png'))
        plt.close()

    pbar.set_description(f"Train loss: {train_loss:.5f}")

torch.save(model.state_dict(), outfile(f'poissonnet_repose_final.pt'))
print("Training complete")
