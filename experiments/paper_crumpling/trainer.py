import os
import json
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime

from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from networks.PoissonNet import PoissonNet
from .dataset import CrumpledPaperDataset
from geometry.operators import construct_mesh_operators
from utils.helpers import cycle, seed_everything, count_parameters, MSE_loss, to_np
from viz.helpers import render_mesh, render_overlayed_meshes, add_text, image_grid
from viz.video import save_tensor_as_video

seed_everything(31415)
device = torch.device('cuda:0')

# === Config ===
with open('./experiments/paper_crumpling/config.json', 'r') as f:
    config = json.load(f)

config['exp_name'] = 'crumpling'
config['device'] = str(device)
model_name = config.get('name', f"{config['arch']}")
batch_size = config['batch_size']
grad_accum = config['grad_accum']
lr = config['lr']
clip_grad_norm = config['clip_grad_norm']
train_steps = config['train_steps']
mass_mse = config['mass_mse']
aug_rot = config['aug_rot']
viz_steps = config['viz_steps']

lambda_v = config['lambda_v']
lambda_g = config['lambda_g']

exp_name = config['exp_name']
os.makedirs(os.path.join('results', exp_name), exist_ok=True)
outfile = lambda x: os.path.join('results', exp_name, x)

# === Data ===
train_dataset = CrumpledPaperDataset(train=True, config=config)
test_dataset = CrumpledPaperDataset(train=False, config=config)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
train_loader = cycle(train_loader)
test_loader = cycle(test_loader)

model = PoissonNet( C_in=3,
                    C_out=3,
                    C_width=config['width'],
                    n_blocks=config['nblocks'],
                    head='njf',
                    extra_features=1, # animation time
                    config=config,)

print('Model parameters:', count_parameters(model))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Prepare source crumpled paper mesh just once for training -- all batches use the same source mesh:
verts_src, faces_src = train_dataset.get_source_cloth()
verts_src, faces_src = verts_src.to(device).unsqueeze(0), faces_src.to(device).unsqueeze(0)
verts_src = verts_src.repeat(batch_size, 1, 1)
faces_src = faces_src.repeat(batch_size, 1, 1)
mass_src, solver_src, G_src, M_src = construct_mesh_operators(verts_src, faces_src, high_precision=True)

@torch.no_grad()
def form_batch(data_loader):
    verts, _, time = next(data_loader)

    verts = verts.to(device)
    time = time.to(device)

    if time.ndim == 1:
        time = time.unsqueeze(0)

    return verts, time

def compute_loss(pred_v, pred_grad, tar_v, G, v_mass, f_mass, mass_weighted=True):
    tar_grad = torch.bmm(G, tar_v)
    loss_v = MSE_loss(pred_v, tar_v, v_mass, mass_weighted)
    loss_g = MSE_loss(pred_grad, tar_grad, f_mass, mass_weighted)
    return loss_v, loss_g

def train_batch(batch_i):
    model.train()

    batch_loss_v = 0
    batch_loss_g = 0
    batch_loss = 0
    accums = 0

    while accums < grad_accum:
        verts_tar, time = form_batch(train_loader)

        preds, preds_grad = model(x_in=verts_src, M=M_src, G=G_src, solver=solver_src, faces=faces_src, vertex_mass=mass_src, extra_features=time)

        # Align pred centroid with target centroid to stabilize training:
        preds_mean = preds.mean(dim=1, keepdim=True)
        tar_mean = verts_tar.mean(dim=1, keepdim=True)
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
        accums += 1

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    optimizer.step()
    optimizer.zero_grad()

    if batch_i % viz_steps == 0:
        vsrc_np = to_np(verts_src[0])
        f_np = to_np(faces_src[0])
        vtar_np = to_np(verts_tar[0])
        preds_np = to_np(preds[0])

        # Render src tar and pred mesh:
        render_src = add_text(render_mesh(vsrc_np, f_np), caption='source')
        render_tar = add_text(render_mesh(vtar_np, f_np), caption='target')
        render_pred = add_text(render_mesh(preds_np, f_np), caption='output')
        render_overlayed = add_text(render_overlayed_meshes([vtar_np, preds_np], [f_np, f_np]), caption='overlayed')
        render = torch.cat([render_src, render_tar, render_pred, render_overlayed], dim=-1)
        save_image(render, outfile('viz_train.png'))

    return batch_loss, batch_loss_v, batch_loss_g

@torch.no_grad()
def render_video(step_i):
    print('Rendering crumpling paper video...')
    model.eval()

    # assume time is in [0, 1]
    nframes = len(train_dataset)
    time = torch.linspace(0, 1, nframes).unsqueeze(-1).to(device)

    # Use global source mesh for all forward passes -- we'll decode using batch size 1:
    _verts_src = verts_src[0].unsqueeze(0)
    _faces_src = faces_src[0].unsqueeze(0)
    _mass_src = mass_src[0].unsqueeze(0)
    _M_src = M_src[0].unsqueeze(0)
    _G_src = G_src[0].unsqueeze(0)
    _solver_src = solver_src[:1]

    all_frames = []
    pbar = tqdm(range(0, nframes), dynamic_ncols=True, desc='Decoding and rendering crumpling paper video')
    for i in pbar:
        t = time[i:i+1]
        preds, _ = model(x_in=_verts_src, M=_M_src, G=_G_src, solver=_solver_src, faces=_faces_src, vertex_mass=_mass_src, extra_features=t)
        preds = preds - preds.mean(dim=1, keepdim=True)
        preds_np = to_np(preds[0])
        f_np = to_np(_faces_src[0])

        # render alongside ground truth:
        t_to_i = int(t.item() * (len(train_dataset) - 1))
        vgt_np = to_np(train_dataset.__getitem__(t_to_i)[0])
        render_pred = add_text(render_mesh(preds_np, f_np), caption=f'Prediction - time={t.item():.2f}')
        render_gt = add_text(render_mesh(vgt_np, f_np), caption=f'Ground truth - time={t.item():.2f}')
        render = torch.cat([render_pred, render_gt], dim=-1)
        all_frames += [render]

    all_frames = torch.stack(all_frames, dim=0) # (nframes, 3, H, W)
    save_tensor_as_video(all_frames, outfile(f'crumpling_paper_{step_i}'), reencode=True, boomerang=True, fps=24)


train_losses = []
train_losses_v = []
train_losses_g = []
pbar = tqdm(range(train_steps), dynamic_ncols=True)
for step_i in pbar:
    train_loss, train_loss_v, train_loss_g = train_batch(step_i)
    train_losses += [train_loss]
    train_losses_v += [train_loss_v]
    train_losses_g += [train_loss_g]
    
    if step_i % 4000 == 0 and step_i > 0:
        render_video(step_i)
        torch.save(model.state_dict(), outfile(f'poissonnet_crumpling_paper_{step_i}.pt'))

    if step_i % 100 == 0:
        print("Step {} - Train overall: {:.5f}".format(step_i, train_loss))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 20))
        ax[0].plot(train_losses, label='Train')
        ax[0].plot(train_losses_v, label='vertex')
        ax[0].plot(train_losses_g, label='gradient')
        ax[0].set_ylim(0, 0.8)
        ax[0].legend()
        ax[0].set_title('Train loss')
        plt.tight_layout()
        plt.savefig(outfile('viz_loss.png'))
        plt.close()

    pbar.set_description(f"Train loss: {train_loss:.5f}")

torch.save(model.state_dict(), outfile(f'poissonnet_crumpling_paper.pt'))