import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import geometry.point_utils as Point
from torchvision.utils import save_image
from utils.helpers import *
from viz.helpers import *
from .dataset import MOYOSegmentationDataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from networks.PoissonNet import PoissonNet
from geometry.operators import construct_mesh_operators
from utils.helpers import cycle, seed_everything, count_parameters

seed_everything(31415)
device = torch.device('cuda:0')

# === Config ===
with open('./experiments/segmentation/config.json', 'r') as f:
    config = json.load(f)
config['exp_name'] = 'segmentation'
config['device'] = str(device)
batch_size = config['batch_size']
grad_accum = config['grad_accum']
lr = config['lr']
clip_grad_norm = config['clip_grad_norm']
train_steps = config['train_steps']
viz_steps = config['viz_steps']
model_name = config.get('name', 'moyo-segmentation')

exp_name = config['exp_name']
os.makedirs(os.path.join('results', exp_name), exist_ok=True)
outfile = lambda x: os.path.join('results', exp_name, x)

data_dir = './smplx_data'
train_dataset = MOYOSegmentationDataset(data_dir=data_dir, train=True, config=config)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
train_loader = cycle(train_loader)

test_dataset = MOYOSegmentationDataset(data_dir=data_dir, train=False, config=config)
test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
test_loader = cycle(test_loader)

model = PoissonNet(C_in=3,
                    C_out=train_dataset.n_class,
                    C_width=config['width'],
                    n_blocks=config['nblocks'],
                    head=config['head'],
                    outputs_at='faces',
                    last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                    config=config,)
    
print('Model parameters:', count_parameters(model))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

@torch.no_grad()
def form_batch(loader, augment=False):
    verts, faces, labels = next(loader)

    if verts.ndim == 2: # ensure batch dimension:
        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)
        labels = labels.unsqueeze(0)

    if augment:
        verts = Point.random_rotate_points_batched(verts)
        scale_xyz = torch.rand(verts.shape[0], 1, 1) * 0.6 + 0.7
        verts = verts * scale_xyz
        shift = torch.randn(verts.shape[0], 1, 3) * 0.25
        verts = verts + shift

    verts = verts.to(device)
    faces = faces.to(device)
    labels = labels.to(device)

    vertex_mass, solver, G, M = construct_mesh_operators(verts, faces, high_precision=True)
    
    return verts, faces, vertex_mass, solver, G, M, labels

def train_batch(batch_i):
    model.train()

    batch_loss = 0
    batch_correct = 0
    batch_faces = 0
    batch_samples = 0

    while batch_samples < grad_accum:
        verts, faces, vertex_mass, solver, G, M, labels = form_batch(train_loader, augment=True)
        
        preds = model(
            x_in=verts,
            M=M, 
            G=G, 
            solver=solver, 
            faces=faces, 
            vertex_mass=vertex_mass
        ) # (B, F, cls)
        preds = preds.squeeze(0)
        labels = labels.squeeze(0)
        
        loss = torch.nn.functional.nll_loss(preds, labels) / grad_accum
        loss.backward()
        
        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()

        batch_correct += this_correct
        batch_loss += loss.item()
        batch_faces += preds.shape[0] # number of faces
        batch_samples += 1
    
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    if batch_i % viz_steps == 0:
        with torch.no_grad():
            verts_np = to_np(verts[0])
            preds_np = to_np(preds)
            faces_np = to_np(faces[0])
            labels_np = to_np(labels)
            class_np = np.argmax(preds_np, axis=1) # map preds to class ids

            render_pred = add_text(render_segmentation(verts_np, faces_np, class_np), 'Prediction')
            render_gt = add_text(render_segmentation(verts_np, faces_np, labels_np), 'Ground Truth')
            render = torch.cat([render_pred, render_gt], dim=-1)
            save_image(render, outfile('viz_train.png'))

    batch_acc = batch_correct / batch_faces
    return batch_loss, batch_acc

@torch.no_grad()
def test():
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_samples = 0

    # sample 32 random indices for visualization:
    render_idx = torch.randint(0, len(test_dataset), (32,)).tolist()
    renders = []
    for i in range(len(test_dataset)):
        verts, faces, vertex_mass, solver, G, M, labels = form_batch(test_loader)

        preds = model(
            x_in=verts,
            M=M, 
            G=G, 
            solver=solver, 
            faces=faces, 
            vertex_mass=vertex_mass
        )
        preds = preds.squeeze(0)
        labels = labels.squeeze(0)
        loss = torch.nn.functional.nll_loss(preds, labels)
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        
        test_correct += this_correct
        test_loss += loss.item()
        test_samples += preds.shape[0] # number of faces

        # Viz:
        verts_np = to_np(verts[0])
        preds_np = to_np(preds)
        faces_np = to_np(faces[0])
        labels_np = to_np(labels)
        class_np = np.argmax(preds_np, axis=1) # map preds to class ids

        # Render src tar and pred mesh:
        if i in render_idx:
            render_pred = add_text(render_segmentation(verts_np, faces_np, class_np), 'Prediction')
            render_gt = add_text(render_segmentation(verts_np, faces_np, labels_np), 'Ground Truth')
            render = torch.cat([render_pred, render_gt], dim=-1)
            renders += [render]
    
    renders = image_grid(renders)
    save_image(renders, outfile('viz_test.png'))
    
    test_loss = test_loss / len(test_dataset)
    test_acc = test_correct / test_samples
    return test_loss, test_acc


train_losses = []
train_accs = []
test_losses = []
test_accs = []
test_steps = []
test_loss, test_acc = 0.0, 0.0
best_test_acc = 0.0
pbar = tqdm(range(train_steps), dynamic_ncols=True)
for step_i in pbar:
    train_loss, train_acc = train_batch(step_i)
    train_losses += [train_loss]
    train_accs += [train_acc]
    
    if step_i % viz_steps == 0 and step_i > 0:
        test_loss, test_acc = test()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), outfile(f'{model_name}_{step_i}_{test_acc:.4f}.pt'))
        test_losses += [test_loss]
        test_accs += [test_acc]
        test_steps += [step_i]
    
    if step_i % 50 == 0: 
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        ax[0].plot(train_losses, label=f'{model_name} Train Loss')
        ax[0].set_title(f'Loss - {model_name}')
        ax[0].set_yscale('log')
        ax[0].plot(test_steps, test_losses, label=f'{model_name} Test Loss')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(train_accs, label=f'{model_name} Train Accuracy')
        ax[1].set_title(f'Training Accuracy - {model_name}')
        ax[1].plot(test_steps, test_accs, label=f'{model_name} Test Accuracy')

        # horizonal line for best test accuracy:
        best_test_acc = max(test_accs) if len(test_accs) > 0 else 0
        ax[1].axhline(y=best_test_acc, color='r', linestyle='--', label=f'Best Test Acc: {100*best_test_acc:.2f}%', alpha=0.5)
        ax[1].grid(True)
        ax[1].legend()
        
        plt.tight_layout()
        plt.savefig(outfile('loss_plot.png'))
        plt.close()
    
    best_test_acc = max(test_accs) if len(test_accs) > 0 else 0
    pbar.set_description(f"Train Loss: {train_loss:.6f}, Train Acc: {(100*train_acc):.2f}% | Test Loss: {test_loss:.6f}, Test Acc: {(100*test_acc):.2f}%, Best Test Acc: {(100*best_test_acc):.2f}%")

# Save model and loss/acc history:
torch.save(model.state_dict(), outfile(f'{model_name}.pt'))
np.savez(outfile(f'losses_{model_name}.npz'), 
            train_losses=train_losses,
            test_losses=test_losses,
            train_accs=train_accs,
            test_accs=test_accs)
print("Training complete")