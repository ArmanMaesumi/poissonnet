import os
import json
import torch
import matplotlib.pyplot as plt
import geometry.point_utils as Point

from tqdm import tqdm
from torch.utils.data import DataLoader
from networks.PoissonNet import PoissonNet
from .dataset import Shrec11MeshDataset_Simplified
from geometry.operators import construct_mesh_operators
from utils.helpers import cycle, seed_everything, count_parameters, label_smoothing_log_loss

seed_everything(31415)
device = torch.device('cuda:0')

# --- Config ---
with open('./experiments/shrec11_classification/config.json', 'r') as f:
    config = json.load(f)
config['exp_name'] = 'shrec11'
batch_size = config['batch_size']
grad_accum = config['grad_accum']
lr = config['lr']
clip_grad_norm = config['clip_grad_norm']
train_steps = config['train_steps']
viz_steps = config['viz_steps']
label_smoothing = config.get('label_smoothing', 0.2)

exp_name = config['exp_name']
os.makedirs(os.path.join('results', exp_name), exist_ok=True)
outfile = lambda x: os.path.join('results', exp_name, x)

train_dataset = Shrec11MeshDataset_Simplified(train=True, config=config)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
train_loader = cycle(train_loader)

dataset_entry_dict = train_dataset.entries
test_dataset = Shrec11MeshDataset_Simplified(train=False, config=config, entry_dict=dataset_entry_dict)
test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
test_loader = cycle(test_loader)

model = PoissonNet( C_in=3,
                    C_out=train_dataset.n_class,
                    C_width=config['width'],
                    n_blocks=config['nblocks'],
                    head=config['head'],
                    outputs_at='global_mean',
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

    verts = verts.to(device)
    faces = faces.to(device)
    labels = labels.to(device)

    # apply rotation, scale, and shift augmentation:
    if augment:
        verts = Point.random_rotate_points_batched(verts)
        scale_xyz = torch.rand(verts.shape[0], 1, 1, device=verts.device) * 0.6 + 0.7
        verts = verts * scale_xyz
        shift = torch.randn(verts.shape[0], 1, 3, device=verts.device) * 0.25
        verts = verts + shift

    vert_mass, solver, G, M = construct_mesh_operators(verts, faces, high_precision=True)
    return verts, faces, vert_mass, solver, G, M, labels

def train_batch(batch_i):
    model.train()

    batch_loss = 0
    batch_correct = 0
    batch_samples = 0

    while batch_samples < grad_accum:
        verts, faces, vert_mass, solver, G, M, labels = form_batch(train_loader, augment=True)

        preds = model(
            x_in=verts, 
            M=M, 
            G=G, 
            solver=solver, 
            faces=faces, 
            vertex_mass=vert_mass
        ) # (B, num_classes)

        preds = preds.squeeze(0)
        labels = labels.squeeze(0)
        loss = label_smoothing_log_loss(preds, labels, smoothing=label_smoothing) / grad_accum
        loss.backward()
        
        pred_labels = torch.max(preds, dim=0).indices
        this_correct = pred_labels.eq(labels).sum().item()

        batch_correct += this_correct
        batch_loss += loss.item()
        batch_samples += 1
    
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    batch_acc = batch_correct / batch_samples    
    return batch_loss, batch_acc

@torch.no_grad()
def test():
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_samples = 0

    for i in range(len(test_dataset)):
        verts, faces, vert_mass, solver, G, M, labels = form_batch(test_loader)

        preds = model(
            x_in=verts,
            M=M, 
            G=G, 
            solver=solver, 
            faces=faces, 
            vertex_mass=vert_mass
        )
        preds = preds.squeeze(0)
        labels = labels.squeeze(0)
        loss = label_smoothing_log_loss(preds, labels, smoothing=label_smoothing)
        
        pred_labels = torch.max(preds, dim=0).indices
        this_correct = pred_labels.eq(labels).sum().item()

        test_correct += this_correct
        test_loss += loss.item()
        test_samples += 1
    
    test_loss = test_loss / test_samples
    test_acc = test_correct / test_samples
    return test_loss, test_acc


train_losses, test_losses = [], []
train_accs, test_accs = [], []
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
        if test_acc > best_test_acc: # save new best model
            best_test_acc = test_acc
            torch.save(model.state_dict(), outfile(f'poissonnet_shrec11_{step_i}_{test_acc:.4f}.pt'))
        test_losses += [test_loss]
        test_accs += [test_acc]
        test_steps += [step_i]
    
    if step_i % 50 == 0:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        ax[0].plot(train_losses, label=f'Train Loss')
        ax[0].set_title(f'Log Loss - shrec11')
        ax[0].set_yscale('log')
        ax[0].plot(test_steps, test_losses, label=f'Test Loss')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(train_accs, label=f'Train Accuracy')
        ax[1].set_title(f'Training Accuracy - shrec11')
        ax[1].plot(test_steps, test_accs, label=f'Test Accuracy')

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

# Save last model:
test_loss, test_acc = test()
torch.save(model.state_dict(), outfile(f'poissonnet_shrec11_final_{test_acc:.4f}.pt'))
print(f"Training complete. Best Test Acc: {(100*best_test_acc):.2f}%")
