import json
import os
import random
import igl
import torch
import numpy as np
import matplotlib.pyplot as plt
import panopti

import geometry.point_utils as Point
from networks.PoissonNet import PoissonNet
from geometry.operators import construct_mesh_operators
from utils.helpers import count_parameters, to_np

# === global setup ===
device = torch.device('cuda:0')
pano = panopti.connect(server_url='http://localhost:8080', viewer_id='client') 

# === load pretrained model ===
with open('./experiments/reposing/config.json', 'r') as f:
    config = json.load(f)

model = PoissonNet(C_in=3,
                    C_out=3,
                    C_width=config['width'],
                    n_blocks=config['nblocks'],
                    head='njf',
                    extra_features=config['extra_features'], # smplx pose parameters
                    config=config,)
chkpt = torch.load('./experiments/reposing/poissonnet_reposing_87000_0.0243.pt')
model.load_state_dict(chkpt['model'])
model.to(device)
print('Loaded model with parameters:', count_parameters(model))

# === load example poses === 
poses = np.load('./demo_meshes/example_poses.npy')
poses = torch.from_numpy(poses).float()
pose_idx = 0 # keep track of current pose in viewer

@torch.no_grad()
def run_model(viewer):
    global pose_idx
    obj_name = viewer.get('Characters').value()
    verts_np, faces_np = igl.read_triangle_mesh(f'./demo_meshes/{obj_name}.obj')
    verts_np = Point.normalize_mesh(verts_np, faces_np, mode='surface_area')

    verts = torch.from_numpy(verts_np).float().to(device).unsqueeze(0)
    faces = torch.from_numpy(faces_np).long().to(device).unsqueeze(0)
    vert_mass, solvers, G, M = construct_mesh_operators(verts, faces)

    pose_params = poses[pose_idx].to(device).unsqueeze(0)

    preds, _ = model(
        x_in=verts, 
        M=M, 
        G=G, 
        solver=solvers, 
        faces=faces, 
        vertex_mass=vert_mass,
        extra_features=pose_params
    )
    preds = preds.squeeze(0)

    # Display meshes in panopti viewer:
    pano.add_mesh(
        name='source mesh',
        vertices=verts_np,
        faces=faces_np,
        vertex_colors=np.ones_like(verts_np) * [0.85, 0.85, 1.0],
    )

    pano.add_mesh(
        name='poissonnet output',
        vertices=to_np(preds),
        faces=faces_np,
        vertex_colors=np.ones_like(verts_np) * [1.0, 0.5, 0.5],
        position=(1.0, 0.0, 0.0)
    )

# === setup panopti scene ===
# Mesh picker dropdown menu:
objs = [o for o in os.listdir('./demo_meshes') if o.endswith('.obj')]
objs = [o.split('.')[0] for o in objs]
pano.dropdown(callback=None, name='Characters', options=objs)

pano.button(callback=run_model, name='Run Model')

def callback_prev_pose(viewer):
    global pose_idx
    pose_idx -= 1
    pose_idx = pose_idx % poses.shape[0]
    run_model(viewer)
pano.button(callback=callback_prev_pose, name='Prev Pose')

def callback_next_pose(viewer):
    global pose_idx
    pose_idx += 1
    pose_idx = pose_idx % poses.shape[0]
    run_model(viewer)
pano.button(callback=callback_next_pose, name='Next Pose')

# Call run_model at initialization:
run_model(pano)

pano.hold()