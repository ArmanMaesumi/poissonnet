import igl
import math
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

# ================= Transformations =================

def normalize_mesh(v, f, mode='unit_sphere', discretization_aware=True, return_scale_shift=False):
    '''
    Normalizes mesh vertices using one of the following modes:
    - 'unit_sphere': centers the mesh and scales it to fit in unit sphere
    - 'surface_area': centers the mesh and scales it to have unit surface area

    `discretization_aware=True` optionally shifts the mesh using proper area-weighted centroid of the mesh
    '''
    eps = 1e-8

    use_torch = torch.is_tensor(v)
    if use_torch:
        device, dtype = v.device, v.dtype
        v_np = v.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
    else:
        v_np, f_np = v, f

    # compute (area-weighted) centroid in NumPy
    if discretization_aware or mode == 'surface_area':
        areas = igl.doublearea(v_np, f_np) * 0.5
        total_area = areas.sum()
        tri_centers = v_np[f_np].mean(axis=1)
        centroid_np = (areas[:,None] * tri_centers).sum(axis=0) / total_area
    else:
        centroid_np = v_np.mean(axis=0)

    v_centered_np = v_np - centroid_np

    if mode == 'unit_sphere':
        # max distance from origin
        dists = np.linalg.norm(v_centered_np, axis=1)
        scale_np = dists.max()
    elif mode == 'surface_area':
        scale_np = np.sqrt(total_area) + eps
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    v_out = v_centered_np / scale_np

    if use_torch:
        v_out = torch.from_numpy(v_out).to(device=device, dtype=dtype)
    
    if return_scale_shift:
        return v_out, scale_np, centroid_np
    
    return v_out

def normalize_trimesh(mesh, mode='unit_sphere', discretization_aware=True):
    '''
    Normalizes trimesh mesh vertices using one of the following modes:
    - 'unit_sphere': centers the mesh and scales it to fit in unit sphere
    - 'surface_area': centers the mesh and scales it to have unit surface area

    `discretization_aware` optionally shifts the mesh using proper centroid of the mesh

    Returns a *new* normalized trimesh object.
    '''
    _, scale, shift = normalize_mesh(mesh.vertices, mesh.faces, mode=mode, discretization_aware=discretization_aware, return_scale_shift=True)
    transform_mat = np.eye(4)
    transform_mat[:3, :3] /= scale
    transform_mat[:3, 3] = -shift / scale
    normalized_mesh = mesh.copy()
    normalized_mesh.apply_transform(transform_mat)
    return normalized_mesh

def random_rotate_points_batched(pts, randgen=None, return_rot=False):
    R = random_rotation_matrix_batched(pts.shape[0], generator=randgen, device=pts.device, dtype=pts.dtype)
    if return_rot:
        return torch.matmul(pts, R), R
    return torch.matmul(pts, R)

def random_rotation_matrix_batched(batch_size, generator=None, device=None, dtype=torch.float):
    if device is None:
        device = torch.device('cpu')

    # Sample three independent uniform random numbers per batch element.
    rand = torch.rand(batch_size, 3, device=device, dtype=dtype, generator=generator)
    theta = rand[:, 0] * (2 * math.pi)  # Rotation about the Z-axis.
    phi   = rand[:, 1] * (2 * math.pi)  # Direction for pole deflection.
    z     = rand[:, 2] * 2.0            # Magnitude of pole deflection.

    r = torch.sqrt(z)
    Vx = torch.sin(phi) * r
    Vy = torch.cos(phi) * r
    Vz = torch.sqrt(2.0 - z)
    V = torch.stack([Vx, Vy, Vz], dim=1)  # (batch_size, 3)

    st = torch.sin(theta)
    ct = torch.cos(theta)

    # Build batch of rotation matrices about the Z-axis.
    R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = ct
    R[:, 0, 1] = st
    R[:, 1, 0] = -st
    R[:, 1, 1] = ct
    R[:, 2, 2] = 1.0

    # Compute the Householder reflection matrix for each batch element.
    # Outer product V*V^T for each vector.
    outer = V.unsqueeze(2) * V.unsqueeze(1)  # (batch_size, 3, 3)
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    # Compute (V V^T - I) for each element, then multiply by R.
    M = (outer - I) @ R  # (batch_size, 3, 3)

    return M

def rotate_point_cloud(pc, rot):
    '''
    Rotates point cloud using rotation matrix
    '''
    if isinstance(rot, np.ndarray):
        rot = torch.from_numpy(rot).float().to(pc.device)
    
    return pc @ rot.T

# ================= Augmentations =================

def shuffle_point_cloud(pc):
    '''
    Shuffles batch of point clouds
    '''
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().numpy()
    np.random.shuffle(pc)
    if isinstance(pc, torch.Tensor):
        pc = torch.from_numpy(pc).to(pc.device)
    return pc

def random_point_cloud_rotation(pc):
    '''
    Randomly rotates point cloud
    '''
    rot = R.random().as_matrix()
    if isinstance(pc, torch.Tensor):
        rot = torch.from_numpy(rot).float().to(pc.device)
    pc = rotate_point_cloud(pc, rot)
    return pc, rot
