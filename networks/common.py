import torch
import torch.nn as nn
import numpy as np

class PoissonSolver(torch.autograd.Function):
    # Interface with external cholesky solver (Cholespy)
    @staticmethod
    def forward(ctx, solver, rhs: torch.Tensor):
        # Solve Lx = rhs
        ctx.solver = solver
        x = torch.zeros_like(rhs, device=rhs.device, dtype=rhs.dtype)
        solver.solve(rhs, x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        f_grad = None
        if ctx.needs_input_grad[1]:
            grad_output = grad_output.contiguous()
            f_grad = torch.zeros_like(grad_output)
            ctx.solver.solve(grad_output, f_grad)
        del ctx.solver
        return None, f_grad

# ┌────────────────────────┐
#   Common network blocks:
# └────────────────────────┘

class Mlp(nn.Module):
    def __init__(self, in_c, out_c, width, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_c, width),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(width, out_c)
        )
    def forward(self, x):
        return self.layers(x)

class PoissonBlockMLP(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        width,
        act_layer=nn.GELU,
        init_values=1.0,
        drop=0.,
        drop_path=0.,
        norm=True,
        grad_inputs=False,
        extra_feats=0
    ):
        super().__init__()        
        self.norm_pde  = lambda x,mass=None: x # default no norm
        self.norm_grad = lambda x,mass=None: x
        if norm:
            norm_type = 'function'
            norm_c = in_c // 3 if grad_inputs else in_c // 2
            self.norm_pde  = PoissonNetNorm(norm_type, norm_c)
            self.norm_grad = PoissonNetNorm(norm_type, norm_c) if grad_inputs else None

        in_c = in_c + extra_feats
        self.mlp1 = Mlp(in_c, width, width, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(out_c))  # LayerScale
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp2 = Mlp(width, out_c, width, act_layer=act_layer, drop=drop)
        self.ls2 = nn.Parameter(init_values * torch.ones(out_c))
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pde_sol, grad_features=None, extra_features=None, mass=None):
        mlp_in = [x, self.norm_pde(pde_sol, mass)]
        if extra_features is not None:
            mlp_in.append(extra_features)
        if grad_features is not None:
            mlp_in.append(self.norm_grad(grad_features, mass))
        mlp_in = torch.cat(mlp_in, dim=-1)
        
        residual = self.ls1 * self.mlp1(mlp_in)
        residual = self.drop_path1(residual)
        out = x + residual

        residual = self.ls2 * self.mlp2(out)
        residual = self.drop_path2(residual)
        return out + residual

class MlpBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        width,
        act_layer=nn.GELU,
        init_values=1.0,
        drop=0.0,
        drop_path=0.0,
        norm=True
    ):
        super().__init__()
        self.norm = lambda x, mass=None: x # default no norm
        if norm: self.norm = PoissonNetNorm('function', in_c)

        self.mlp1 = Mlp(in_c, out_c, width, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(out_c))  # LayerScale
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, mass=None):
        residual = self.ls1 * self.mlp1(self.norm(x, mass))
        residual = self.drop_path1(residual)
        return self.proj(x) + residual


class PoissonNetNorm(nn.Module):
    # Normalization layer that supports two modes:
    # - 'vertex': normalizes each vertex feature vector separately (i.e. point-wise normalization)
    # - 'function': normalizes each function (channel) defined over the surface separately
    def __init__(self, mode, hidden_size, eps=1e-12, elementwise_affine=True, bias=True):
        super().__init__()
        assert mode in ['vertex', 'function'], f"Unknown mode: {mode}"
        self.mode = mode

        if mode == 'vertex':  
            self.d_ = -1
        elif mode == 'function':
            self.d_ = -2

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.variance_epsilon = eps

    def forward(self, x, mass=None):
        d = self.d_

        # optional discretization-aware normalization, only for 'function' mode:
        if mass is not None and self.mode == 'function': 
            mass_sum = mass.sum(dim=1, keepdim=True)  # (B, 1)
            mass = mass.unsqueeze(-1)  # (B, V, 1)

            # Channel-wise mean / variance (B, 1, C)
            u = (mass * x).sum(dim=1) / (mass_sum + 1e-12)
            s = (mass * (x - u)**2).sum(dim=1) / (mass_sum + 1e-12)
        else:
            u = x.mean(dim=d, keepdim=True)
            s = (x - u).pow(2).mean(dim=d, keepdim=True)

        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if self.weight is not None:
            x = x * self.weight.view(1, 1, -1)
        if self.bias is not None:
            x = x + self.bias.view(1, 1, -1)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        assert 0. <= drop_prob <= 1., f"drop_prob {drop_prob} not in [0, 1]"
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(x.new_full(shape, keep_prob))
        return x * mask / keep_prob

# ┌──────────────────────┐
#   Remapping functions:
# └──────────────────────┘

def vertices_to_faces(x, faces):
    # https://github.com/nmwsharp/diffusion-net
    x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
    faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
    xf = torch.gather(x_gather, 1, faces_gather)
    x_out = torch.mean(xf, dim=-1)
    return x_out

def faces_to_vertices(face_data, faces, face_areas, num_vertices):
    """
    Maps face data (B, F, C) to vertex data (B, V, C) in a discretization-invariant manner.

    face_data: Tensor of shape (B, F, C), signal on each face.
    faces: Tensor of shape (B, F, 3), each row contains vertex indices for a face.
    face_areas: Tensor of shape (B, F), computed face areas.
    num_vertices: Integer, total number of vertices (V).
    
    Returns:
        vertex_data: Tensor of shape (B, V, C) containing area-weighted averages.
    """
    B, F, C = face_data.shape

    # Expand face data to each of the 3 vertices per face.
    # (B, F, 3, C) where each face's data is repeated for each vertex.
    face_data_expanded = face_data.unsqueeze(2).expand(B, F, 3, C)
    face_data_flat = face_data_expanded.reshape(B, F * 3, C)
    
    # Similarly, flatten face indices: (B, F*3)
    faces_flat = faces.reshape(B, F * 3)
    
    # Each face contributes with weight (face_area / 3) to each vertex.
    weights = (face_areas.unsqueeze(-1) / 3).expand(B, F, 3)
    weights_flat = weights.reshape(B, F * 3)
    
    # Allocate accumulation tensors for weighted signals and weights.
    vertex_data = torch.zeros(B, num_vertices, C, device=face_data.device, dtype=face_data.dtype)
    vertex_mass = torch.zeros(B, num_vertices, device=face_data.device, dtype=face_data.dtype)
    
    # Scatter-add the weighted face signals to the corresponding vertices.
    vertex_data = vertex_data.scatter_add(
        dim=1,
        index=faces_flat.unsqueeze(-1).expand(B, F * 3, C),
        src=face_data_flat * weights_flat.unsqueeze(-1)
    )
    
    # Scatter-add the weights to compute total mass per vertex.
    vertex_mass = vertex_mass.scatter_add(
        dim=1,
        index=faces_flat,
        src=weights_flat
    )
    
    # Normalize by the accumulated weights at each vertex.
    vertex_data = vertex_data / vertex_mass.unsqueeze(-1)
    
    return vertex_data

def output_at(x, faces, mass, domain='verts'):
    ''' Remaps input vertex signal `x: (B, V, C)` from vertices to faces or global mean '''

    assert x.ndim == 3, "Input signal must have shape (B, V, C)"
    
    if domain == 'vertices' or domain == 'verts':
        return x
    
    if domain == 'faces': 
        return vertices_to_faces(x, faces)
    
    if domain == 'global_mean': 
        return torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)

    raise ValueError(f"Unknown domain: {domain}")