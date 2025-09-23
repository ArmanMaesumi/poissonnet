import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PoissonSolver, PoissonBlockMLP, output_at, vertices_to_faces, faces_to_vertices

class NJFHead(nn.Module):
    def __init__(self, in_c, out_c, width, config={}):
        super().__init__()
        cmlp_nlayers = config.get('cmlp_nlayers', 2)
        cmlp_modulate = config.get('cmlp_modulate', False)
        self.grad_mlp = ComplexMLP(in_c=in_c, out_c=out_c, width=width, num_layers=cmlp_nlayers, modulate=cmlp_modulate)

    def forward(self, x_in, M, G, solver, faces, vertex_mass, original_grads=None, **kwargs):
        B, V, C = x_in.shape
        F = M.shape[1] // 2

        grads_in = torch.bmm(G, x_in) # (B, 2F, C)
        x_faces = vertices_to_faces(x_in, faces) # (B, F, C)
        grads = self.grad_mlp(grads_in, x_faces) # (B, 2F, C)
        if original_grads is not None:
            grads = grads + original_grads

        # Solve Poisson equation Lu = ∇^T @ face_areas * grads
        rhs = torch.bmm(G.transpose(1, 2), M.unsqueeze(-1) * grads) # (B, V, C)
        u = torch.empty_like(rhs)
        for b in range(B):
            u[b] = PoissonSolver.apply(solver[b], rhs[b].contiguous())
        
        # nullify area-weighted mean:
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)
        return u, grads

class ComplexRotationScale(nn.Module):
    '''
    Modulation layer that applies a per-channel rotation and scale 
    to complex features that is dependent on scalar features x.
    '''
    def __init__(self, in_c, out_c):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(in_c, in_c),
            nn.GELU(),
            nn.Linear(in_c, 2*out_c), # output interpretted as [phase, scale]
        )
        self.scale_softplus = nn.Softplus()

    def forward(self, x, f_real, f_imag):
        # x: (B, F, C) scalar features averaged onto faces
        # f_real, f_imag: (B, F, C) real and imaginary parts of vector features.

        phase, scale = self.modulator(x).chunk(2, dim=-1)
        scale = self.scale_softplus(scale) + 1e-8 # map scale into positive values
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        # Rotate by phase, and apply scale
        real_out = (f_real * cos - f_imag * sin) * scale
        imag_out = (f_real * sin + f_imag * cos) * scale
        return real_out, imag_out

class ComplexLayer(nn.Module):
    """
    A complex-valued linear map (C x C), plus an optional magnitude-based nonlinearity.
    Operation:
        z_out := W @ z_in ,  where W is CxC complex mat.
        Then optionally apply:  z_out <- GELU(|z_out| + mag_bias) * (z_out / |z_out|).
    """
    def __init__(self, in_c, out_c, nonlin=True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.nonlin = nonlin

        # Real and imaginary parts of complex weight matrix
        self.lin_real = nn.Linear(in_c, out_c, bias=False)
        self.lin_imag = nn.Linear(in_c, out_c, bias=False)

        if self.nonlin:
            self.mag_bias = nn.Parameter(torch.zeros(out_c), requires_grad=True)
            self.gelu = nn.GELU()

    def forward(self, f_real, f_imag):
        # f_real, f_imag: (B, F, C) real and imaginary parts of vector features.

        y_real = self.lin_real(f_real) - self.lin_imag(f_imag) # (B, F, C)
        y_imag = self.lin_real(f_imag) + self.lin_imag(f_real) # (B, F, C)

        if self.nonlin:
            r = torch.sqrt(y_real**2 + y_imag**2 + 1e-8)
            r_activated = self.gelu(r + self.mag_bias)
            scale = r_activated / (r + 1e-8)
            y_real = y_real * scale
            y_imag = y_imag * scale

        return y_real, y_imag

class ComplexMLP(nn.Module):
    def __init__(self, in_c, out_c, width, num_layers=3, modulate=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.layers = nn.ModuleList()

        for i in range(num_layers-1):
            self.layers.append(ComplexLayer(in_c if i == 0 else width, width, nonlin=True))
        self.layers.append(ComplexLayer(width, out_c, nonlin=False))
        
        self.modulator = None
        if modulate:
            self.modulator = ComplexRotationScale(in_c, in_c)

    def forward(self, f, x=None):
        """
        f: (B, 2F, C) with interleaved real/imag convention, OR (f_real, f_imag) tuple
        x: (B, F, C) scalar face-based features, used for modulation layer if enabled
        Returns: (B, 2F, C) interleaved transformed vector features
        """
        if isinstance(f, tuple):
            f_real, f_imag = f
        else:
            f_real = f[:, 0::2, :]
            f_imag = f[:, 1::2, :]
        B_, F_, C_ = f_real.shape

        out_re, out_im = f_real, f_imag

        if self.modulator is not None:
            out_re, out_im = self.modulator(x, out_re, out_im)

        for layer in self.layers:
            out_re, out_im = layer(out_re, out_im)

        out = torch.empty(B_, 2*F_, self.out_c, device=f_real.device)
        out[:, 0::2, :] = out_re
        out[:, 1::2, :] = out_im
        return out

class PoissonBlock(nn.Module):
    def __init__(self, in_c, out_c, width, extra_feats=0, config={}):
        super().__init__()
        drop_path = config.get('drop_path', 0.0)
        dropout_p = config.get('dropout_p', 0.0)
        cmlp_nlayers = config.get('cmlp_nlayers', 2)
        mlp_norm = config.get('mlp_norm', False)
        self.mass_norm = config.get('mass_norm', False)
        self.inner_prod_features = config.get('inner_prod_features', False)
        self.cmlp_modulate = config.get('cmlp_modulate', True)

        self.grad_mlp = ComplexMLP(in_c=in_c, out_c=out_c, width=width, num_layers=cmlp_nlayers, modulate=self.cmlp_modulate)

        mlp_in = in_c + out_c # [x_in, pde_sol]
        if self.inner_prod_features: # optional
            self.grad_features = ComplexLayer(in_c=out_c, out_c=out_c, nonlin=False)
            self.grad_scaler = nn.Parameter(1e-2 * torch.ones(out_c), requires_grad=True)
            mlp_in += out_c
            
        self.vert_mlp = PoissonBlockMLP(in_c=mlp_in, out_c=out_c, width=width, drop_path=drop_path, drop=dropout_p, grad_inputs=self.inner_prod_features, norm=mlp_norm, extra_feats=extra_feats)

    def forward(self, x_in, M, G, solver, faces, vertex_mass, extra_features=None, **kwargs):
        '''
        - x_in:             (B, V, C)   scalar vertex features
        - M:                (B, 2F)     interleaved face areas [A0, A0, A1, A1, ...]
        - G:                (B, 2F, V)  intrinsic gradient operator
        - solver:           [B,]        list of solver objects
        - faces:            (B, F, 3)   face indices
        - vertex_mass:      (B, V)      lumped vertex masses
        - extra_features:   (B, V, C)   additional features to be concatenated to the input of the MLP

        1. Computes transformed gradient features:
                grads = VectorMLP(∇x_in, x_in)
        2. Solve Poisson equation: 
                Lu = ∇ ⋅ (M * grads)
        3. Compute new vertex features:
                out = MLP([x_in, u]) + x_in
        See Fig. 2 in the paper for a diagram of the block.
        '''
        B, V, C = x_in.shape
        F = M.shape[1] // 2

        grads_in = torch.bmm(G, x_in) # (B, 2F, C)
        x_face = vertices_to_faces(x_in, faces) # (B, F, C)
        grads = self.grad_mlp(grads_in, x_face) # (B, 2F, C)

        # Solve Poisson equation Lu = ∇^T @ face_areas * grads
        rhs = torch.bmm(G.transpose(1, 2), M.unsqueeze(-1) * grads) # (B, V, C)
        u = torch.empty_like(rhs) # container for poisson solve
        for b in range(B):
            # solver only accepts 128 simultaneous solves, so split channels if needed -- these for loops should be cheap
            if rhs.shape[-1] > 128:
                for j in range(0, rhs.shape[-1], 128):
                    u[b][:, j:j+128] = PoissonSolver.apply(solver[b], rhs[b][:, j:j+128].contiguous())
            else:
                u[b] = PoissonSolver.apply(solver[b], rhs[b].contiguous())

        # nullify area-weighted mean:
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)

        # Optionally compute inner products of transformed gradient features -- see DiffusionNet inner-product features https://github.com/nmwsharp/diffusion-net
        gradient_features = None
        if self.inner_prod_features:
            ginX, ginY = grads_in[:, 0::2, :], grads_in[:, 1::2, :]
            gX, gY = grads[:, 0::2, :], grads[:, 1::2, :]
            gX, gY = self.grad_features(gX, gY)
            inner_prod = gX * ginX + gY * ginY # (B, F, C)
            face_areas = M[:, 0::2] # (B, F)
            gradient_features = faces_to_vertices(inner_prod, faces, face_areas, num_vertices=V) # (B,V,C)
            gradient_features = torch.tanh(gradient_features * self.grad_scaler)

        out = self.vert_mlp(x_in, 
                            u, 
                            gradient_features, 
                            extra_features=extra_features, 
                            mass=vertex_mass if self.mass_norm else None)

        return out, grads
    
class PoissonNet(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, n_blocks=4, head=None, extra_features=0, outputs_at='vertices', last_activation=nn.Identity(), config={}, **kwargs):   
        super().__init__()
        assert head in ['linear', 'mlp', 'njf', None], f"Invalid head type: {head}. Choose from ['linear', 'mlp', 'njf', None]."
        self.outputs_at = outputs_at
        self.last_act = last_activation

        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = n_blocks
        self.head_type = head

        if extra_features > 0:
            # Extra MLP to process auxiliary features:
            self.extra_feat_mlp = nn.Sequential(
                nn.Linear(extra_features, C_width),
                nn.GELU(),
                nn.Linear(C_width, C_width),
            )
            extra_features = C_width

        self.proj_in = nn.Linear(C_in, C_width)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(PoissonBlock(in_c=C_width, out_c=C_width, width=C_width, extra_feats=extra_features, config=config))

        self.head = nn.Identity()
        if head == 'njf':
            self.head = NJFHead(in_c=C_width, out_c=3, width=C_width, config=config)
        elif head == 'linear':
            self.head = nn.Linear(C_width, C_out)
        else:
            raise ValueError(f"Invalid head type: {head}. Choose from ['njf', 'linear'].")

    def forward(self, x_in, M, G, solver, faces, vertex_mass, extra_features=None, **kwargs):
        '''
        - x_in:             (B, V, C)   input scalar vertex features
        - M:                (B, 2F)     interleaved face areas [A0, A0, A1, A1, ...]
        - G:                (B, 2F, V)  intrinsic gradient operator
        - solver:           [B,]        list of solver objects
        - faces:            (B, F, 3)   face indices
        - vertex_mass:      (B, V)      lumped vertex masses
        - extra_features:   (B, V, C)   additional features to be concatenated to the input of block MLPs
        '''
        if self.head_type == 'njf': # compute (B, 2F, 3) input gradients used for njf head
            with torch.no_grad():
                grad_in = torch.bmm(G, x_in)

        if extra_features is not None:
            extra_features = self.extra_feat_mlp(extra_features)
            if extra_features.ndim == 2: # Assume (B, C)
                extra_features = extra_features.unsqueeze(1).expand(-1, x_in.shape[1], -1)

        x = self.proj_in(x_in) # (B, V, C_width)
        for block in self.blocks:
            x, _ = block(x, M, G, solver, faces, vertex_mass, extra_features=extra_features, **kwargs)

        if self.head_type == 'njf':
            out, grads = self.head(x, M, G, solver, faces, vertex_mass, original_grads=grad_in)
            return out, grads
        
        if self.head_type == 'linear':
            out = self.head(x)
        elif self.head_type == 'mlp':
            out = self.head_out(self.head(x))
        
        out = output_at(out, faces, vertex_mass, domain=self.outputs_at) # remap signal to verts/faces/global mean
        out = self.last_act(out)
        return out