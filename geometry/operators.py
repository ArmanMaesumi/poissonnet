import torch
import cholespy
import torch_mesh_ops as TMO

@torch.no_grad()
def construct_mesh_operators(V, F, high_precision=False):
    '''
    Creates the following operators for a mesh. Uses PyTorch CUDA extension.
    - vertex_mass:  (B, V)      lumped vertex masses
    - solver:       [B,]        list of Cholesky solvers for mesh's cotangent Laplacian
    - G:            (B, 2F, V)  face-based intrinsic gradient operator
    - M:            (B, 2F)     interleaved face areas

    Optionally set high_precision to True to use double precision in intermediate
    computations. Note the operators will be returned in float precision regardless.
    '''
    # ensure contiguous memory layout before calling CUDA extension
    V = V.contiguous()
    F = F.contiguous()
    if V.ndim == 2:
        V = V.unsqueeze(0)
        F = F.unsqueeze(0)
            
    if high_precision:
        V = V.double()

    nB, nV, _ = V.shape
    device = V.device

    vert_mass       = TMO.vertex_mass_batched(V, F, 1e-8)                     # (b, v)
    face_areas      = TMO.face_areas_batched(V, F)                            # (b, f)
    edge_lengths    = TMO.edge_lengths_batched(V, F)                          # (b, f, 3)
    G               = TMO.intrinsic_gradient_batched(edge_lengths, F)         # (b, 2f, v)

    vert_mass = vert_mass.float()
    face_areas = face_areas.float()
    G = G.float()
    M = torch.repeat_interleave(face_areas, 2, dim=-1) # grad operator is interleaved convention, so M is interleaved too

    solvers = []
    Ls = TMO.cotangent_laplacian_batched(V, F, 1e-10).float() # (b, v, v)

    # Create Cholesky solver for each Laplacian in batch
    # This could be block-diagonalized, but speedup seems marginal
    for bi in range(nB):
        L = Ls[bi]
        eps = 1e-6
        sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
        L = L + sparse_eps_diag # (b, v, v)
        
        nretry = 0
        while nretry < 5:
            try:
                nrows = L.shape[-1]
                ii = L._indices()[0]
                jj = L._indices()[1]
                x = L._values()
                solver = cholespy.CholeskySolverF(
                    n_rows=nrows, ii=ii, jj=jj,
                    x=x, type=cholespy.MatrixType.COO
                )
                solvers.append(solver)
                break
            except Exception as e:
                eps = eps * 10.0
                sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
                L = L + sparse_eps_diag
                nretry += 1
        if nretry >= 5:
            raise RuntimeError(f"Failed to create Cholesky solver for mesh {bi} in batch")
    return vert_mass, solvers, G, M
