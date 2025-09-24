uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install pyrender libigl cholespy einops scipy matplotlib tqdm trimesh pillow panopti 'smplx[all]'
uv pip install --no-build-isolation git+https://github.com/ArmanMaesumi/torch_mesh_ops