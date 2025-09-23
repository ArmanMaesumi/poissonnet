conda create -n "poissonnet" python==3.12
conda activate poissonnet
pip3 install torch torchvision
pip3 install pyrender libigl cholespy einops scipy matplotlib tqdm trimesh pillow
pip3 install panopti
pip3 install smplx[all]

git clone https://github.com/ArmanMaesumi/torch_mesh_ops
cd torch_mesh_ops
python setup.py install
cd ..
rm -rf torch_mesh_ops