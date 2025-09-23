import os
import torch
from torch.utils.data import Dataset

class MOYOBakedDataset(Dataset):
    def __init__(self, data_dir, train, config):
        data_pack = 'train_baked_hands_16000_5.0.pt' if train else 'val_baked_hands_2000_5.0.pt'
        data_pack = torch.load(os.path.join(data_dir, data_pack))
        self.src_verts = data_pack['src_verts']
        self.tar_verts = data_pack['tar_verts']
        self.tar_poses = data_pack['tar_poses']
        self.faces = data_pack['faces']
        self.num_pose_params = self.tar_poses.shape[-1] # 153

    def __len__(self):
        return len(self.src_verts)
    
    def __getitem__(self, idx):
        return self.src_verts[idx], self.tar_verts[idx], self.faces, self.tar_poses[idx]
