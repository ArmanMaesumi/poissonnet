import tqdm
import torch
import numpy as np
import geometry.point_utils as Point
from torch.utils.data import Dataset

class CrumpledPaperDataset(Dataset):
    def __init__(self, train, config):
        meshes = np.load('./experiments/paper_crumpling/crumpling_animation.npz')
        nframes = len(meshes['verts'])
        verts = meshes['verts']
        faces = meshes['faces']
        print('Loaded crumpling paper data:', verts.shape, faces.shape)

        # hardcode time values:
        self.t_vals = torch.linspace(0, 1, nframes).unsqueeze(1)

        for i in tqdm.tqdm(range(len(verts))):
            v = verts[i]
            v = Point.normalize_unit_surface_area(v, faces)
            v = v - np.mean(v, axis=0)
            verts[i] = v # update in place

        self.verts = torch.from_numpy(verts).float()
        self.faces = torch.from_numpy(faces).long()

    def __len__(self):
        return len(self.verts)

    def __getitem__(self, idx):
        return self.verts[idx], self.faces, self.t_vals[idx]
    
    def get_source_cloth(self):
        # the *last* frame of the sequence is the "flat" piece of paper:
        verts = self.verts[-1]
        return verts, self.faces

