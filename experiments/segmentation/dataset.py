import os
import igl
import json
import torch
import random
import numpy as np
import geometry.point_utils as Point
from torch.utils.data import Dataset

class MOYOSegmentationDataset(Dataset):
    def __init__(self, data_dir, train, config):
        self.train = train

        data_pack = 'train_baked_hands_16000_5.0.pt' if train else 'val_baked_hands_2000_5.0.pt'
        data_pack = torch.load(os.path.join(data_dir, data_pack))
        src_verts = data_pack['src_verts']
        tar_verts = data_pack['tar_verts']

        # merge into one big array, this experiment doesnt care about src/target
        self.verts = torch.cat([src_verts, tar_verts], axis=0) # (N, V, 3)
        self.faces = data_pack['faces']
        
        # load semantic labels and preprocess them for our needs:
        with open('./smplx_data/smplx_vertex_segmentation.json', 'r') as f:
            smplx_seg = json.load(f)
            self.n_class = len(smplx_seg.keys())
            self.labels = np.zeros((self.verts.shape[1],), dtype=np.int32) 

            # in our dataset we deleted: [leftEye,rightEye,eyeballs], so the indices need to be shifted accordingly.
            leftEye_indices = smplx_seg['leftEye']
            rightEye_indices = smplx_seg['rightEye']
            eyeballs_indices = smplx_seg['eyeballs']
            
            # shift depending on which vertex groups come before the current range:
            for i, k in enumerate(smplx_seg.keys()):
                if k in ['leftEye', 'rightEye', 'eyeballs']:
                    continue
                v_indices = smplx_seg[k]
                shift = 0
                for s in [leftEye_indices, rightEye_indices, eyeballs_indices]:
                    if s[0] < v_indices[0]:
                        shift += len(s)
                v_indices = np.array(v_indices) - shift
                self.labels[v_indices] = i
        self.labels = torch.from_numpy(self.labels).long() # (V,) vertex-based semantic labels

        # Finally, move labels to faces using majority vote per-face
        vlabels = self.labels[self.faces]
        self.labels = torch.mode(vlabels, dim=1).values.long() # (F,)

    def __len__(self):
        return len(self.verts)
    
    def __getitem__(self, idx):
        return self.verts[idx], self.faces, self.labels
