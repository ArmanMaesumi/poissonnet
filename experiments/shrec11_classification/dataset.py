import os
import torch
import igl
import numpy as np
import geometry.point_utils as Point
from torch.utils.data import Dataset

"""
Simplified version of the Shrec11 dataset from:
https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz

Download `shrec_16.tar.gz` into `./experiments/shrec11_classification/`
and run `tar -xzvf ./shrec_16.tar.gz` to extract the dataset.
"""
class Shrec11MeshDataset_Simplified(Dataset):
    def __init__(self, train, config, entry_dict=None):
        self.train = train
        self.root_dir = './experiments/shrec11_classification/shrec_16'
        self.n_class = 30 
        self.split_size = 0.5 # take 50% of each class for training

        self.class_names = [ 'alien', 'ants', 'armadillo', 'bird1', 'bird2', 'camel', 'cat', 'centaur', 'dinosaur', 'dino_ske', 'dog1', 'dog2', 'flamingo', 'glasses', 'gorilla', 'hand', 'horse', 'lamp', 'laptop', 'man', 'myScissor', 'octopus', 'pliers', 'rabbit', 'santa', 'shark', 'snake', 'spiders', 'two_balls', 'woman']
        self.entries = {}

        self.verts_list = []
        self.faces_list = []
        self.normals_list = []
        self.labels_list = []

        for class_idx, class_name in enumerate(self.class_names):
            if self.train:
                # Get all mesh files for this class:
                mesh_files = []
                for t in ['test', 'train']:
                    files = os.listdir(os.path.join(self.root_dir, class_name, t))
                    for f in files:
                        mesh_files.append(os.path.join(self.root_dir, class_name, t, f))

                # Keep random subset for training (according to split_size):
                num_train = int(len(mesh_files) * self.split_size)
                train_idx = np.random.permutation(len(mesh_files))[:num_train]
                train_files = [mesh_files[i] for i in train_idx]
                test_files = [mesh_files[i] for i in range(len(mesh_files)) if i not in train_idx]
                self.entries[class_name] = {
                    'train': train_files,
                    'test': test_files
                }
                load_files = train_files
                print(f"[Train] {class_name}: {len(train_files)}")
                print(f"[Test] {class_name}: {len(test_files)}")
            else:
                # test data is already set aside by train dataset in dict:
                load_files = entry_dict[class_name]['test']

            # Load meshes:
            for f in load_files:
                verts, faces = igl.read_triangle_mesh(f)
                verts = Point.normalize_mesh(verts, faces, mode='surface_area', discretization_aware=True)
                verts = torch.tensor(verts).float()
                faces = torch.tensor(faces)
                self.verts_list.append(verts)
                self.faces_list.append(faces)
                self.labels_list.append(torch.tensor(class_idx))


    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.labels_list[idx]