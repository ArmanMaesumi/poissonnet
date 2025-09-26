import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb
from torchvision.utils import make_grid, save_image

# os.environ['PYOPENGL_PLATFORM'] = 'egl' # you may need to use this for headless rendering
import trimesh
import pyrender

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from utils.helpers import to_np

default_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'aqua', 'navy', 'black']

seg_colors8 = [(0.122, 0.467, 0.706),
                (1.000, 0.498, 0.055),
                (0.173, 0.627, 0.173),
                (0.839, 0.153, 0.157),
                (0.580, 0.404, 0.741),
                (0.549, 0.337, 0.294),
                (0.890, 0.467, 0.761),
                (0.498, 0.498, 0.498)]

segmentation_colors10 = plt.get_cmap('tab10').colors
segmentation_colors20 = plt.get_cmap('tab20').colors
segmentation_colors30 = [(0.968, 0.441, 0.536),
                        (0.969, 0.454, 0.408),
                        (0.954, 0.478, 0.196),
                        (0.862, 0.536, 0.195),
                        (0.793, 0.571, 0.195),
                        (0.735, 0.595, 0.194),
                        (0.68, 0.615, 0.194),
                        (0.623, 0.633, 0.194),
                        (0.557, 0.651, 0.193),
                        (0.468, 0.67, 0.193),
                        (0.313, 0.693, 0.192),
                        (0.196, 0.697, 0.361),
                        (0.201, 0.691, 0.48),
                        (0.205, 0.686, 0.549),
                        (0.208, 0.681, 0.6),
                        (0.21, 0.677, 0.643),
                        (0.213, 0.673, 0.684),
                        (0.216, 0.668, 0.726),
                        (0.22, 0.663, 0.773),
                        (0.225, 0.654, 0.834),
                        (0.233, 0.64, 0.926),
                        (0.433, 0.607, 0.959),
                        (0.583, 0.57, 0.958),
                        (0.697, 0.528, 0.958),
                        (0.8, 0.477, 0.958),
                        (0.908, 0.402, 0.958),
                        # (0.96, 0.375, 0.893),
                        # (0.962, 0.398, 0.801),
                        # (0.964, 0.414, 0.719),
                        # (0.966, 0.428, 0.637)
                        ]

def render_mesh(verts, faces, face_colors=None, vertex_colors=None, rot_mat=None, flat_shading=True, res=256, filename=None):
    mesh = trimesh.Trimesh(vertices=to_np(verts), faces=to_np(faces), process=False)
    
    mesh.face_colors = np.ones_like(mesh.faces) * 255
    if rot_mat is not None:
        mesh.vertices = mesh.vertices @ rot_mat.T
    
    if face_colors is not None:    
        if face_colors.dtype == np.int64 or face_colors.dtype == np.int32:
            face_colors = np.array([to_rgb(default_colors[i % len(default_colors)]) for i in face_colors])
        mesh.visual.face_colors = face_colors
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors

    flags = pyrender.constants.RenderFlags.OFFSCREEN | pyrender.constants.RenderFlags.SKIP_CULL_FACES
    if flat_shading:
        flags |= pyrender.constants.RenderFlags.FLAT

    camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
    camera = pyrender.Node(camera=camera, translation=[0,0.0,1.15], rotation=[0, 0, 0, 1])
    light = pyrender.SpotLight(color=np.ones(3), intensity=8.0)
    scene = pyrender.Scene(bg_color=[1., 1., 1.])
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(light, pose=camera.matrix)
    scene.add_node(camera)
    scene.add(mesh, pose=np.eye(4))

    r = pyrender.OffscreenRenderer(res, res)
    color, _ = r.render(scene, flags=flags)
    color = color.astype(np.float32) / 255.
    color = torch.from_numpy(color).permute(2, 0, 1)

    if filename is not None:
        save_image(color, filename)

    return color
