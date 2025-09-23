import os
import PIL
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image, make_grid
from utils.helpers import to_np
import viz.mesh_render as Render
from matplotlib.colors import to_rgb


def render_segmentation(verts, faces, labels, save_mesh=False):
    verts = to_np(verts)
    faces = to_np(faces)
    labels = to_np(labels)
    colors = np.array(Render.segmentation_colors30)
    ncolors = len(colors)
    label_max = labels.max()
    sample = np.linspace(0, ncolors-1, label_max+1).astype(int) # evenly sample colors for each class
    colors = colors[sample]
    colors = colors.tolist()

    face_colors = np.array([to_rgb(colors[i % len(colors)]) for i in labels])
    render = Render.render_mesh(verts, faces, face_colors=face_colors)

    if save_mesh:
        import trimesh, uuid
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        vert_cols = trimesh.visual.color.face_to_vertex_color(mesh, face_colors)
        mesh.visual.vertex_colors = vert_cols
        mesh.export(f'segmentation_{uuid.uuid4()}.ply')

    return render

def render_mesh(verts, faces, vertex_colors=None, res=512):
    verts = to_np(verts)
    faces = to_np(faces)
    if vertex_colors is None:
        vertex_colors = np.ones_like(verts) * [1.0, 0.5, 0.5]
    render = Render.render_mesh(verts, faces, vertex_colors=vertex_colors, flat_shading=False, res=res)
    return render

def render_overlayed_meshes(vert_list, faces_list):
    all_verts = []
    all_faces = []
    vert_offset = 0

    # use different colors for each mesh:
    color_pal = Render.default_colors
    vertex_colors = []

    for i, (verts, faces) in enumerate(zip(vert_list, faces_list)):
        verts_np = to_np(verts)
        faces_np = to_np(faces)
        
        all_verts.append(verts_np)
        # Add faces to the combined list, but offset the indices
        all_faces.append(faces_np + vert_offset)
        vert_offset += verts_np.shape[0]

        # assume color:
        color = to_rgb(color_pal[i % len(color_pal)])
        v_color = 0.6*np.ones_like(verts_np) * color
        vertex_colors.append(v_color)
    
    combined_verts = np.concatenate(all_verts, axis=0)
    combined_faces = np.concatenate(all_faces, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    return render_mesh(combined_verts, combined_faces, vertex_colors)

def add_text(img, caption, coords=(5, 5), color=(0, 0, 0), text_size=20):
    img_pil = F.to_pil_image(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        default_font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
        font = ImageFont.truetype(default_font_path, text_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text(coords, caption, fill=color, font=font)
    return F.to_tensor(img_pil)

def image_grid(img_list, fp=None):
    '''
    Exports a stack of images as a large square gallery.
    (or as close to square as possible)

    `img_list`: (N, C, H, W)
    '''
    if isinstance(img_list, list):
        # check if all images have the same shape
        if not all([img.shape == img_list[0].shape for img in img_list]):
            # fix this by resizing all images to the same shape
            max_shape = max([img.shape[-1] for img in img_list])
            img_list = [torch.nn.functional.interpolate(img[None], size=(max_shape, max_shape), mode='bilinear', align_corners=False)[0] for img in img_list]

        img_list = torch.stack(img_list, dim=0)
        
    nrows = int(math.sqrt(len(img_list)))
    grid = make_grid(img_list, nrow=nrows)
    if fp is not None:
        save_image(grid, fp)
    return grid

def mpl_to_rgb(fig, close=True):
    '''
    Renders a matplotlib figure directly to an RGB buffer
    in (C, H, W) convention.
    '''
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if close:
        plt.close(fig)
    return torch.tensor(np.transpose(data, axes=(2,0,1))).float() / 255.
