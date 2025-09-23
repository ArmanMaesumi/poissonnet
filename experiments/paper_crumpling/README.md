## Crumpling paper mesh:
To replicate the crumpling paper experiment from the original paper, you first need to buy the source mesh from TurboSquid: [https://www.turbosquid.com/3d-models/animated-paper-crumpling-1794996](https://www.turbosquid.com/3d-models/animated-paper-crumpling-1794996). Due to licensing, we cannot release the source geometry for free.

## Converting .abc to .npz
To prepare the data, we first need to convert the Alembic animation file to a NumPy array containing the (temporal) vertex tensor and face tensor. You may run the Blender script below to do this conversion. Simply pass the Alembic file from the TurboSquid asset through this script, and save the resulting `.npz` file to `experiments/paper_crumpling/crumpling_animation.npz`. Then you are ready to train:
```bash
python -m experiments.paper_crumpling.trainer
```

### Blender script:
```bash
blender -b --python export_abc_to_npz.py -- \
        --abc /path/to/animation.abc \
        --out /path/to/animation.npz
```

```python
import argparse, sys, bpy, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--abc", required=True)
parser.add_argument("--out", required=True)
args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])

bpy.ops.wm.read_factory_settings(use_empty=True)

# import alembic file:
bpy.ops.wm.alembic_import(
        filepath=args.abc,
        set_frame_range=True,
        validate_meshes=True)

# everything from the Alembic file is now in the current collection
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if len(objs) != 1:
    raise RuntimeError("script assumes exactly one mesh object")
obj = objs[0]

# triangulate once (if mesh is a quad mesh)
tri_mod = obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
tri_mod.quad_method = 'SHORTEST_DIAGONAL'
tri_mod.ngon_method = 'BEAUTY'

depsgraph = bpy.context.evaluated_depsgraph_get()

scene = bpy.context.scene
start, end = scene.frame_start, scene.frame_end
n_frames = end - start + 1

# use frame 0 to grab constant topology (faces)
scene.frame_set(start)
mesh_eval = obj.evaluated_get(depsgraph).to_mesh()
faces = np.asarray([[v for v in p.vertices] for p in mesh_eval.polygons],
                   dtype=np.int32)
vcount = len(mesh_eval.vertices)
obj.to_mesh_clear()

verts = np.empty((n_frames, vcount, 3), dtype=np.float32)
for i, f in enumerate(range(start, end + 1)):
    scene.frame_set(f)
    mesh_eval = obj.evaluated_get(depsgraph).to_mesh()
    verts[i, :, :] = np.asarray([v.co[:] for v in mesh_eval.vertices],
                                dtype=np.float32)
    obj.to_mesh_clear()

np.savez_compressed(args.out, verts=verts, faces=faces)
```