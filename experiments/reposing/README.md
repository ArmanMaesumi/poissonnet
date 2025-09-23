## Training

For training, please first follow the instructions in `/smplx_data/README.md` to create the SMPL-X + MOYO reposing dataset used in our paper. 

Once the dataset is created, you can run training from the PoissonNet root directory via:

```bash
python -m experiments.reposing.trainer
```

All relevant hyperparameters are in `config.json`

## Interactive demo

We provide an interactive viewer that lets you cycle through pose variations of a handful of characters. Interactive demos are created using [Panopti](https://github.com/ArmanMaesumi/panopti). With Panopti installed, simply launch a server using, e.g.:
```bash
python -m panopti.run_server --host localhost --port 8080
```
From the project root you can then run our demo script with a separate terminal:
```bash
python -m experiments.reposing.test
```
The viewer will automatically load `obj` meshes located in `demo_meshes/`, and it will cycle through several poses contained in `demo_meshes/example_poses.npy`.