This experiment reuses the same dataset as in `reposing`. If you'd like to train from scratch, please first follow the instructions in `/smplx_data/README.md` to create the SMPL-X + MOYO reposing dataset used in our paper. 

Once the dataset is created, you can run training from the PoissonNet root directory via:

```bash
python -m experiments.segmentation.trainer
```

All relevant hyperparameters are in `config.json`