# SHREC11 Classification

First download the simplified SHREC11 meshes from [https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz](https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz), provided by the authors of MeshCNN. Move the archive to `experiments/shrec11_classification/shrec_16.tar.gz` and run 

```bash
tar -xzvf ./shrec_16.tar.gz
```

You should now have a directory `shrec_16` in located at `experiments/shrec11_classification/shrec16/...`.

Finally, you may run the training script with

```bash
python -m experiments.shrec11_classification.trainer
```

Results and visualizations will be saved to `results/shrec11_classification/`