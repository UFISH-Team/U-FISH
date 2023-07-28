# U-FISH 🎣

Unified, U-Net based, deep learning method for FISH spot detection, trained on diverse datasets.

The underlying concept of our method, U-FISH, acknowledges the significant variations in image background and signal spot features due to differences in experimental conditions, hybridization targets, and imaging parameters across various data sources. A single algorithm or parameter setting often falls short in accommodating all these image types. Therefore, we introduce an image-to-image U-Net model to enhance input images by suppressing background noise, eliminating post-enhancement noise, and normalizing the size of signal spots. We then apply conventional methods for spot detection on these enhanced images, thus achieving higher accuracy.

## TODO List

- [ ] API
- [ ] Utils
    + [ ] Plotting tool for show TP, FP, FN
- [ ] CLI
- [ ] Napari plugin
- [ ] Add more datasets
    + [ ] MER-FISH
    + [ ] seqFISH
    + [ ] ExSeq
- [ ] Try other signal footprint
    + [ ] Gaussian
    + [ ] Other shape
- [ ] Benchmark
- [ ] 3D integration method
- [ ] Support for zarr format
- [ ] Upload to BioImageIO model zoo
- [ ] Documentation

## Usage

```bash
pip install u-fish
```

## Dataset

The dataset is available at [HuggingFace Datasets](https://huggingface.co/datasets/GangCaoLab/FISH_spots):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/GangCaoLab/FISH_spots
```
