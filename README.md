# U-FISH 🎣

Unified, U-Net based, deep learning method for FISH spot detection, trained on diverse datasets.

The underlying concept of our method, U-FISH, acknowledges the significant variations in image background and signal spot features due to differences in experimental conditions, hybridization targets, and imaging parameters across various data sources. A single algorithm or parameter setting often falls short in accommodating all these image types. Therefore, we introduce an image-to-image U-Net model to enhance input images by suppressing background noise, eliminating post-enhancement noise, and normalizing the size of signal spots. We then apply conventional methods for spot detection on these enhanced images, thus achieving higher accuracy.

## TODO List

- [x] API
  + [x] Inference
  + [x] Evaluation
  + [x] Plotting tool for show TP, FP, FN
  + [x] Training
- [x] CLI
  + [x] Inference
  + [ ] Evaluation
  + [x] Plotting tool for show TP, FP, FN
  + [x] Training
- [ ] Napari plugin
- [ ] deploy
  + [x] setup.py
  + [x] upload model weights to huggingface
  + [ ] upload package to PyPI
  + [ ] Upload to BioImageIO model zoo
- [ ] Add more datasets
    + [x] ExSeq
    + [ ] MER-FISH
    + [ ] seqFISH
- [x] Try other signal footprint
    + [x] Gaussian
    + [ ] Other shape
- [ ] Benchmark
- [ ] 3D integration method
- [ ] Support for zarr format
- [ ] Documentation

## Usage

```bash
pip install u-fish
```

API for inference and evaluation:

```python
from skimage import io
from ufish.api import UFish

ufish = UFish()
ufish.load_weights("path/to/weights")  # loading model weights
# or download from huggingface:
# ufish.load_weights_from_internet()

# inference
img = io.imread("path/to/image")
pred_spots = ufish.predict(img)

# plot prediction result
fig_spots = ufish.plot_result(img, pred_spots)

# evaluate
true_spots = pd.read_csv("path/to/true_spots.csv")
metrics = ufish.evaluate_result(spots, true_spots)

# plot evaluation result
fig_eval = ufish.plot_evaluation_result(
    img, pred_spots, true_spots, cutoff=3.0)
```

API for training:

```python
from ufish.api import UFish

ufish = UFish()

# loading a pre-trained model
# if you want to train from scratch, just skip this step
ufish.load_weights("path/to/weights")

ufish.train(
    'path/to/train_dir',
    'path/to/val_dir',
    num_epochs=100,
    lr=1e-4,
    model_save_path='path/to/save/model',
)
```

CLI usage:

```bash
# list all sub-commands:
$ python -m ufish

# using --help to see details of each sub-command, e.g.:
$ ufish pred_2d_img --help

# predict one image
$ ufish pred_2d_img input.tiff output.csv

# predict all images in a directory
$ ufish pred_2d_imgs input_dir output_dir

# load a trained model and predict
$ ufish load_weights path/to/weights - pred_2d_img input.tiff output.csv

# training from scratch
$ ufish train path/to/train_dir path/to/val_dir --model_save_path path/to/save/model

# training from a pre-trained model (fine-tuning)
$ ufish load_weights path/to/weights - train path/to/train_dir path/to/val_dir --model_save_path path/to/save/model
```

## Dataset

The dataset is available at [HuggingFace Datasets](https://huggingface.co/datasets/GangCaoLab/FISH_spots):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/datasets/GangCaoLab/FISH_spots
```
