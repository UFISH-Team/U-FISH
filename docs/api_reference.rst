API Reference
=============

This section provides detailed documentation for the U-FISH Python API.

UFish Class
-----------

The main class for U-FISH functionality.

.. autoclass:: ufish.api.UFish
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from ufish.api import UFish
   
   # Initialize
   ufish = UFish()
   
   # Load weights
   ufish.load_weights()
   
   # Predict
   spots, enhanced = ufish.predict(image)

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   UFish(
       model_type='ufish_2d',  # Model type: 'ufish_2d' or 'ufish_3d'
       device='auto',          # Device: 'cpu', 'gpu', or 'auto'
       num_workers=4,          # Number of workers for data loading
       verbose=True            # Print progress messages
   )

Core Methods
------------

load_weights
~~~~~~~~~~~~

Load model weights from file or default location.

.. code-block:: python

   # Load default weights
   ufish.load_weights()
   
   # Load from file
   ufish.load_weights('path/to/model.onnx')
   
   # Load from HuggingFace
   ufish.load_weights_from_internet()

Parameters:

* **weights_path** (str, optional): Path to weights file. If None, loads default weights.
* **download** (bool): Whether to download weights if not found locally.

predict
~~~~~~~

Predict FISH spots in an image.

.. code-block:: python

   spots, enhanced = ufish.predict(
       image,
       threshold=0.5,
       min_distance=3,
       exclude_border=True,
       normalize=True,
       enhance_only=False
   )

Parameters:

* **image** (np.ndarray): Input image (2D or 3D).
* **threshold** (float): Detection threshold (0-1).
* **min_distance** (int): Minimum distance between spots.
* **exclude_border** (bool): Exclude spots on image border.
* **normalize** (bool): Normalize input image.
* **enhance_only** (bool): Return only enhanced image without spot detection.

Returns:

* **spots** (pd.DataFrame): Detected spots with columns [y, x] or [z, y, x].
* **enhanced** (np.ndarray): Enhanced image.

evaluate_result
~~~~~~~~~~~~~~~

Evaluate prediction against ground truth.

.. code-block:: python

   metrics = ufish.evaluate_result(
       pred_spots,
       true_spots,
       cutoff=3.0,
       return_matches=False
   )

Parameters:

* **pred_spots** (pd.DataFrame): Predicted spots.
* **true_spots** (pd.DataFrame): Ground truth spots.
* **cutoff** (float): Maximum distance for matching.
* **return_matches** (bool): Return matched pairs.

Returns:

* **metrics** (dict): Dictionary containing:
  
  - precision: TP / (TP + FP)
  - recall: TP / (TP + FN)
  - f1: F1 score
  - tp: True positives count
  - fp: False positives count
  - fn: False negatives count

plot_result
~~~~~~~~~~~

Visualize detected spots on image.

.. code-block:: python

   fig = ufish.plot_result(
       image,
       spots,
       figsize=(10, 10),
       spot_size=20,
       spot_color='red',
       show_numbers=False,
       z_slice=None
   )

Parameters:

* **image** (np.ndarray): Original image.
* **spots** (pd.DataFrame): Detected spots.
* **figsize** (tuple): Figure size.
* **spot_size** (int): Size of spot markers.
* **spot_color** (str): Color of spot markers.
* **show_numbers** (bool): Show spot numbers.
* **z_slice** (int, optional): For 3D images, which slice to show.

Returns:

* **fig** (matplotlib.figure.Figure): Figure object.

plot_evaluate
~~~~~~~~~~~~~

Visualize evaluation results showing TP, FP, FN.

.. code-block:: python

   fig = ufish.plot_evaluate(
       image,
       pred_spots,
       true_spots,
       cutoff=3.0,
       figsize=(15, 5),
       z_slice=None
   )

Parameters:

* **image** (np.ndarray): Original image.
* **pred_spots** (pd.DataFrame): Predicted spots.
* **true_spots** (pd.DataFrame): Ground truth spots.
* **cutoff** (float): Matching distance threshold.
* **figsize** (tuple): Figure size.
* **z_slice** (int, optional): For 3D images, which slice to show.

Returns:

* **fig** (matplotlib.figure.Figure): Figure with three panels showing predictions, ground truth, and evaluation.

train
~~~~~

Train or fine-tune the model.

.. code-block:: python

   history = ufish.train(
       train_dir,
       val_dir,
       num_epochs=100,
       batch_size=8,
       lr=1e-4,
       lr_scheduler='cosine',
       model_save_path='model.pt',
       checkpoint_interval=10,
       early_stopping_patience=20,
       augmentation=True,
       mixed_precision=True
   )

Parameters:

* **train_dir** (str): Training data directory.
* **val_dir** (str): Validation data directory.
* **num_epochs** (int): Number of training epochs.
* **batch_size** (int): Batch size.
* **lr** (float): Learning rate.
* **lr_scheduler** (str): Learning rate scheduler ('cosine', 'step', or None).
* **model_save_path** (str): Path to save best model.
* **checkpoint_interval** (int): Save checkpoint every N epochs.
* **early_stopping_patience** (int): Early stopping patience.
* **augmentation** (bool): Use data augmentation.
* **mixed_precision** (bool): Use mixed precision training.

Returns:

* **history** (dict): Training history with loss and metrics.

Data Module
-----------

Functions for data loading and preprocessing.

.. automodule:: ufish.data
   :members:
   :undoc-members:

Dataset Classes
~~~~~~~~~~~~~~~

FISHDataset
^^^^^^^^^^^

PyTorch dataset for FISH images and labels.

.. code-block:: python

   from ufish.data import FISHDataset
   
   dataset = FISHDataset(
       image_dir='data/images',
       label_dir='data/labels',
       transform=None,
       target_transform=None
   )

DataLoader Creation
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torch.utils.data import DataLoader
   
   dataloader = DataLoader(
       dataset,
       batch_size=8,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )

Utility Functions
-----------------

Image Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ufish.utils import (
       normalize_image,
       pad_image,
       tile_image,
       stitch_tiles,
       apply_clahe
   )
   
   # Normalize image
   img_norm = normalize_image(img, percentile=(1, 99))
   
   # Pad image for tiling
   img_padded = pad_image(img, tile_size=512, overlap=64)
   
   # Create tiles
   tiles = tile_image(img, tile_size=512, overlap=64)
   
   # Stitch tiles back
   stitched = stitch_tiles(tiles, original_shape, overlap=64)

Spot Processing
~~~~~~~~~~~~~~~

.. code-block:: python

   from ufish.utils import (
       filter_border_spots,
       merge_duplicate_spots,
       spots_to_mask,
       mask_to_spots
   )
   
   # Filter border spots
   filtered = filter_border_spots(spots, image_shape, border=10)
   
   # Merge nearby spots
   merged = merge_duplicate_spots(spots, min_distance=3)
   
   # Convert spots to mask
   mask = spots_to_mask(spots, image_shape)
   
   # Convert mask to spots
   spots = mask_to_spots(mask)

File I/O
~~~~~~~~

.. code-block:: python

   from ufish.utils import (
       read_zarr_chunk,
       write_zarr_chunk,
       read_n5_dataset,
       save_spots_csv
   )
   
   # Read zarr chunk
   chunk = read_zarr_chunk('data.zarr', chunk_coords)
   
   # Save spots to CSV
   save_spots_csv(spots, 'output.csv', include_confidence=True)

Model Module
------------

Low-level model operations.

.. automodule:: ufish.model
   :members:
   :undoc-members:

Model Architecture
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ufish.model import UNet2D, UNet3D
   
   # Create 2D U-Net
   model_2d = UNet2D(
       in_channels=1,
       out_channels=1,
       features=[16, 32, 64, 128]
   )
   
   # Create 3D U-Net
   model_3d = UNet3D(
       in_channels=1,
       out_channels=1,
       features=[16, 32, 64]
   )

ONNX Export
~~~~~~~~~~~

.. code-block:: python

   from ufish.model import export_to_onnx
   
   export_to_onnx(
       model,
       input_shape=(1, 1, 512, 512),
       output_path='model.onnx',
       opset_version=11
   )

Advanced Features
-----------------

Custom Peak Detection
~~~~~~~~~~~~~~~~~~~~~

Implement custom peak detection:

.. code-block:: python

   from skimage.feature import peak_local_max
   
   class CustomUFish(UFish):
       def detect_peaks(self, enhanced_image, **kwargs):
           # Custom peak detection logic
           peaks = peak_local_max(
               enhanced_image,
               min_distance=kwargs.get('min_distance', 3),
               threshold_abs=kwargs.get('threshold', 0.5),
               exclude_border=kwargs.get('exclude_border', True),
               p_norm=2  # Custom parameter
           )
           return peaks

Ensemble Prediction
~~~~~~~~~~~~~~~~~~~

Combine multiple models:

.. code-block:: python

   class EnsembleUFish:
       def __init__(self, model_paths):
           self.models = [UFish() for _ in model_paths]
           for model, path in zip(self.models, model_paths):
               model.load_weights(path)
       
       def predict(self, image):
           all_enhanced = []
           for model in self.models:
               _, enhanced = model.predict(image, enhance_only=True)
               all_enhanced.append(enhanced)
           
           # Average enhanced images
           avg_enhanced = np.mean(all_enhanced, axis=0)
           
           # Detect spots on averaged result
           spots = self.detect_spots(avg_enhanced)
           return spots, avg_enhanced

Batch GPU Processing
~~~~~~~~~~~~~~~~~~~~

Process multiple images on GPU efficiently:

.. code-block:: python

   import torch
   
   def batch_predict_gpu(ufish, images, batch_size=16):
       results = []
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       for i in range(0, len(images), batch_size):
           batch = images[i:i+batch_size]
           batch_tensor = torch.stack([torch.from_numpy(img) for img in batch])
           batch_tensor = batch_tensor.to(device)
           
           with torch.no_grad():
               enhanced_batch = ufish.model(batch_tensor)
           
           for enhanced in enhanced_batch:
               spots = ufish.detect_spots(enhanced.cpu().numpy())
               results.append(spots)
       
       return results 