User Guide
==========

This comprehensive guide covers all aspects of using U-FISH for FISH spot detection.

Overview
--------

U-FISH (Unified FISH) is a deep learning-based tool for detecting fluorescent spots in FISH (Fluorescence in situ Hybridization) images. It uses a U-Net architecture to enhance images and detect spots with high accuracy across diverse experimental conditions.

Understanding U-FISH
--------------------

How It Works
~~~~~~~~~~~~

U-FISH employs a two-stage approach:

1. **Image Enhancement**: A U-Net model processes the input image to create a unified enhanced image (UEI) that normalizes different backgrounds and signal characteristics.
2. **Spot Detection**: Local maxima detection on the enhanced image identifies spot locations.

Model Architecture
~~~~~~~~~~~~~~~~~~

The U-FISH model is based on U-Net with several optimizations:

* Lightweight design with only 160k parameters
* ONNX format for fast inference
* Support for 2D and 3D images
* Multi-scale feature extraction

Input Images
------------

Supported Formats
~~~~~~~~~~~~~~~~~

U-FISH accepts images in various formats:

* **File formats**: TIFF, PNG, JPG, and any format supported by scikit-image
* **Data types**: uint8, uint16, float32, float64
* **Dimensions**: 2D (Y, X) or 3D (Z, Y, X)
* **Channels**: Single-channel or multi-channel (process each channel separately)

Image Preprocessing
~~~~~~~~~~~~~~~~~~~

U-FISH automatically handles image preprocessing:

.. code-block:: python

   # No manual preprocessing needed
   img = io.imread("image.tiff")
   spots, enhanced = ufish.predict(img)
   
   # For noisy images, you might want to apply denoising first
   from skimage.filters import gaussian
   img_denoised = gaussian(img, sigma=1)
   spots, enhanced = ufish.predict(img_denoised)

Working with Different Data Types
---------------------------------

2D Images
~~~~~~~~~

Standard 2D image processing:

.. code-block:: python

   # Load 2D image
   img_2d = io.imread("2d_image.tiff")
   
   # Predict spots
   spots_2d, enhanced_2d = ufish.predict(img_2d)
   
   # Results contain x, y coordinates
   print(spots_2d[['y', 'x']].head())

3D Image Stacks
~~~~~~~~~~~~~~~

For 3D images, U-FISH offers multiple processing strategies:

.. code-block:: python

   # Method 1: Direct 3D processing (default)
   img_3d = io.imread("3d_stack.tiff")
   spots_3d, enhanced_3d = ufish.predict(img_3d)
   
   # Method 2: 3D blending for better accuracy
   spots_3d, enhanced_3d = ufish.predict(img_3d, use_3d_blend=True)
   
   # Method 3: Process slice by slice
   all_spots = []
   for z, slice_img in enumerate(img_3d):
       spots, _ = ufish.predict(slice_img)
       spots['z'] = z
       all_spots.append(spots)
   spots_combined = pd.concat(all_spots)

Large Images and Tiling
~~~~~~~~~~~~~~~~~~~~~~~

For images that don't fit in memory:

.. code-block:: python

   # Process large images in tiles
   from ufish.utils import process_large_image
   
   spots = process_large_image(
       image_path="huge_image.tiff",
       ufish_model=ufish,
       tile_size=1024,
       overlap=128
   )

Zarr and OME-Zarr Support
~~~~~~~~~~~~~~~~~~~~~~~~~

U-FISH supports efficient processing of large-scale data:

.. code-block:: python

   import zarr
   
   # Open zarr array
   z_arr = zarr.open("data.zarr", mode='r')
   
   # Process chunks
   for chunk_coords in z_arr.chunk_coords:
       chunk_data = z_arr[chunk_coords]
       spots, _ = ufish.predict(chunk_data)
       # Process spots...

Advanced Prediction Options
---------------------------

Adjusting Sensitivity
~~~~~~~~~~~~~~~~~~~~~

Control detection sensitivity:

.. code-block:: python

   # More sensitive detection (lower threshold)
   spots_sensitive = ufish.predict(img, threshold=0.3)
   
   # Less sensitive detection (higher threshold)
   spots_strict = ufish.predict(img, threshold=0.7)
   
   # Custom peak detection parameters
   spots_custom = ufish.predict(
       img,
       min_distance=5,  # Minimum distance between spots
       threshold_abs=0.5  # Absolute threshold
   )

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple images efficiently:

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   
   image_dir = Path("images/")
   results = {}
   
   for img_path in image_dir.glob("*.tiff"):
       img = io.imread(img_path)
       spots, _ = ufish.predict(img)
       results[img_path.name] = spots
   
   # Save all results
   for name, spots in results.items():
       spots.to_csv(f"results/{name}.csv", index=False)

Performance Optimization
------------------------

GPU Acceleration
~~~~~~~~~~~~~~~~

Enable GPU for faster processing:

.. code-block:: python

   # Check if GPU is available
   import onnxruntime as ort
   
   providers = ort.get_available_providers()
   print(f"Available providers: {providers}")
   
   # Use GPU if available
   ufish = UFish(device='gpu' if 'CUDAExecutionProvider' in providers else 'cpu')

Memory Management
~~~~~~~~~~~~~~~~~

For memory-constrained systems:

.. code-block:: python

   # Process in smaller batches
   def process_with_limited_memory(img_3d, batch_size=10):
       results = []
       for i in range(0, img_3d.shape[0], batch_size):
           batch = img_3d[i:i+batch_size]
           spots, _ = ufish.predict(batch)
           spots['z'] += i  # Adjust z coordinates
           results.append(spots)
       return pd.concat(results)

Evaluation and Metrics
----------------------

Matching Predictions to Ground Truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U-FISH uses Hungarian algorithm for optimal matching:

.. code-block:: python

   # Evaluate with different cutoff distances
   for cutoff in [2.0, 3.0, 4.0]:
       metrics = ufish.evaluate_result(pred_spots, true_spots, cutoff=cutoff)
       print(f"Cutoff {cutoff}: F1={metrics['f1']:.3f}")

Custom Evaluation
~~~~~~~~~~~~~~~~~

Implement custom evaluation metrics:

.. code-block:: python

   from scipy.spatial import distance_matrix
   
   def custom_evaluate(pred, true, cutoff=3.0):
       # Compute distance matrix
       dist = distance_matrix(pred[['y', 'x']], true[['y', 'x']])
       
       # Find matches
       matches = dist < cutoff
       tp = matches.any(axis=1).sum()
       fp = len(pred) - tp
       fn = len(true) - matches.any(axis=0).sum()
       
       return {'TP': tp, 'FP': fp, 'FN': fn}

Training and Fine-tuning
------------------------

Preparing Training Data
~~~~~~~~~~~~~~~~~~~~~~~

Organize your data:

.. code-block:: text

   training_data/
   ├── images/
   │   ├── img_001.tiff
   │   ├── img_002.tiff
   │   └── ...
   └── labels/
       ├── img_001.csv  # Columns: y, x, (z)
       ├── img_002.csv
       └── ...

Data Augmentation
~~~~~~~~~~~~~~~~~

U-FISH applies augmentation during training:

* Random flips and rotations
* Intensity adjustments
* Noise addition
* Elastic deformations

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fine-tune with custom parameters
   ufish.train(
       train_dir='data/train',
       val_dir='data/val',
       num_epochs=100,
       batch_size=8,
       lr=1e-4,
       lr_scheduler='cosine',
       early_stopping_patience=10,
       model_save_path='models/finetuned.pt'
   )

Monitoring Training
~~~~~~~~~~~~~~~~~~~

Training progress is logged:

.. code-block:: python

   # Check training history
   history = ufish.training_history
   
   import matplotlib.pyplot as plt
   
   plt.plot(history['train_loss'], label='Train')
   plt.plot(history['val_loss'], label='Validation')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()

Best Practices
--------------

1. **Image Quality**
   
   * Ensure good signal-to-noise ratio
   * Avoid saturated pixels
   * Use consistent imaging parameters

2. **Model Selection**
   
   * Start with pre-trained weights
   * Fine-tune on representative data
   * Validate on held-out test set

3. **Parameter Tuning**
   
   * Adjust threshold based on your data
   * Consider spot density when setting min_distance
   * Use visualization to verify results

4. **Performance**
   
   * Use GPU for large datasets
   * Process in batches for memory efficiency
   * Enable tiling for very large images

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Low detection rate**

* Decrease threshold value
* Check image normalization
* Consider fine-tuning on your data

**Too many false positives**

* Increase threshold value
* Adjust min_distance parameter
* Apply image denoising

**Memory errors**

* Process images in smaller chunks
* Reduce batch size
* Use tiling for large images

**Slow processing**

* Enable GPU acceleration
* Process multiple images in parallel
* Use ONNX runtime optimizations 