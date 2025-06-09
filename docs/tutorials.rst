Tutorials
=========

This section provides step-by-step tutorials for common U-FISH use cases.

Tutorial 1: Basic FISH Spot Detection
-------------------------------------

This tutorial covers the basics of detecting FISH spots in 2D images.

Step 1: Installation
~~~~~~~~~~~~~~~~~~~~

First, install U-FISH:

.. code-block:: bash

   pip install ufish

Step 2: Load and Prepare Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from skimage import io
   import matplotlib.pyplot as plt
   from ufish.api import UFish
   
   # Load your FISH image
   image = io.imread("sample_fish_image.tiff")
   
   # Display the image
   plt.figure(figsize=(8, 8))
   plt.imshow(image, cmap='gray')
   plt.title("Original FISH Image")
   plt.axis('off')
   plt.show()

Step 3: Initialize U-FISH
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create U-FISH instance
   ufish = UFish()
   
   # Load pre-trained weights
   ufish.load_weights()
   print("Model loaded successfully!")

Step 4: Detect Spots
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect spots
   spots, enhanced_image = ufish.predict(image)
   
   print(f"Detected {len(spots)} spots")
   print("\nFirst 5 spots:")
   print(spots.head())

Step 5: Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot results
   fig = ufish.plot_result(image, spots, figsize=(12, 6))
   plt.tight_layout()
   plt.show()
   
   # Save results
   spots.to_csv("detected_spots.csv", index=False)
   print("Results saved to detected_spots.csv")

Tutorial 2: Fine-tuning on Custom Data
--------------------------------------

This tutorial shows how to fine-tune U-FISH on your own dataset.

Preparing Your Data
~~~~~~~~~~~~~~~~~~~

Organize your data in the following structure:

.. code-block:: text

   custom_data/
   ├── train/
   │   ├── images/
   │   │   ├── img_001.tiff
   │   │   ├── img_002.tiff
   │   │   └── ...
   │   └── labels/
   │       ├── img_001.csv
   │       ├── img_002.csv
   │       └── ...
   └── val/
       ├── images/
       └── labels/

Each CSV file should have columns: y, x (and z for 3D images).

Loading Pre-trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ufish.api import UFish
   
   # Initialize with pre-trained weights
   ufish = UFish()
   ufish.load_weights()

Fine-tuning
~~~~~~~~~~~

.. code-block:: python

   # Fine-tune on your data
   history = ufish.train(
       train_dir='custom_data/train',
       val_dir='custom_data/val',
       num_epochs=50,
       batch_size=4,
       lr=5e-5,  # Lower learning rate for fine-tuning
       model_save_path='finetuned_model.pt',
       early_stopping_patience=10
   )

Monitoring Training
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot training history
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 2, 1)
   plt.plot(history['epoch'], history['train_loss'], label='Train')
   plt.plot(history['epoch'], history['val_loss'], label='Validation')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.title('Training Loss')
   
   plt.subplot(1, 2, 2)
   plt.plot(history['epoch'], history['val_f1'])
   plt.xlabel('Epoch')
   plt.ylabel('F1 Score')
   plt.title('Validation F1 Score')
   
   plt.tight_layout()
   plt.show()

Testing Fine-tuned Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load fine-tuned model
   ufish_finetuned = UFish()
   ufish_finetuned.load_weights('finetuned_model.pt')
   
   # Test on new image
   test_image = io.imread("test_image.tiff")
   spots, _ = ufish_finetuned.predict(test_image)
   
   print(f"Detected {len(spots)} spots with fine-tuned model")

Tutorial 3: Processing 3D Image Stacks
--------------------------------------

This tutorial demonstrates how to work with 3D FISH images.

Loading 3D Data
~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from skimage import io
   from ufish.api import UFish
   
   # Load 3D image stack (Z, Y, X)
   image_3d = io.imread("fish_stack.tiff")
   print(f"Image shape: {image_3d.shape}")
   
   # Initialize U-FISH
   ufish = UFish()
   ufish.load_weights()

Method 1: Direct 3D Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process entire 3D stack at once
   spots_3d, enhanced_3d = ufish.predict(image_3d)
   
   print(f"Detected {len(spots_3d)} spots in 3D")
   print("\nColumns:", spots_3d.columns.tolist())
   print("\nFirst 5 spots:")
   print(spots_3d.head())

Method 2: 3D Blending
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use 3D blending for better accuracy
   spots_blend, enhanced_blend = ufish.predict(
       image_3d, 
       use_3d_blend=True
   )
   
   print(f"Detected {len(spots_blend)} spots with 3D blending")

Visualizing 3D Results
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   # 3D scatter plot of detected spots
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   ax.scatter(
       spots_3d['x'], 
       spots_3d['y'], 
       spots_3d['z'],
       c='red', 
       s=20, 
       alpha=0.6
   )
   
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title(f'3D FISH Spots ({len(spots_3d)} detected)')
   
   plt.show()

Analyzing Z-distribution
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze spot distribution across Z
   plt.figure(figsize=(10, 4))
   
   plt.subplot(1, 2, 1)
   plt.hist(spots_3d['z'], bins=image_3d.shape[0], alpha=0.7)
   plt.xlabel('Z slice')
   plt.ylabel('Number of spots')
   plt.title('Spot Distribution Across Z')
   
   plt.subplot(1, 2, 2)
   for z in range(0, image_3d.shape[0], 5):
       z_spots = spots_3d[spots_3d['z'] == z]
       plt.scatter(z_spots['x'], z_spots['y'], s=5, alpha=0.5, label=f'Z={z}')
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title('XY Distribution (every 5th Z slice)')
   plt.axis('equal')
   
   plt.tight_layout()
   plt.show()

Tutorial 4: Batch Processing Large Datasets
-------------------------------------------

This tutorial covers efficient processing of large datasets.

Setting Up Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from concurrent.futures import ProcessPoolExecutor
   from ufish.api import UFish
   import numpy as np
   
   def process_single_image(img_path, model_path):
       """Process a single image"""
       # Each process loads its own model
       ufish = UFish()
       ufish.load_weights(model_path)
       
       # Load and process image
       image = io.imread(img_path)
       spots, _ = ufish.predict(image)
       
       return {
           'filename': img_path.name,
           'n_spots': len(spots),
           'spots': spots
       }

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Directory with images
   image_dir = Path("large_dataset/images")
   image_files = list(image_dir.glob("*.tiff"))
   
   # Save model for parallel processing
   model_path = "model_for_batch.onnx"
   
   # Process in parallel
   results = []
   with ProcessPoolExecutor(max_workers=4) as executor:
       futures = [
           executor.submit(process_single_image, img_path, model_path)
           for img_path in image_files
       ]
       
       for i, future in enumerate(futures):
           result = future.result()
           results.append(result)
           print(f"Processed {i+1}/{len(image_files)}: {result['filename']} "
                 f"({result['n_spots']} spots)")

Saving Batch Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create output directory
   output_dir = Path("large_dataset/results")
   output_dir.mkdir(exist_ok=True)
   
   # Save individual results
   for result in results:
       output_path = output_dir / f"{result['filename']}.csv"
       result['spots'].to_csv(output_path, index=False)
   
   # Create summary
   summary = pd.DataFrame([
       {
           'filename': r['filename'],
           'n_spots': r['n_spots'],
           'mean_x': r['spots']['x'].mean(),
           'mean_y': r['spots']['y'].mean(),
           'std_x': r['spots']['x'].std(),
           'std_y': r['spots']['y'].std()
       }
       for r in results
   ])
   
   summary.to_csv(output_dir / "summary.csv", index=False)
   print(f"\nSummary saved to {output_dir}/summary.csv")

Tutorial 5: Working with Large Images (Tiling)
-----------------------------------------------

This tutorial shows how to process images that are too large to fit in memory.

Implementing Tiling
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def process_large_image_tiled(image_path, ufish, tile_size=1024, overlap=128):
       """Process large image using tiling"""
       from skimage import io
       import numpy as np
       
       # Read image metadata without loading
       image = io.imread(image_path)
       height, width = image.shape[:2]
       
       all_spots = []
       
       # Calculate tile positions
       for y in range(0, height - overlap, tile_size - overlap):
           for x in range(0, width - overlap, tile_size - overlap):
               # Define tile boundaries
               y_end = min(y + tile_size, height)
               x_end = min(x + tile_size, width)
               
               # Extract tile
               tile = image[y:y_end, x:x_end]
               
               # Process tile
               tile_spots, _ = ufish.predict(tile)
               
               # Adjust coordinates to global space
               tile_spots['y'] += y
               tile_spots['x'] += x
               
               # Filter spots near tile edges (except image edges)
               if x > 0:
                   tile_spots = tile_spots[tile_spots['x'] > x + overlap//2]
               if y > 0:
                   tile_spots = tile_spots[tile_spots['y'] > y + overlap//2]
               if x_end < width:
                   tile_spots = tile_spots[tile_spots['x'] < x_end - overlap//2]
               if y_end < height:
                   tile_spots = tile_spots[tile_spots['y'] < y_end - overlap//2]
               
               all_spots.append(tile_spots)
       
       # Combine all spots
       final_spots = pd.concat(all_spots, ignore_index=True)
       return final_spots

Using the Tiling Function
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize U-FISH
   ufish = UFish()
   ufish.load_weights()
   
   # Process large image
   large_image_path = "very_large_image.tiff"
   spots = process_large_image_tiled(
       large_image_path, 
       ufish,
       tile_size=2048,
       overlap=256
   )
   
   print(f"Detected {len(spots)} spots in large image")

Tutorial 6: Custom Evaluation and Metrics
-----------------------------------------

This tutorial demonstrates how to perform custom evaluation of results.

Creating Ground Truth
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If you have manual annotations in ImageJ format
   def imagej_to_dataframe(roi_file):
       """Convert ImageJ ROI to dataframe"""
       from read_roi import read_roi
       
       rois = read_roi(roi_file)
       spots = []
       
       for roi_name, roi in rois.items():
           if roi['type'] == 'point':
               for x, y in zip(roi['x'], roi['y']):
                   spots.append({'x': x, 'y': y})
       
       return pd.DataFrame(spots)

Custom Matching Function
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy.spatial.distance import cdist
   from scipy.optimize import linear_sum_assignment
   
   def custom_match_spots(pred_spots, true_spots, max_distance=3.0):
       """Custom spot matching with detailed statistics"""
       
       # Calculate distance matrix
       pred_coords = pred_spots[['y', 'x']].values
       true_coords = true_spots[['y', 'x']].values
       dist_matrix = cdist(pred_coords, true_coords)
       
       # Hungarian matching
       row_ind, col_ind = linear_sum_assignment(dist_matrix)
       
       # Filter matches by distance
       matches = []
       for i, j in zip(row_ind, col_ind):
           if dist_matrix[i, j] <= max_distance:
               matches.append({
                   'pred_idx': i,
                   'true_idx': j,
                   'distance': dist_matrix[i, j],
                   'pred_y': pred_coords[i, 0],
                   'pred_x': pred_coords[i, 1],
                   'true_y': true_coords[j, 0],
                   'true_x': true_coords[j, 1]
               })
       
       matches_df = pd.DataFrame(matches)
       
       # Calculate metrics
       tp = len(matches_df)
       fp = len(pred_spots) - tp
       fn = len(true_spots) - tp
       
       precision = tp / (tp + fp) if (tp + fp) > 0 else 0
       recall = tp / (tp + fn) if (tp + fn) > 0 else 0
       f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
       
       return {
           'matches': matches_df,
           'tp': tp,
           'fp': fp,
           'fn': fn,
           'precision': precision,
           'recall': recall,
           'f1': f1,
           'mean_distance': matches_df['distance'].mean() if len(matches_df) > 0 else np.nan
       }

Detailed Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load predictions and ground truth
   pred_spots = pd.read_csv("predicted_spots.csv")
   true_spots = pd.read_csv("ground_truth_spots.csv")
   
   # Perform matching
   results = custom_match_spots(pred_spots, true_spots, max_distance=3.0)
   
   print(f"Precision: {results['precision']:.3f}")
   print(f"Recall: {results['recall']:.3f}")
   print(f"F1 Score: {results['f1']:.3f}")
   print(f"Mean match distance: {results['mean_distance']:.2f} pixels")
   
   # Analyze match distances
   if len(results['matches']) > 0:
       plt.figure(figsize=(10, 4))
       
       plt.subplot(1, 2, 1)
       plt.hist(results['matches']['distance'], bins=30, alpha=0.7)
       plt.xlabel('Match Distance (pixels)')
       plt.ylabel('Count')
       plt.title('Distribution of Match Distances')
       
       plt.subplot(1, 2, 2)
       plt.scatter(
           results['matches']['true_x'], 
           results['matches']['true_y'],
           c=results['matches']['distance'],
           cmap='viridis',
           s=20
       )
       plt.colorbar(label='Match Distance')
       plt.xlabel('X')
       plt.ylabel('Y')
       plt.title('Spatial Distribution of Match Quality')
       
       plt.tight_layout()
       plt.show()

Best Practices Summary
----------------------

1. **Image Quality**
   - Ensure consistent imaging conditions
   - Check for saturation and noise levels
   - Consider preprocessing if needed

2. **Model Selection**
   - Start with pre-trained weights
   - Fine-tune on representative samples
   - Validate on independent test set

3. **Parameter Optimization**
   - Adjust threshold based on SNR
   - Set min_distance based on expected spot density
   - Use visualization to verify parameters

4. **Performance Tips**
   - Use GPU for large datasets
   - Implement tiling for very large images
   - Process in parallel when possible

5. **Validation**
   - Always evaluate on ground truth when available
   - Check both precision and recall
   - Visualize errors to understand failure modes 