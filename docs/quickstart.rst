Quick Start Guide
=================

This guide will help you get started with U-FISH quickly. We'll cover the basic workflow for detecting FISH spots in your images.

Basic Workflow
--------------

1. **Import U-FISH and load an image**
2. **Initialize the model**
3. **Run prediction**
4. **Visualize results**

Simple Example
--------------

Here's a minimal example to detect FISH spots:

.. code-block:: python

   from skimage import io
   from ufish.api import UFish
   
   # Initialize U-FISH
   ufish = UFish()
   
   # Load pre-trained weights
   ufish.load_weights()
   
   # Load your image
   img = io.imread("path/to/your/image.tiff")
   
   # Predict spots
   pred_spots, enhanced_img = ufish.predict(img)
   
   # pred_spots is a pandas DataFrame with columns: y, x, (z if 3D)
   print(f"Detected {len(pred_spots)} spots")
   print(pred_spots.head())

Visualizing Results
-------------------

U-FISH provides built-in visualization functions:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot the detected spots on the original image
   fig = ufish.plot_result(img, pred_spots)
   plt.show()

Evaluating Performance
----------------------

If you have ground truth annotations:

.. code-block:: python

   import pandas as pd
   
   # Load ground truth spots
   true_spots = pd.read_csv("path/to/ground_truth.csv")
   
   # Evaluate predictions
   metrics = ufish.evaluate_result(pred_spots, true_spots, cutoff=3.0)
   print(f"Precision: {metrics['precision']:.3f}")
   print(f"Recall: {metrics['recall']:.3f}")
   print(f"F1 Score: {metrics['f1']:.3f}")
   
   # Visualize TP, FP, FN
   fig_eval = ufish.plot_evaluate(img, pred_spots, true_spots, cutoff=3.0)
   plt.show()

Command Line Usage
------------------

U-FISH also provides a command-line interface:

.. code-block:: bash

   # Predict spots in a single image
   ufish predict input.tiff output.csv
   
   # Process multiple images
   ufish predict-imgs input_directory/ output_directory/
   
   # Use custom model weights
   ufish load-weights custom_model.onnx - predict input.tiff output.csv

Working with 3D Images
----------------------

U-FISH supports 3D image stacks:

.. code-block:: python

   # Load a 3D image (Z, Y, X)
   img_3d = io.imread("path/to/stack.tiff")
   
   # Predict in 3D
   spots_3d, enhanced_3d = ufish.predict(img_3d)
   
   # spots_3d will have columns: z, y, x
   print(spots_3d.head())

Multi-channel Images
--------------------

For multi-channel images:

.. code-block:: python

   # If your image has shape (C, Y, X) or (C, Z, Y, X)
   # Process each channel separately
   all_spots = []
   
   for c in range(img.shape[0]):
       channel_img = img[c]
       spots, _ = ufish.predict(channel_img)
       spots['channel'] = c
       all_spots.append(spots)
   
   # Combine results
   combined_spots = pd.concat(all_spots, ignore_index=True)

Fine-tuning on Your Data
------------------------

To adapt U-FISH to your specific data:

.. code-block:: python

   # Load pre-trained model
   ufish.load_weights()
   
   # Fine-tune on your data
   ufish.train(
       train_dir='path/to/training_data/',
       val_dir='path/to/validation_data/',
       num_epochs=50,
       lr=1e-4,
       model_save_path='my_finetuned_model.pt'
   )

Data Format for Training
~~~~~~~~~~~~~~~~~~~~~~~~

Training data should be organized as:

.. code-block:: text

   train_dir/
   ├── images/
   │   ├── image001.tiff
   │   ├── image002.tiff
   │   └── ...
   └── labels/
       ├── image001.csv  # Columns: y, x, (z if 3D)
       ├── image002.csv
       └── ...

Next Steps
----------

* Read the :doc:`user_guide` for detailed information
* Check out :doc:`tutorials` for specific use cases
* Explore the :doc:`api_reference` for advanced features
* Learn about :doc:`cli_reference` for batch processing 