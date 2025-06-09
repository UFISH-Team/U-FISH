CLI Reference
=============

U-FISH provides a comprehensive command-line interface for all major operations.

Overview
--------

The U-FISH CLI can be accessed via:

.. code-block:: bash

   ufish <command> [options]

Or alternatively:

.. code-block:: bash

   python -m ufish <command> [options]

To see all available commands:

.. code-block:: bash

   ufish --help

Commands
--------

predict
~~~~~~~

Predict FISH spots in a single image.

.. code-block:: bash

   ufish predict <input_image> <output_csv> [options]

**Arguments:**

* ``input_image``: Path to input image file
* ``output_csv``: Path to save detected spots

**Options:**

* ``--threshold``: Detection threshold (default: 0.5)
* ``--min-distance``: Minimum distance between spots (default: 3)
* ``--model``: Path to custom model file
* ``--save-enhanced``: Save enhanced image to file
* ``--device``: Device to use ('cpu' or 'gpu', default: auto)
* ``--verbose``: Enable verbose output

**Examples:**

.. code-block:: bash

   # Basic usage
   ufish predict image.tiff spots.csv
   
   # With custom threshold
   ufish predict image.tiff spots.csv --threshold 0.3
   
   # Save enhanced image
   ufish predict image.tiff spots.csv --save-enhanced enhanced.tiff
   
   # Use GPU
   ufish predict image.tiff spots.csv --device gpu

predict-imgs
~~~~~~~~~~~~

Process multiple images in a directory.

.. code-block:: bash

   ufish predict-imgs <input_dir> <output_dir> [options]

**Arguments:**

* ``input_dir``: Directory containing input images
* ``output_dir``: Directory to save results

**Options:**

* ``--pattern``: File pattern to match (default: "*.tif*")
* ``--threshold``: Detection threshold (default: 0.5)
* ``--min-distance``: Minimum distance between spots (default: 3)
* ``--model``: Path to custom model file
* ``--save-enhanced``: Save enhanced images
* ``--parallel``: Number of parallel workers (default: 1)
* ``--recursive``: Process subdirectories recursively

**Examples:**

.. code-block:: bash

   # Process all TIFF files
   ufish predict-imgs ./images ./results
   
   # Process specific pattern
   ufish predict-imgs ./images ./results --pattern "*_DAPI_*.tiff"
   
   # Parallel processing
   ufish predict-imgs ./images ./results --parallel 4
   
   # Recursive with enhanced images
   ufish predict-imgs ./images ./results --recursive --save-enhanced

train
~~~~~

Train or fine-tune a model.

.. code-block:: bash

   ufish train <train_dir> <val_dir> [options]

**Arguments:**

* ``train_dir``: Training data directory
* ``val_dir``: Validation data directory

**Options:**

* ``--epochs``: Number of training epochs (default: 100)
* ``--batch-size``: Batch size (default: 8)
* ``--lr``: Learning rate (default: 1e-4)
* ``--model-save-dir``: Directory to save model
* ``--checkpoint-interval``: Save checkpoint every N epochs
* ``--resume``: Resume from checkpoint
* ``--pretrained``: Path to pretrained model
* ``--augmentation``: Enable data augmentation
* ``--device``: Device to use ('cpu' or 'gpu')

**Examples:**

.. code-block:: bash

   # Train from scratch
   ufish train ./train ./val --model-save-dir ./models
   
   # Fine-tune pretrained model
   ufish train ./train ./val --pretrained ufish_default.onnx
   
   # Resume training
   ufish train ./train ./val --resume checkpoint_epoch_50.pt
   
   # Custom parameters
   ufish train ./train ./val --epochs 200 --lr 5e-5 --batch-size 16

evaluate
~~~~~~~~

Evaluate model performance on test data.

.. code-block:: bash

   ufish evaluate <test_dir> [options]

**Arguments:**

* ``test_dir``: Test data directory

**Options:**

* ``--model``: Path to model file
* ``--cutoff``: Matching distance cutoff (default: 3.0)
* ``--output``: Save evaluation results to file
* ``--plot``: Generate evaluation plots
* ``--per-image``: Show per-image results

**Examples:**

.. code-block:: bash

   # Basic evaluation
   ufish evaluate ./test
   
   # With custom model and cutoff
   ufish evaluate ./test --model custom.onnx --cutoff 2.5
   
   # Save detailed results
   ufish evaluate ./test --output eval_results.json --per-image

convert
~~~~~~~

Convert between different model formats.

.. code-block:: bash

   ufish convert <input_model> <output_model> [options]

**Arguments:**

* ``input_model``: Input model file
* ``output_model``: Output model file

**Options:**

* ``--input-format``: Input format ('pytorch', 'onnx')
* ``--output-format``: Output format ('pytorch', 'onnx')
* ``--opset-version``: ONNX opset version (default: 11)
* ``--simplify``: Simplify ONNX model

**Examples:**

.. code-block:: bash

   # PyTorch to ONNX
   ufish convert model.pt model.onnx
   
   # ONNX simplification
   ufish convert model.onnx model_simple.onnx --simplify

benchmark
~~~~~~~~~

Run performance benchmarks.

.. code-block:: bash

   ufish benchmark <image_or_dir> [options]

**Arguments:**

* ``image_or_dir``: Image file or directory

**Options:**

* ``--model``: Path to model file
* ``--iterations``: Number of iterations (default: 100)
* ``--warmup``: Warmup iterations (default: 10)
* ``--device``: Device to use
* ``--batch-size``: Batch size for throughput test

**Examples:**

.. code-block:: bash

   # Single image benchmark
   ufish benchmark image.tiff --iterations 100
   
   # Directory benchmark
   ufish benchmark ./images --device gpu --batch-size 8

load-weights
~~~~~~~~~~~~

Load model weights for chaining with other commands.

.. code-block:: bash

   ufish load-weights <weights_path> - <next_command>

**Examples:**

.. code-block:: bash

   # Load weights then predict
   ufish load-weights custom.onnx - predict image.tiff spots.csv
   
   # Load weights then batch process
   ufish load-weights custom.onnx - predict-imgs ./input ./output

visualize
~~~~~~~~~

Visualize detection results.

.. code-block:: bash

   ufish visualize <image> <spots_csv> [options]

**Arguments:**

* ``image``: Input image file
* ``spots_csv``: CSV file with detected spots

**Options:**

* ``--output``: Output image file
* ``--true-spots``: Ground truth spots CSV for comparison
* ``--cutoff``: Matching cutoff for evaluation
* ``--spot-size``: Size of spot markers
* ``--dpi``: Output resolution

**Examples:**

.. code-block:: bash

   # Basic visualization
   ufish visualize image.tiff spots.csv --output result.png
   
   # With ground truth comparison
   ufish visualize image.tiff spots.csv --true-spots truth.csv

Configuration Files
-------------------

U-FISH supports configuration files for complex workflows:

.. code-block:: yaml

   # config.yaml
   model:
     path: "custom_model.onnx"
     device: "gpu"
   
   prediction:
     threshold: 0.4
     min_distance: 3
     save_enhanced: true
   
   training:
     epochs: 150
     batch_size: 16
     learning_rate: 1e-4
     augmentation: true

Use configuration:

.. code-block:: bash

   ufish --config config.yaml predict-imgs ./input ./output

Environment Variables
---------------------

U-FISH recognizes several environment variables:

* ``UFISH_MODEL_PATH``: Default model path
* ``UFISH_CACHE_DIR``: Cache directory for downloaded models
* ``UFISH_DEVICE``: Default device ('cpu' or 'gpu')
* ``UFISH_LOG_LEVEL``: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

Example:

.. code-block:: bash

   export UFISH_DEVICE=gpu
   export UFISH_MODEL_PATH=/models/custom.onnx
   ufish predict image.tiff spots.csv

Batch Scripts
-------------

Example batch processing script:

.. code-block:: bash

   #!/bin/bash
   # process_batch.sh
   
   # Set parameters
   MODEL="finetuned_model.onnx"
   THRESHOLD=0.4
   
   # Process each subdirectory
   for dir in data/*/; do
       name=$(basename "$dir")
       echo "Processing $name..."
       
       ufish load-weights $MODEL - predict-imgs \
           "$dir/images" \
           "results/$name" \
           --threshold $THRESHOLD \
           --save-enhanced \
           --parallel 4
   done

PowerShell example for Windows:

.. code-block:: powershell

   # process_batch.ps1
   
   $model = "finetuned_model.onnx"
   $threshold = 0.4
   
   Get-ChildItem -Path "data" -Directory | ForEach-Object {
       $name = $_.Name
       Write-Host "Processing $name..."
       
       ufish load-weights $model - predict-imgs `
           "$($_.FullName)\images" `
           "results\$name" `
           --threshold $threshold `
           --save-enhanced `
           --parallel 4
   }

Tips and Tricks
---------------

1. **Chaining Commands**: Use ``-`` to chain commands:

   .. code-block:: bash

      ufish load-weights model.onnx - evaluate test/ - visualize test/img.tiff results.csv

2. **Parallel Processing**: Use ``--parallel`` for batch operations:

   .. code-block:: bash

      ufish predict-imgs ./large_dataset ./results --parallel $(nproc)

3. **Memory Management**: For large images, use tiling:

   .. code-block:: bash

      ufish predict large_image.tiff spots.csv --tile-size 1024 --overlap 128

4. **Progress Monitoring**: Use ``--verbose`` for detailed progress:

   .. code-block:: bash

      ufish predict-imgs ./images ./results --verbose --parallel 4 