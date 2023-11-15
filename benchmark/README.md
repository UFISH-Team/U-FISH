# Benchmarking

## Table of Contents

1. [Introduction](#introduction)
2. [Deep Learning methods](#deep-learning-methods)
3. [Ruled-based methods](#rule-based-methods)



## Introduction

U-FISH's performance was benchmarked against deep learning methods and ruled-based methods for single-molecule spot detection in situ hybridization (FISH) images, including U-FISH, DeepBlink, DetNet, SpotLearn, Big-FISH, RS-FISH, Starfish, and TrackMate. These comparisons were conducted on diverse datasets, which included both simulated and real data with significant variability among them. 

In this study, we compared U-FISH with three different deep learning methods: DeepBlink, implemented in Python with TensorFlow; DetNet, implemented in Python using PyTorch; and SpotLearn, also implemented in Python with PyTorch. Notably, SpotLearn's approach of predicting binary masks instead of exact point localization sets it apart. To ensure a thorough analysis, we integrated SpotLearn's network with the U-FISH methodology to obtain comprehensive comparative results.

Given the extensive parameter choices involved in traditional methods, we adopted a grid search approach to pinpoint the optimal parameters for testing. The evaluation metrics used for comparison were the F1-score and mean distance. It should be noted that a single parameter set may not be universally applicable across all datasets in conventional methods due to the inherent variability of the data. To ensure a fair and comprehensive comparison with deep learning methods, we performed a grid search for each type of dataset, identifying an optimal parameter set for each. This methodological approach ensures that the unique characteristics of each dataset are taken into consideration, allowing for a more accurate and equitable comparison of U-FISH's performance with other ruled-based methods in the field.

Unless specifically stated otherwise, the parameters mentioned above are used to conduct grid search for both two-dimensional and three-dimensional benchmarking tasks.


## Deep Learning methods


**U-FISH & deepBlink**

Evaluation of U-FISH and DeepBlink was carried out with default parameters. Despite conducting hyperparameter grid searches, significant improvements were not observed, so we retained the default settings.



**SpotLearn & DetNet**

For a precise assessment of DetNet and SpotLearn, both networks were implemented in the PyTorch framework, following the methods outlined in their respective research papers. We also undertook grid searches for DetNet's linearly distributed sigmoid shifts parameter $\alpha$ and SpotLearn's target process parameter during training. To ensure comprehensiveness, ten different values of alpha and six different values for target process were considered.


## Rule-based methods


**Big-FISH**

We employ dense region decomposition for spot detection. As Big-FISH calculates a default value for each parameter automatically, the grid search range for some parameters was derived from the default value. 
* Grid search parameters:
                 
                    SigmaYX: default + [-0.5, -0.25, 0, 0.25, 0.5]
                    Threshold: default indices: [-6, -3, -2, -1, 0, 1, 2, 3, 6]
                    Alpha: 0.5-0.8, step size += 0.1
                    Beta: 0.8-1.2, step size += 0.1
                    Gamma: 4-6, step size += 0.5

  

**RS-FISH**

Given that RS-FISH possesses a parameter known as intensityThreshold, the selection range for this parameter can vary significantly based on different image data types. To preclude the generation of a substantial computational load, we scaled the seven categories of datasets to (0, 255) prior to executing the subsequent grid search.
* Grid search parameters:
  
                    DoGSigma = 1-2, step size += 0.25
                    DoGthreshold = 0.003-0.06, step size *= 1.5 
                    SupportRadius: 2-4, step size += 1
                    InlierRatio: 0.1
                    MaxError: 1.5000
                    IntensityThreshold: [0, 10, 50, 100, 150, 200, 255]


  
**Starfish**

 Starfish offers several methods for spot detection, and we have chosen the BlobDetector as our analysis pipeline.
* Grid search parameters:
  
                    Sigma = 1-6, step size += 1
                    Threshold = 0.000095-0.15, step size += 0.00005


  
**TrackMate**

Similar to the RS-FISH method, images need to be scaled to the range of (0,  255) when using TrackMate grid search best parameters.
* Grid search parameters:
  
                    Detector: options are 'log', 'dog'
                    Radius = [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
                    Threshold = 0.1-15, step_size *= 1.05 

