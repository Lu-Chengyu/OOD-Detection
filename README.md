## Overview

This script performs several preprocessing and analysis steps on feature data, including normalization, dimensionality reduction using Incremental PCA, and calculation of Mahalanobis distances. The results are used to identify out-of-distribution (OOD) samples and to intersect these samples with specific test sets. The script generates several output files that can be used for further analysis.

## Requirements

- Python 3
- NumPy
- SciPy
- scikit-learn
- Memory-mapped files support

## Steps

1. **Normalization**:
    - Load training and test feature data from memory-mapped files.
    - Normalize the training features using `StandardScaler`.
    - Save the scaler's mean and scale.
    - Normalize the test features using the saved scaler parameters.

2. **Incremental PCA**:
    - Perform Incremental PCA on normalized training and test feature data to reduce dimensionality to 400 components.
    - Save the transformed data to memory-mapped files.

3. **Mahalanobis Distance Calculation**:
    - Load the PCA-transformed training and test feature data.
    - Calculate the inverse covariance matrix of the training data.
    - Compute Mahalanobis distances between the test and training samples.
    - Identify samples with sum of distances above a threshold and the smallest 5% sums.

4. **Finding OOD Images**:
    - Use the indices of OOD samples to find the corresponding lines in the original test set file.
    - Intersect the OOD samples with different test conditions and save the intersection results.

## Usage

1. **Set Up Paths**:
    - Update the `featureDir`, `line_numbers_file`, `original_file`, and `testFile` paths according to your file structure.

2. **Run the Script**:
    - Execute the script in your Python environment.

3. **Output Files**:
    - `scaler_mean.npy`: Mean values used for normalization.
    - `scaler_scale.npy`: Scale values used for normalization.
    - `indices_above_threshold.txt`: Indices of samples with sum of distances above the threshold.
    - `indices_5_percentD.txt`: Indices of samples with the smallest 5% sums.
    - `output.txt`: OOD samples corresponding to the lines in the original test set.
    - `intersection_result_i.txt`: Intersection of OOD samples with specific test conditions.

## Example

```python
python oodDetection.py
