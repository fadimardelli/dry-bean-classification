# Softmax Regression Implementation Guide

## Overview

Yes, it is **absolutely possible** to create a categorical multinomial model (Softmax Regression) with the available code! The `logistic_regression.py` file already contains all the necessary functions.

## Available Functions

### 1. `my_softmax(z)`
- **Purpose**: Converts logits to probability distribution
- **Input**: 1D array (N,) or 2D array (m, K)
- **Output**: Softmax probabilities (same shape as input)
- **Note**: Updated to handle both 1D and 2D arrays

### 2. `gradient_descent_softmax(X, Y_one_hot, initial_all_w, initial_all_b, alpha, num_iters)`
- **Purpose**: Trains a softmax regression model
- **Input**:
  - `X`: Features (m, n)
  - `Y_one_hot`: One-hot encoded labels (m, K)
  - `initial_all_w`: Initial weights (K, n)
  - `initial_all_b`: Initial biases (K, 1)
  - `alpha`: Learning rate
  - `num_iters`: Number of iterations
- **Output**: `(all_w, all_b, J_history)`
  - `all_w`: Learned weights (K, n)
  - `all_b`: Learned biases (K, 1)
  - `J_history`: Cost history

### 3. `predict_multiclass_softmax(X, all_w, all_b)`
- **Purpose**: Makes predictions using softmax model
- **Input**:
  - `X`: Features (m, n)
  - `all_w`: Weights (K, n)
  - `all_b`: Biases (K, 1)
- **Output**: `(predictions, probabilities)`
  - `predictions`: Class predictions (m,)
  - `probabilities`: Softmax probabilities (m, K)

## Implementation Steps

### Step 1: Prepare One-Hot Encoded Labels

```python
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse_output=False)
Y_train_onehot = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
Y_cv_onehot = onehot_encoder.transform(y_cv.reshape(-1, 1))
Y_test_onehot = onehot_encoder.transform(y_test.reshape(-1, 1))
```

### Step 2: Initialize Parameters

```python
# Initialize weights and biases for all K classes
initial_all_w = np.zeros((num_classes, n_features))  # (K, n)
initial_all_b = np.zeros((num_classes, 1))  # (K, 1)
```

### Step 3: Train the Model

```python
lg = LogisticRegression_lr()

alpha = 0.01  # Learning rate
num_iters = 1000  # Number of iterations

all_w, all_b, J_history = lg.gradient_descent_softmax(
    X_train_scaled, 
    Y_train_onehot, 
    initial_all_w, 
    initial_all_b, 
    alpha, 
    num_iters
)
```

### Step 4: Make Predictions

```python
y_pred, y_proba = lg.predict_multiclass_softmax(X_test_scaled, all_w, all_b)
```

### Step 5: Evaluate

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Verify probabilities sum to 1
print(f"Probability sum: {np.sum(y_proba[0]):.6f}")  # Should be 1.0
```

## Key Differences: One-vs-All vs Softmax

| Aspect | One-vs-All | Softmax Regression |
|--------|-----------|-------------------|
| **Number of Models** | K separate binary models | 1 unified model |
| **Training** | Train K times | Train once |
| **Probabilities** | May not sum to 1 | Always sum to 1 |
| **Class Competition** | No (independent) | Yes (compete for probability) |
| **Loss Function** | Binary Cross-Entropy (K times) | Categorical Cross-Entropy (once) |
| **Activation** | Sigmoid (K times) | Softmax (once) |
| **Parameters** | K × (n+1) separate | K × (n+1) unified |

## Advantages of Softmax Regression

1. **Single Unified Model**: All classes learned together
2. **Valid Probability Distribution**: Probabilities always sum to 1
3. **Class Competition**: Classes compete for probability mass
4. **Often Better Performance**: Usually achieves higher accuracy
5. **More Efficient**: Single training pass instead of K passes

## Code Fix Applied

The `my_softmax()` function has been updated to handle both:
- **1D arrays**: Original implementation for single examples
- **2D arrays**: Vectorized version for batch processing (applies softmax row-wise)

This ensures compatibility with both `gradient_descent_softmax()` and `predict_multiclass_softmax()`.

## Example Usage in Notebook

The notebook `02_model1_LogisticRegression_Lab.ipynb` now includes:
- Cell 47: Markdown explanation of Softmax Regression
- Cell 48: Training code
- Cell 49: Evaluation code
- Cell 50: Comparison with One-vs-All
- Cell 51: Classification report
- Cell 52: Cost history plot
- Cell 53: Confusion matrix

## Summary

**Yes, you can create a categorical multinomial model!** The code already supports it through:
- `gradient_descent_softmax()` for training
- `predict_multiclass_softmax()` for predictions
- `my_softmax()` for probability computation

Simply use one-hot encoded labels and the softmax methods instead of the One-vs-All approach.

