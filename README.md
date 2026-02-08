# gboost

A gradient boosting machine library implemented from scratch in Go. Supports both regression (MSE) and binary classification (Log Loss), with dataset utilities, model serialization, and a scikit-learn-comparable API.

```go
model := gboost.New(gboost.DefaultConfig())
model.Fit(X, y)
predictions := model.Predict(X)
```

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How Gradient Boosting Works](#how-gradient-boosting-works)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Comparison with scikit-learn](#comparison-with-scikit-learn)
- [Project Structure](#project-structure)
- [References](#references)

## Installation

```bash
go get github.com/ahmedaabouzied/gboost
```

Requires Go 1.21+ (uses generics).

## Quick Start

### Regression

```go
package main

import (
    "fmt"
    "github.com/ahmedaabouzied/gboost"
)

func main() {
    // Training data: y = 2*x1 + 3*x2
    X := [][]float64{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}
    y := []float64{8, 13, 18, 23, 28}

    cfg := gboost.DefaultConfig()
    cfg.NEstimators = 50
    cfg.MaxDepth = 3

    model := gboost.New(cfg)
    model.Fit(X, y)

    fmt.Println(model.PredictSingle([]float64{6, 7})) // ~33
}
```

### Binary Classification

```go
cfg := gboost.DefaultConfig()
cfg.Loss = "logloss"
cfg.NEstimators = 100
cfg.MaxDepth = 3

model := gboost.New(cfg)
model.Fit(XTrain, yTrain) // y values must be 0.0 or 1.0

prob := model.PredictProba(sample)     // P(y=1) for a single sample
probs := model.PredictProbaAll(XTest)  // P(y=1) for all samples
```

### Loading CSV Data

```go
// Load a CSV file with automatic label encoding for non-numeric columns.
// Target column supports negative indexing (-1 = last column).
ds, err := gboost.LoadCSV("data/iris_binary.csv", -1, true)

// Split into train/test with reproducible shuffling.
XTrain, XTest, yTrain, yTest, err := ds.Split(0.2, 42)
```

### Save and Load Models

```go
model.Save("model.json")

loaded, err := gboost.Load("model.json")
loaded.Predict(XTest)
```

## How Gradient Boosting Works

Gradient boosting builds an ensemble of weak learners (decision trees) sequentially. Each tree corrects the errors of the previous ensemble by fitting to the **negative gradient** of the loss function. The final prediction is the sum of all tree outputs, scaled by a learning rate.

### The Algorithm

Given training data `(X, y)` with `n` samples, a differentiable loss function `L(y, F(x))`, and `M` boosting iterations:

**Step 1: Initialize with a constant prediction**

```
F_0(x) = argmin_c  sum( L(y_i, c) )
```

For MSE this is the mean of `y`. For Log Loss this is the log-odds of the positive class:

```
F_0(x) = log(p / (1 - p))    where p = mean(y)
```

**Step 2: For each iteration m = 1, 2, ..., M:**

Compute the negative gradient (pseudo-residuals) for each sample:

```
r_im = -[ dL(y_i, F(x_i)) / dF(x_i) ]    evaluated at F = F_{m-1}
```

Fit a regression tree `h_m(x)` to the pseudo-residuals `r_im`.

Update the model:

```
F_m(x) = F_{m-1}(x) + lr * h_m(x)
```

where `lr` is the learning rate (shrinkage factor).

**Step 3: Output the final model**

```
F_M(x) = F_0(x) + lr * sum( h_m(x) )    for m = 1 to M
```

### Loss Functions

#### Mean Squared Error (Regression)

```
L(y, F) = (1/2) * (y - F)^2
```

The negative gradient (pseudo-residual) is simply the residual:

```
r_i = y_i - F(x_i)
```

The initial prediction is the mean of `y`:

```
F_0 = mean(y)
```

#### Log Loss (Binary Classification)

```
L(y, F) = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

where `p = sigmoid(F) = 1 / (1 + exp(-F))`.

The negative gradient is:

```
r_i = y_i - sigmoid(F(x_i))
```

The initial prediction is the log-odds:

```
F_0 = log(sum(y) / (n - sum(y)))
```

For classification, raw predictions are passed through the sigmoid function to produce probabilities:

```
P(y=1 | x) = sigmoid(F_M(x)) = 1 / (1 + exp(-F_M(x)))
```

### Tree Building

Each tree is built by recursively finding the best binary split. For each internal node:

1. For every feature and every unique threshold value in that feature, evaluate the split
2. The best split maximizes **variance reduction** (information gain):

```
Gain = Var(parent) - ( (n_left/n) * Var(left) + (n_right/n) * Var(right) )
```

3. Recursion stops when:
   - Maximum depth is reached
   - A node has fewer samples than `MinSamplesLeaf`
   - No valid split improves variance

Leaf nodes predict the **mean of the pseudo-residuals** that reach them.

### Learning Rate (Shrinkage)

The learning rate `lr` (default 0.1) scales each tree's contribution. Smaller values require more trees but generally produce better generalization:

```
F_m(x) = F_{m-1}(x) + lr * h_m(x)
```

This is the "slow learning" principle from Friedman (2001): taking many small steps in function space outperforms taking fewer large steps.

### Subsampling

When `SubsampleRatio < 1.0`, each tree is trained on a random subset of the training data. This introduces stochasticity that can reduce overfitting, as described in Friedman (2002).

## API Reference

### Config

```go
type Config struct {
    NEstimators    int     // Number of boosting rounds (trees). Default: 100
    LearningRate   float64 // Shrinkage factor per tree. Default: 0.1
    MaxDepth       int     // Maximum depth of each tree. Default: 6
    MinSamplesLeaf int     // Minimum samples required in a leaf. Default: 1
    SubsampleRatio float64 // Fraction of samples used per tree. Default: 1.0
    Loss           string  // "mse" for regression, "logloss" for classification. Default: "mse"
}

func DefaultConfig() Config
```

### GBM

```go
func New(cfg Config) *GBM

func (g *GBM) Fit(X [][]float64, y []float64) error
func (g *GBM) Predict(X [][]float64) []float64         // Raw predictions (regression or log-odds)
func (g *GBM) PredictSingle(x []float64) float64        // Raw prediction for one sample
func (g *GBM) PredictProba(x []float64) float64          // P(y=1) for one sample (classification)
func (g *GBM) PredictProbaAll(X [][]float64) []float64   // P(y=1) for all samples (classification)
func (g *GBM) Save(path string) error                    // Save model to JSON
func Load(path string) (*GBM, error)                      // Load model from JSON
```

### Dataset Utilities

```go
type Dataset struct {
    X              [][]float64
    Y              []float64
    Encodings      map[int]map[string]float64  // Feature label encodings (featureIndex -> string -> value)
    TargetEncoding map[string]float64          // Target label encoding (nil if numeric)
    Header         []string                     // Column names (nil if no header)
}

// Load a CSV file. Non-numeric columns are automatically label-encoded.
// targetColumn supports negative indexing (-1 = last column).
func LoadCSV(path string, targetColumn int, hasHeader bool) (*Dataset, error)

// Split into train/test sets with shuffling.
func TrainTestSplit(X [][]float64, y []float64, testRatio float64, seed int64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error)

// Convenience method on Dataset.
func (ds *Dataset) Split(testRatio float64, seed int64) (XTrain, XTest [][]float64, yTrain, yTest []float64, err error)
```

## Examples

### Regression Example

A synthetic regression demo is included in `cmd/demo/`:

```bash
go run ./cmd/demo/
```

This generates data following `y = 2*x1 + 3*x2 + noise`, trains a GBM, and reports RMSE on train and test sets.

### Binary Classification Example (Iris)

A binary classification example using the Iris dataset is included in `cmd/iris/`:

```bash
go run ./cmd/iris/
```

This loads the Iris dataset (filtered to versicolor vs. virginica), trains a GBM with Log Loss, and reports accuracy and per-sample predictions. It also saves `data/iris_train.csv` and `data/iris_test.csv` for comparison with Python.

**Output:**

```
Loaded 100 samples, 4 features
Train: 80 samples, Test: 20 samples

Saved data/iris_train.csv and data/iris_test.csv

--- Hyperparameters ---
NEstimators:    100
LearningRate:   0.10
MaxDepth:       3
MinSamplesLeaf: 1
SubsampleRatio: 1.00

--- Test Set Predictions ---
Index  Actual   Predicted  Prob(1)
0      1        1          0.8989
1      1        1          0.8989
2      1        0          0.2598
3      1        1          0.8989
4      0        0          0.1010
5      1        1          0.8989
6      0        0          0.1420
7      0        0          0.1010
8      0        0          0.1010
9      1        1          0.8989
10     1        1          0.8989
11     1        1          0.8989
12     0        0          0.1010
13     1        1          0.8989
14     0        0          0.1010
15     0        0          0.1010
16     0        1          0.8989
17     1        1          0.8989
18     0        0          0.1010
19     0        0          0.1010

--- Results ---
Correct: 18 / 20
Accuracy: 90.00%
Train Accuracy: 100.00%
Test Log Loss: 0.2802
```

## Comparison with scikit-learn

To validate correctness, gboost was benchmarked against scikit-learn's `GradientBoostingClassifier` on the Iris dataset (binary: versicolor vs. virginica). Both models were trained on **identical data splits** with **identical hyperparameters**.

### Setup

- 100 samples (50 versicolor, 50 virginica), 4 features (sepal length/width, petal length/width)
- 80/20 train/test split with fixed seed
- Hyperparameters: 100 trees, learning rate 0.1, max depth 3, min samples leaf 1, no subsampling

### Results

| Metric | gboost (Go) | scikit-learn (Python) |
|---|---|---|
| **Test Accuracy** | **90.00%** | **90.00%** |
| **Train Accuracy** | 100.00% | 100.00% |
| **Misclassified Samples** | #2, #16 | #2, #16 |
| **Test Log Loss** | 0.2802 | 0.7246 |

### Per-Sample Probability Comparison

| Sample | Actual | Go Prob(1) | sklearn Prob(1) | Go Pred | sklearn Pred |
|---|---|---|---|---|---|
| 0 | 1 | 0.8989 | 1.0000 | 1 | 1 |
| 1 | 1 | 0.8989 | 0.9999 | 1 | 1 |
| 2 | 1 | 0.2598 | 0.0042 | 0 | 0 |
| 3 | 1 | 0.8989 | 1.0000 | 1 | 1 |
| 4 | 0 | 0.1010 | 0.0002 | 0 | 0 |
| 5 | 1 | 0.8989 | 1.0000 | 1 | 1 |
| 6 | 0 | 0.1420 | 0.0002 | 0 | 0 |
| 7 | 0 | 0.1010 | 0.0002 | 0 | 0 |
| 8 | 0 | 0.1010 | 0.0002 | 0 | 0 |
| 9 | 1 | 0.8989 | 1.0000 | 1 | 1 |
| 10 | 1 | 0.8989 | 1.0000 | 1 | 1 |
| 11 | 1 | 0.8989 | 0.9996 | 1 | 1 |
| 12 | 0 | 0.1010 | 0.0001 | 0 | 0 |
| 13 | 1 | 0.8989 | 0.9999 | 1 | 1 |
| 14 | 0 | 0.1010 | 0.0001 | 0 | 0 |
| 15 | 0 | 0.1010 | 0.0001 | 0 | 0 |
| 16 | 0 | 0.8989 | 0.9999 | 1 | 1 |
| 17 | 1 | 0.8989 | 1.0000 | 1 | 1 |
| 18 | 0 | 0.1010 | 0.0002 | 0 | 0 |
| 19 | 0 | 0.1010 | 0.0002 | 0 | 0 |

### Analysis

**Both models produce identical predictions on all 20 test samples**, including the same two misclassifications (samples 2 and 16). This confirms the core gradient boosting algorithm is implemented correctly.

The key difference is in **probability calibration**:

- **scikit-learn** produces extreme probabilities (0.9999, 0.0001) because it uses Newton-Raphson optimization for leaf values, computing optimal weights using both the first derivative (gradient) and second derivative (Hessian) of the loss function. This makes each tree more effective, pushing raw scores further from zero over 100 iterations.

- **gboost** produces moderate probabilities (0.8989, 0.1010) because it uses the simpler approach of setting leaf values to the mean of pseudo-residuals. This is the basic formulation from Friedman (2001) before the Newton step optimization.

The practical effect: gboost actually achieves **better log loss** (0.2802 vs 0.7246) on this test set because its moderate confidence on the two wrong predictions incurs less penalty than scikit-learn's overconfident wrong predictions. In general, however, the Newton-Raphson leaf optimization improves convergence speed and is an area for future enhancement.

### Reproducing the Comparison

**Go (run locally):**

```bash
go run ./cmd/iris/
```

This produces `data/iris_train.csv` and `data/iris_test.csv`.

**Python (run in Colab or locally):**

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

train = pd.read_csv("iris_train.csv")
test = pd.read_csv("iris_test.csv")

X_train = train.drop("label", axis=1).values
y_train = train["label"].values
X_test = test.drop("label", axis=1).values
y_test = test["label"].values

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, preds) * 100))
print("Train Accuracy: {:.2f}%".format(
    accuracy_score(y_train, model.predict(X_train)) * 100
))
print("Test Log Loss: {:.4f}".format(log_loss(y_test, probs)))
```

## Project Structure

```
gboost/
    config.go          # Config struct and DefaultConfig()
    gboost.go          # GBM struct, Fit, Predict, PredictProba
    tree.go            # Decision tree: Node, Split, buildTree, findBestSplit
    loss.go            # Loss interface, MSELoss, LogLoss
    math.go            # Generic math utilities (mean, sum, variance, sigmoid)
    util.go            # Helper functions (sort, uniq, validation)
    dataset.go         # LoadCSV, TrainTestSplit, Dataset struct
    serialize.go       # JSON Save/Load for model persistence
    errors.go          # Sentinel errors
    *_test.go          # Tests for each module (~97.9% coverage)
    cmd/
        demo/main.go   # Regression example with synthetic data
        iris/main.go   # Classification example with Iris dataset
    data/
        iris_binary.csv # Iris dataset (versicolor vs virginica)
```

## Testing

```bash
go test ./...          # Run all tests
go test -v ./...       # Verbose output
go test -cover ./...   # With coverage report
```

Test coverage is approximately 97.9% across all modules.

## References

1. **Friedman, J.H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232. [[pdf]](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
   - The foundational paper introducing gradient boosting as a general framework for building additive models by sequentially fitting to negative gradients.

2. **Friedman, J.H. (2002).** "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367-378.
   - Introduces stochastic subsampling (training each tree on a random subset of rows), which reduces overfitting and improves generalization.

3. **scikit-learn: GradientBoostingClassifier.** [[documentation]](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)
   - The Python reference implementation used for comparison. scikit-learn extends the basic algorithm with Newton-Raphson leaf optimization (using second derivatives for optimal leaf weights), Friedman's split improvement criterion, and other enhancements.

## License

MIT
