 # Himmelblau function GP model
This model approximates the Himmelblau function:
$
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
$

This model: 
- Generates training and testing data using LHS sampling between $[-5, 5]$ in both dimensions
- Evaluates $f(x, y)$
- Fits a GP with a RBF kernel
- Evaluates model accuracy using RMSE


# How to Run
## Setup

Install uv:
```bash
pip install uv
```

Create environment and install dependencies:
```bash
uv sync
```

# Code Components
## src/
### gp_model/
1. data_processing.py - LHS sampling + standardisation and de-standardisation + converts arrays to tensors 