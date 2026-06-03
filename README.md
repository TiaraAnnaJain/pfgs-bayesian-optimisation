 # Himmelblau function GP model
This model approximates the Himmelblau function:
$
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
$

This model: 
- Generates training and testing data using LHS sampling between $[-5, 5]$ in both dimensions
- Evaluates $f(x, y)$
- Fits a GP with a RBF kernel
- Evaluates model accuracy using RMSE and RRMSE


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
### data/
1. data_processing.py - himmelblau function, data generation, LHS sampling 
2. data_models.py - data model for GP + standardisation and de-standardisation

### gp_model/
1. model.py - Gaussian Process model. 
2. analysis.py - computes rmse and rrmse. 