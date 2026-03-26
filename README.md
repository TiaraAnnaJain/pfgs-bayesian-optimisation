 # Himmelblau function GP model
This model approximates the Himmelblau function:
$$
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
$$

This model: 
- Generates training and testing data using LHS sampling between $[-5, 5]$ in both dimensions
- Evaluates $f(x, y)$
- Fits a GP with a RBF kernel
- Evaluates model accuracy using RMSE


# How to Run
## Dependencies
Install uv:

    pip install uv

Create environment and install dependencies:

    uv sync
