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
Use your environment of choice:

    pip install -r requirements.txt

# Code Components
## src/
### gp_model/
1. data_processing.py - LHS sampling + standardisation and de-standardisation + converts arrays to tensors 