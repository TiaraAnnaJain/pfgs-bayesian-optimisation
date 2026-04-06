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
1. data_processing.py - Himmelblau function, data generation, LHS sampling 
2. data_models.py - Data model for GP + standardisation and de-standardisation

### gp_model/
1. model.py - Gaussian Process model. 
2. analysis.py - Computes rmse and rrmse. 

### pareto_front/
1. gp_problem.py - Problem wrapper for GP model. 
2. optimizer.py - Runs NSGA-2 to generate pareto front
3. optimal_point.py - Calculates the optimal point on the pareto front with reference to the utopia point.
4. alpha_calculator.py - Calculates the UCB coefficient of the optimal point on pareto front. 