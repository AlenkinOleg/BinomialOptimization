import numpy as np
import GPy
import GPyOpt
import pickle
import sys
import getopt
import math
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, beta


def value_generator(x, objective, n_trials=20):
    """
    Generates sample from Bi(objective(x), n_trials)
    Parameters:
    
        x             - point of parameter space where we want to get a sample;
        objective     - function: X->probability_space, of course it should have values from 0 to 1;
        n_trials      - the second parameter of binomial distribution.
        
    Returns:
    
        Generated sample.
    """
    
    values = objective(x)
    Ysim = np.random.binomial(n_trials, values)
    
    return Ysim.reshape(-1, 1)

def expected_improvement(mean_values, std_values, opt_value):
    """
    Expected imrovement acquisition function for classic Bayesian optimization.
    Parameters:
    
        mean_values     - mean of gaussian distribution;
        std_values      - standard deviation;
        opt_value       - current best value of objective function.
        
    Returns:
    
        Value of acquisition function.
    """
    
    improvement = (opt_value - mean_values).ravel()
    std_values = std_values.ravel()
    EI = improvement * norm.cdf(improvement / std_values) + std_values * norm.pdf(improvement / std_values)
    
    return EI

def expected_improvement_approx(mean_values, std_values, opt_value, binomial, n_sample=500):
    """
    Expected imrovement acquisition function for approximated inference.
    Parameters:
    
        mean_values     - mean of gaussian distribution for latent variable;
        std_values      - standard deviation for latent variable;
        opt_value       - current best value of objective function;
        binomial        - GPy binomial likelihood;
        n_sample        - number of samples from distribution.
    
    Returns:
    
        Value of acquisition function.
    """
    
    EI = []
    
    for mean, std in zip(mean_values, std_values):
        samples = np.random.normal(mean, std, n_sample)
        samples = samples[binomial.gp_link.transf(samples)<opt_value]
        EI.append(np.mean(opt_value - binomial.gp_link.transf(samples)))
        
    return np.array(EI)

def fidelity_decision(low_trials, successful, min_value, treshold_proba=0.5):
    """
    Rule for making decision: continue investigate this point or move to another using EI acquisition function.
    Parameters:
        
        low_trials      - low number of samples already generated;
        successful      - number of successful trials;
        min_value       - current optimal value;
        treshold_proba  - if probability to beat current minimum more than this treshold we deside to continue investigate the point.
    
    Returns:
        
        Decision, boolean.
    """
    
    n = low_trials
    k = successful
    posterior_ps = beta(k+1, n-k+1)
    
    if posterior_ps.cdf(min_value) > treshold_proba:
        return True
    return False

def get_new_point(model, lower_bounds, upper_bounds, opt_value,
                  multistart=10, seed=None, method='gaussian', n_sample=500, constraints=None, optimization_method='L-BFGS-B'):
    """
    
    Parameters:
    
        model                                - GP or GGPM model of the objective function;
        lower_bounds, upper_bounds           - array-like, lower and upper bounds of x;
        multistart                           - number of multistart runs;
        seed                                 - np.random.RandomState;
        method                               - gaussian or approximated;
        opt_value                            - current optimal value;
        n_sample                             - number of points for approximated EI calculation;
        constraints                          - constraints on parameters;
        optimization_method                  - method for acquisition function optimization.
    
    Returns:
    
        tuple - argmin of the objective function and min value of the objective
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    lower_bounds = np.array(lower_bounds).reshape(1, -1)
    upper_bounds = np.array(upper_bounds).reshape(1, -1)

    random_initial_points = np.random.uniform(lower_bounds, upper_bounds, size=(multistart, lower_bounds.shape[1]))

    def acquisition(x):
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        if method=='gaussian':
            mean_values, variance = model.predict(x)
            std_values = np.sqrt(variance)
            
            return -expected_improvement(mean_values, std_values, opt_value)
        
        elif method=='laplace':
            mean_values, variance = model._raw_predict(x)
            std_values = np.sqrt(variance)
            
            return -expected_improvement_approx(mean_values, std_values, opt_value, GPy.likelihoods.Binomial(), n_sample)

    best_result = None
    best_value = np.inf
    for random_point in random_initial_points:
        
        try:
            result = minimize(acquisition, random_point, method=optimization_method, 
                              bounds=np.vstack((lower_bounds, upper_bounds)).T,
                              constraints=constraints)
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        except:
            print("bad point")

    return best_result.x, best_result.fun

def optimization_step(training_points, training_values, objective, trials=None, n_trials_low=20, 
                      n_trials_high=np.nan, lower_bounds=None, upper_bounds=None, kernel=GPy.kern.RBF(1), 
                      method='gaussian', treshold_proba=0.5, constraints=None):
    
    if trials.ndim != 2:
        trials = trials.reshape(-1, 1)
    
    if method=='gaussian':
        model = GPy.models.GPRegression(training_points, training_values / trials, kernel)
        
    elif method=='laplace':
        binomial = GPy.likelihoods.Binomial()
        model = GPy.core.GP(training_points, training_values, kernel=kernel, 
                              Y_metadata={'trials': trials},
                              inference_method=GPy.inference.latent_function_inference.laplace.Laplace(),
                              likelihood=binomial)
    else:
        raise ValueError("method must be gaussian or laplace.")
        
    model.optimize_restarts(num_restarts=10, verbose=False)
    
    if constraints:
        new_point, criterion_value = get_new_point(model, opt_value=np.min(training_values/trials),
                                                   lower_bounds=lower_bounds, upper_bounds=upper_bounds, method=method,
                                                   constraints=constraints, optimization_method='SLSQP')
    else:
        new_point, criterion_value = get_new_point(model, opt_value=np.min(training_values/trials),
                                                   lower_bounds=lower_bounds, upper_bounds=upper_bounds, method=method,
                                                   optimization_method='L-BFGS-B')
    
    new_point = new_point.reshape(1, -1)
    new_value = np.asarray(objective(new_point, n_trials_low)).reshape(1, -1)
    new_trials = n_trials_low
    training_points = np.vstack([training_points, new_point])
    
    if (n_trials_high >= n_trials_low+1) and (method == 'laplace'):

        if fidelity_decision(n_trials_low, new_value, 
                             model.likelihood.gp_link.transf(np.min(model._raw_predict(training_points)[0])), 
                             treshold_proba):

            new_value = new_value + objective(new_point, n_trials_high-n_trials_low)
            new_trials = n_trials_high
            
    
    trials = np.vstack([trials, np.array([[new_trials]])])
    training_values = np.vstack([training_values, new_value])
        
    return training_points, training_values, trials, model