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
        if len(samples) > 0:
            EI.append(np.mean(opt_value - binomial.gp_link.transf(samples)))
        else:
            EI.append(0)
        
    return np.array(EI)

def fidelity_decision(low_trials, successful, min_value, latent_min_value=None, ei_mean=None, ei_std=None, treshold_proba=0.5):
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
    
    if (ei_mean is not None) and (ei_std is not None) and (latent_min_value is not None):
        
        treshold_proba = norm.cdf(latent_min_value, loc=ei_mean, scale=ei_std)
    
    n = low_trials
    k = successful
    posterior_ps = beta(k+1, n-k+1)
    
    if posterior_ps.cdf(min_value) > treshold_proba:
        return True
    return False

def get_new_point(model, bounds, opt_value, initial_points=None, seed=None, 
                  method='gaussian', n_sample=500, constraints=None, max_iter=5):

    if seed is not None:
        np.random.seed(seed)

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


    problem = GPyOpt.methods.BayesianOptimization(acquisition, bounds, constraints, exact_feval=True, 
                                                  initial_design_numdata=100, batch_size=100)
    problem.run_optimization(max_iter)
    
    return problem.x_opt, problem.fx_opt

def optimization_step(training_points, training_values, objective, space, trials=None, n_trials_low=20, 
                      n_trials_high=np.nan, kernel=GPy.kern.RBF(1), 
                      method='gaussian', treshold_proba=0.5, constraints=None, dinamic_treshold=False):
    
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

    new_point, criterion_value = get_new_point(model, opt_value=np.min(training_values/trials), method=method,
                                               constraints=constraints, bounds=space)
    
    new_point = new_point.reshape(1, -1)
    new_value = np.asarray(objective(new_point, n_trials_low)).reshape(1, -1)
    new_trials = n_trials_low
    training_points = np.vstack([training_points, new_point])
    
    if (n_trials_high >= n_trials_low+1) and (method == 'laplace'):

        ei_mean = None
        ei_std = None
        latent_min = None
        
        if dinamic_treshold:
            
            trials_t = np.vstack([trials, np.array([[new_trials]])])
            training_values_t = np.vstack([training_values, new_value])

            binomial = GPy.likelihoods.Binomial()
            model_t = GPy.core.GP(training_points, training_values_t, kernel=kernel, 
                                  Y_metadata={'trials': trials_t},
                                  inference_method=GPy.inference.latent_function_inference.laplace.Laplace(),
                                  likelihood=binomial)
            model_t.optimize_restarts(num_restarts=10, verbose=False)
            
            ei_point, criterion_value = get_new_point(model_t, opt_value=np.min(training_values_t/trials_t), method=method,
                                                      constraints=constraints, bounds=space)
                
            ei_mean, ei_std = model_t._raw_predict(ei_point.reshape(1, -1))
            latent_min = np.min(model_t._raw_predict(training_points)[0])
            ei_mean = ei_mean[0,0]
            ei_std = ei_std[0,0]
            
        if fidelity_decision(n_trials_low, new_value, 
                             model.likelihood.gp_link.transf(np.min(model._raw_predict(training_points)[0])), latent_min,
                             ei_mean, ei_std, treshold_proba):

            new_value = new_value + objective(new_point, n_trials_high-n_trials_low)
            new_trials = n_trials_high
            
    
    trials = np.vstack([trials, np.array([[new_trials]])])
    training_values = np.vstack([training_values, new_value])
        
    return training_points, training_values, trials, model