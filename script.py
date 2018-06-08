import numpy as np
import GPy
import GPyOpt
import pickle
import sys
import getopt
import math
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, beta

from functionDoESpecial import function_names, function_dimensions, functionDoESpecial

def value_generator(x, n_trials=20):
    values = objective(x)
    Ysim = np.random.binomial(n_trials, values)
    return Ysim.reshape(-1, 1)

def expected_improvement(mean_values, std_values, opt_value):
    
    improvement = (opt_value - mean_values).ravel()
    std_values = std_values.ravel()
    EI = improvement * norm.cdf(improvement / std_values) + std_values * norm.pdf(improvement / std_values)
    
    return EI

def expected_improvement_approx(mean_values, std_values, opt_value, binomial, n_sample=500):
    
    EI = []
    
    for mean, std in zip(mean_values, std_values):
        samples = np.random.normal(mean, std, n_sample)
        samples = samples[samples<opt_value]
        EI.append(np.mean(binomial.gp_link.transf(opt_value) - binomial.gp_link.transf(samples)))
        
    return np.array(EI)

def fidelity_decision(low_trials, successful, min_value, treshold_proba=0.5):
    
    n = low_trials
    k = successful
    posterior_ps = beta(k+1, n-k+1)
    
    if posterior_ps.cdf(min_value) > treshold_proba:
        return True
    return False

def get_new_point(model, lower_bounds, upper_bounds, data=None,
                  multistart=10, seed=None, method='gaussian', n_sample=500):
    """
    Parameters:
        model - GP model of the objective function
        lower_bounds, upper_bounds - array-like, lower and upper bounds of x
        data - tuple(x_training, y_training)
        multistart - number of multistart runs
        seed - np.random.RandomState
        method - gaussian or approximated
        n_sample - number of points for approximated EI calculation
    Returns
        tuple - argmin of the objective function and min value of the objective
    """
    if seed is not None:
        np.random.seed(seed)
    lower_bounds = np.array(lower_bounds).reshape(1, -1)
    upper_bounds = np.array(upper_bounds).reshape(1, -1)

    random_initial_points = np.random.uniform(lower_bounds, upper_bounds, size=(multistart, lower_bounds.shape[1]))

    def objective(x):
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        if method=='gaussian':
            mean_values, variance = model.predict(x)
            std_values = np.sqrt(variance)
            
            return -expected_improvement(mean_values, std_values, data[1].min())
        
        elif method=='laplace':
            mean_values, variance = model._raw_predict(x)
            std_values = np.sqrt(variance)
            
            return -expected_improvement_approx(mean_values, std_values, data[1].min(), GPy.likelihoods.Binomial(), n_sample)

    best_result = None
    best_value = np.inf
    for random_point in random_initial_points:
        
        try:
            result = minimize(objective, random_point, method='L-BFGS-B',
                              bounds=np.vstack((lower_bounds, upper_bounds)).T)
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        except:
            print("bad point")

    return best_result.x, best_result.fun

def optimization_step(training_points, training_values, objective, trials=None, n_trials_low=20, 
                      n_trials_high=np.nan, lower_bounds=None, upper_bounds=None, kernel=GPy.kern.RBF(1), 
                      method='gaussian', treshold_proba=0.5):
    
    if trials.ndim != 2:
        trials = trials.reshape(-1, 1)
    
    if method=='gaussian':
        model = GPy.models.GPRegression(training_points, training_values / trials, kernel)
        
    elif method=='laplace':
        binomial = GPy.likelihoods.Binomial()
        model = GPy.core.GP(training_points, training_values, kernel=GPy.kern.RBF(1), 
                              Y_metadata={'trials': trials},
                              inference_method=GPy.inference.latent_function_inference.laplace.Laplace(),
                              likelihood=binomial)
    else:
        raise ValueError("method must be gaussian or laplace.")
        
    model.optimize_restarts(num_restarts=10, verbose=False)
        
    new_point, criterion_value = get_new_point(model, data=(training_points, training_values),
                                               lower_bounds=lower_bounds, upper_bounds=upper_bounds, method=method)
    
    new_point = new_point.reshape(1, -1)
    new_value = np.asarray(objective(new_point, n_trials_low)).reshape(1, -1)
    new_trials = n_trials_low
    training_points = np.vstack([training_points, new_point])
    
    if (n_trials_high > 0) and (method == 'laplace'):

        if fidelity_decision(n_trials_low, new_value, 
                             model.likelihood.gp_link.transf(np.min(model._raw_predict(training_points)[0])), 
                             treshold_proba):

            new_value = new_value + objective(new_point, n_trials_high)
            new_trials += n_trials_high
            
    
    trials = np.vstack([trials, np.array([[new_trials]])])
    training_values = np.vstack([training_values, new_value])
        
    return training_points, training_values, trials, model

def pipeline(f_name, dims=3, n_iter=60, repeats=10, low=-2, high=2, init_design_amount=5, n_trials=20,
             proba_range=[0, 1]):
    
    objective = lambda x: functionDoESpecial(x.reshape(1, -1), f_name)
    if f_name in function_dimensions.keys():
        dims = function_dimensions[f_name]
    
    lower_bounds = [low] * dims
    upper_bounds = [high] * dims
    
    space = []
    for i in range(len(lower_bounds)):
        space.append({'name': 'x'+str(i), 'type': 'continuous', 'domain': (lower_bounds[i], upper_bounds[i])})

    feasible_region = GPyOpt.Design_space(space=space)
    init_design = GPyOpt.experiment_design.initial_design('random', feasible_region, init_design_amount)
    #search max and min
    argmin = differential_evolution(objective, [(low, high)] * dims).x
    argmax = differential_evolution(lambda x: -1 * objective(x), [(low, high)] * dims).x
    max_v = objective(argmax)
    min_v = objective(argmin)
    #normalize function
    objective = lambda x: (functionDoESpecial(x, f_name) * 0.95 - min_v) / (max_v - min_v)
    
    def value_generator(x, n_trials=20):
        values = objective(x)
        Ysim = np.random.binomial(n_trials, values)
        return Ysim.reshape(-1, 1)
    
    performance_per_n_trials_m = []

    for proba in proba_range:

        stat_per_attempt = []

        for n_attempts in range(repeats):

            X_m = init_design
            Y_m = value_generator(X_m).reshape(-1, 1)
            trials_m = np.ones(X_m.shape[0]).reshape(-1, 1) * n_trials

            m_m = GPy.core.GP(X_m, Y_m, kernel=GPy.kern.RBF(1), 
                              Y_metadata={'trials': trials_m},
                              inference_method=GPy.inference.latent_function_inference.laplace.Laplace(),
                              likelihood=GPy.likelihoods.Binomial())

            lik = GPy.likelihoods.Bernoulli()
            model_mins_m = []
            model_mins_m.append(lik.gp_link.transf(np.min(m_m._raw_predict(X_m)[0])))

            for n_iteration in range(n_iter):

                X_m, Y_m, trials_m, m_m = optimization_step(X_m, Y_m, value_generator,
                                                            lower_bounds=lower_bounds,
                                                            upper_bounds=upper_bounds,
                                                            trials=trials_m, method='laplace', 
                                                            treshold_proba=proba,
                                                            n_trials_low=n_trials, n_trials_high=n_trials)
                model_mins_m.append(lik.gp_link.transf(np.min(m_m._raw_predict(X_m)[0])))

            stat_per_attempt.append([np.array(model_mins_m), trials_m.reshape(-1, 1)])

        performance_per_n_trials_m.append(stat_per_attempt)

    performance_per_n_trials_m = np.array(performance_per_n_trials_m)
    
    return performance_per_n_trials_m

if __name__=='__main__':
    
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "hm:n:", ["name="])
    except getopt.GetoptError:
        print("Wrong options were used.\n")
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == "--name":
            f_name = arg
            if f_name not in function_names:
                print('Wrong name.\n')
                sys.exit()
        else:
            sys.exit(2)
            
    history = pipeline(f_name, proba_range=[1, 0.7, 0.5, 0.3, 0])
    with open(f_name+'.pickle', 'wb') as f:
        pickle.dump(history, f)