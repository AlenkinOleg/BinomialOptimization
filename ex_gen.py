import numpy as np
import GPy
import GPyOpt
import math
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import pickle
import sys
import getopt
import json

from functionDoESpecial import functionDoESpecial, function_dimensions, function_names
from binomial_optimization2 import optimization_step, value_generator

if __name__=='__main__':
    
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "hm:n:", ["f_name=", "bound=", "low_f=", "high_f=", "dims=", "initial_size=",\
                                                   "n_iteration=", "init_strategy=", "n_attempts="])
    except getopt.GetoptError:
        print("Wrong options were used.\n")
        sys.exit(2)
        
    low_f = 25
    high_f = 75
    n_iteration = 80
    initial_size = 15
    init_strategy = 'random'
    n_attempts = 10
    bound = 1
    dims = 2
        
    for opt, arg in opts:
        if opt == "--f_name":
            f_name = arg
            if f_name not in function_names:
                print('Wrong name.\n')
                sys.exit()
        elif opt == "--bound":
            bound = int(arg)
        elif opt == "--low_f":
            low_f = int(arg)
        elif opt == "--high_f":
            high_f = int(arg)
        elif opt == "--dims":
            dims = int(arg)
        elif opt == "--initial_size":
            initial_size = int(arg)
        elif opt == "--n_iteration":
            n_iteration = int(arg)
        elif opt == "--init_strategy":
            init_strategy = arg
        elif opt == "--n_attempts":
            n_attempts = int(arg)
        else:
            sys.exit(2)

    objective = lambda x: functionDoESpecial(x.reshape(1, -1), f_name)
    if f_name in function_dimensions.keys():
        dims = function_dimensions[f_name]

    lower_bounds = [-1 * bound] * dims
    upper_bounds = [bound] * dims

    space = []
    for i in range(len(lower_bounds)):
        space.append({'name': 'x'+str(i), 'type': 'continuous', 'domain': (lower_bounds[i], upper_bounds[i])})

    feasible_region = GPyOpt.Design_space(space=space)
    init_design = GPyOpt.experiment_design.initial_design(init_strategy, feasible_region, initial_size)

    #search max and min
    argmin = differential_evolution(objective, [(-bound, bound)] * dims).x
    argmax = differential_evolution(lambda x: -1 * objective(x), [(-bound, bound)] * dims).x
    max_v = objective(argmax)
    min_v = objective(argmin)
    #normalize function
    objective = lambda x: (functionDoESpecial(x, f_name) * 0.95 - min_v) / (max_v - min_v)

    init_values = value_generator(init_design, objective, n_trials=low_f).reshape(-1, 1)
    
    ker = GPy.kern.RBF(dims)
    
    agg = []

    for fid in [low_f, high_f]:

        mult = []

        for attempt in range(n_attempts):

            X = init_design
            Y = init_values * int(fid / low_f)
            trials = np.ones(len(X)).reshape(-1, 1) * low_f * int(fid / low_f)

            model_mins = []
            model_argmins = []

            for i in range(n_iteration):
                v = lambda x, n_trials: value_generator(x, objective, n_trials)
                X, Y, trials, m = optimization_step(X, Y, v, lower_bounds=lower_bounds,\
                                                    upper_bounds=upper_bounds, trials=trials,\
                                                    n_trials_low=fid, kernel=ker)
                model_mins.append(float(np.min(m.predict(X)[0])))
                model_argmins.append(int(np.argmin(m.predict(X)[0])))

            mult.append([model_mins, X.astype(float).tolist(), Y.astype(int).tolist(), trials.astype(int).tolist(),\
                         model_argmins])

        agg.append(mult)

    for p, d_fid in [(0, False), (1, False), (0.5, False), (0.5, True)]:

        mult = []

        for attempt in range(n_attempts):

            X = init_design
            if p < 1:
                Y = init_values
                trials = np.ones(len(X)).reshape(-1, 1) * low_f
            else:
                Y = init_values * int(high_f / low_f)
                trials = np.ones(len(X)).reshape(-1, 1) * int(high_f / low_f)

            lik = GPy.likelihoods.Bernoulli()
            model_mins = []
            model_argmins = []

            for i in range(n_iteration):
                v = lambda x, n_trials: value_generator(x, objective, n_trials=n_trials)
                X, Y, trials, m = optimization_step(X, Y, v, lower_bounds=lower_bounds,\
                                                    upper_bounds=upper_bounds, trials=trials,\
                                                    n_trials_low=low_f, n_trials_high=high_f,\
                                                    method='laplace', kernel=ker, treshold_proba=p, \
                                                    dinamic_treshold=d_fid)
                model_mins.append(float(lik.gp_link.transf(np.min(m._raw_predict(X)[0]))))
                model_argmins.append(int(np.argmin(m._raw_predict(X)[0])))

            mult.append([model_mins, X.astype(float).tolist(), Y.astype(int).tolist(), trials.astype(int).tolist(),\
                         model_argmins])

        agg.append(mult)
        
    with open('example.txt', 'w') as tf:
        json.dump(agg, tf)
