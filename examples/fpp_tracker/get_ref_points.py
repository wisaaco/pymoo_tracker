import numpy as np
import random
import pickle

from problems import Problem01v3 
from file_utils import get_solution_array

def solutions_to_ref_points(configs):
    """Just format solution file to reference points string"""
    o = get_solution_array(configs.input, n_obj=len(configs.objectives))[-1]
    return o.tolist().__repr__()

def lazy_ref_points(configs):
    o_list = configs.objectives

    if configs.ntw_file is not None:
        ntw = pickle.load(configs.ntw_file)
        problem = Problem01v3(ntw, o_list)
        o_min_list = problem.f_min_list
    else:
        o_min_list = [0.] * len(o_list)

    o = get_solution_array(configs.input, n_obj=len(o_list))[-1]

    for i in range(len(o_list)):
        o[:,i] -= o_min_list[i]
        o[:,i] *= configs.lmb
        o[:,i] += o_min_list[i]

    return o

if __name__ == '__main__':
    from parameters import configs
    random.seed(configs.seed)
    print(lazy_ref_points(configs))



