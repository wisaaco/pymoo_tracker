import os
# os.system('export PYTHONPATH=${PYTHONPATH}:~/Projects/pymoo_tracker')

# 
# export PYTHONPATH=${PYTHONPATH}:~/Projects/pymoo_tracker
# 

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.config import Config
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.util.plotting import plot


from datetime import datetime
import pandas as pd
import sys
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.unsga3 import UNSGA3

np.random.seed(2024)

problems = [get_problem("zdt1"),get_problem("zdt2"),get_problem("zdt3"),get_problem("zdt4"),get_problem("zdt5")]
# problem = get_problem("zdt2")

problem_names = ["zdt%i"%i for i in range(1,6)]
algorithm_names = ["SMSEMOA","NSGA2","UNSGA3","CTAEA"]

#DEBUG
# algorithm_names = ["SMSEMOA"]
# problem_names = ["zdt1"]

pop_size = 100
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithms = [
     SMSEMOA(pop_size=pop_size),
     NSGA2(pop_size=pop_size,eliminate_duplicates=True,save_history = False),
     UNSGA3(ref_dirs, pop_size=pop_size),
     CTAEA(ref_dirs=ref_dirs)
]



for ia, algorithm in enumerate(algorithms):
    print(algorithm_names[ia])
    for ip, problem in enumerate(problems):
        print("\t",problem_names[ip])
# problem = get_problem("c2dtlz2", None, 3, k=5)
        res = minimize(problem,
                    algorithm,
                    ('n_gen', 200),
                    seed=1,
                    save_history = True,
                    save_tracker = True,
                    verbose= False
                    )

        rel_path = os.path.dirname(__file__) #TODO improve this part
        file_tracker = open(rel_path+"/data/test_tracker_%s_%s.csv"%(algorithm_names[ia],problem_names[ip]), "wb+")
        coreDF = pd.DataFrame(columns=algorithm.record_tracker_columns)
        coreDF.to_csv(file_tracker, index=False, encoding='utf-8')

        for igen in range(len(res.history_pop)):
            f_idx_objectives = list()

            for o in res.history_pop[igen]["opt"]:
                    f_idx_objectives.append(o.data["idx"])
                
            data = []
            for individual in res.history_pop[igen]["pop"]:
                ind_data = individual.data 
                if "idx" not in ind_data:
                    print("HERE a PROBLEM")
                else:
                    if ind_data["idx"] in f_idx_objectives: #improve
                        ind_data["isF"] = True
                    else:
                        ind_data["isF"] = False
                data.append(ind_data)

            df = pd.json_normalize(data)
            df["iter"] = igen+1 #move line
            if "mutate_rate" not in df.columns: #TODO This logic should be controlleld in the long
                df[["mutate_rate","mutate"]] = None
            if "crowding" not in df.columns:
                df["crowding"] = None
            if "rank" not in df.columns:
                df["rank"] = None
            df.to_csv(file_tracker,index=False,header=False,columns=algorithm.record_tracker_columns)