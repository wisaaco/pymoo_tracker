from problems import Problem01v3
from problem_tools import *
#from problem_tools import MySampling_v1, MyCrossover_v1, MyCrossover_v2, MyRepair, MyMutation, MyDuplicateElimination, MyCallback
from problem_ilp import ProblemILP

from pymoo.algorithms.moo.nsga2  import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.nsga3  import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.ctaea  import CTAEA
from pymoo.algorithms.moo.sms    import SMSEMOA
from pymoo.algorithms.moo.rvea   import RVEA 

from pymoo.termination import get_termination

from pymoo.operators.sampling.rnd    import IntegerRandomSampling
from pymoo.operators.crossover.sbx   import SBX
from pymoo.operators.mutation.pm     import PM

from pymoo.util.ref_dirs import get_reference_directions

from pymoo.optimize import minimize

import numpy as np
import sys
import os
import pandas as pd

algdict = {
        'NSGA2': NSGA2,
        'RNSGA2': RNSGA2,
        'NSGA3': NSGA3,
        'UNSGA3': UNSGA3,
        'RNSGA3': RNSGA3,
        'CTAEA': CTAEA,
        'SMSEMOA': SMSEMOA,
        'RVEA': RVEA
    }

sampling_v = (
        MySampling_v1,
        MySampling_v2,
        MySampling_v3
    )

crossover_v = (
        MyCrossover_v1,
        MyCrossover_v2,
        MyCrossover_v3
    )

mutation_v = (
        MyMutation_v1,
        MyMutation_v2,
        MyMutation_v3,
        MyMutation_v4
    )

def solve(ntw, configs):

    
    o_list = configs.objectives

    if configs.algorithm == 'ILP' and len(o_list) == 2:
        problem = ProblemILP(ntw, n_replicas=configs.n_replicas, l=configs.lmb, verbose=configs.verbose)

        status = problem.solve()
        solution = None
        o1, o2 = None, None
        while status == 'Optimal':
            solution = problem.getSolutionString()
            o1, o2 = problem.getObjective(0), problem.getObjective(1)
            status = problem.solve()

        if solution is None:
            print("ERROR: No solution could be found.")
            return None

        if configs.print:
            print(solution)

        return "{} {}".format(o1, o2)
        #return "{} {}".format(problem.getObjectiveNormalized(0), problem.getObjectiveNormalized(1))
        #return "{} {}".format(problem.getSingleModeObjective(), '0')

    else:
        my_sampling = sampling_v[configs.sampling_version](
                n_replicas = configs.n_replicas
            )
        
        my_crossover = crossover_v[configs.crossover_version](
                n_replicas = configs.n_replicas
            )

        my_repair = MyRepair(
                n_replicas = configs.n_replicas
            )

        my_mutation = mutation_v[configs.mutation_version](
                p_move     = configs.mutation_prob_move,
                p_change   = configs.mutation_prob_change,
                p_binomial = configs.mutation_prob_binomial,
                n_replicas = configs.n_replicas
            )

        my_callback = MyCallback(
                save_history=configs.save_history
            )

        my_duplicate_elimination = MyDuplicateElimination()

        if configs.single_mode and configs.algorithm in ('NSGA2', 'RNSGA2'):
            print("ERROR: Chosen algorithm does not support single-mode execution.")
            return None

        problem = Problem01v3(ntw, o_list, multimode=not configs.single_mode, l=configs.lmb)
        termination = get_termination(configs.termination_type, configs.n_gen)

        if configs.algorithm == 'NSGA2':
            algorithm = NSGA2(
                    pop_size = configs.pop_size,
                    sampling = my_sampling,
                    crossover = my_crossover,
                    mutation = my_mutation,
                    repair = my_repair,
                    callback = my_callback,
                    eliminate_duplicates = my_duplicate_elimination
                )

        elif configs.algorithm == 'SMSEMOA':
            algorithm = SMSEMOA(
                    sampling=my_sampling,
                    crossover=my_crossover,
                    mutation=my_mutation,
                    repair=my_repair,
                    callback = my_callback,
                    eliminate_duplicates=my_duplicate_elimination
                )

        elif configs.algorithm in ('RNSGA2', 'RNSGA3'):
            # Algorithms that use reference points
            ref_points = np.array(configs.ref_points)

            if configs.algorithm == 'RNSGA2':
                algorithm = RNSGA2(
                        pop_size = configs.pop_size,
                        ref_points = ref_points,
                        sampling = my_sampling,
                        crossover = my_crossover,
                        mutation = my_mutation,
                        repair = my_repair,
                        callback = my_callback,
                        eliminate_duplicates = my_duplicate_elimination
                    )
            else:
                # ERROR: The number of points (n_points = 4) can not be created uniformly.                       # Either choose n_points = 3 (n_partitions = 1) or n_points = 6 (n_partitions = 2)
                # A lo mejor la población tiene que ser divisible entre el número de objetivos
                algorithm = RNSGA3(
                        pop_per_ref_point = configs.pop_size // ref_points.size,
                        ref_points = ref_points,
                        sampling = my_sampling,
                        crossover = my_crossover,
                        mutation = my_mutation,
                        repair = my_repair,
                        callback = my_callback,
                        eliminate_duplicates = my_duplicate_elimination
                    )
                

        else:
            # Algorithms that use reference directions
            ref_dirs = get_reference_directions(
                    'das-dennis',
                    len(configs.objectives),
                    n_partitions=configs.n_partitions
                )

            if configs.algorithm in ('CTAEA', 'RVEA'):
                # Algorithms without population size
                algorithm = algdict[configs.algorithm] (
                        ref_dirs = ref_dirs,
                        sampling = my_sampling,
                        crossover = my_crossover,
                        mutation = my_mutation,
                        repair = my_repair,
                        callback = my_callback,
                        eliminate_duplicates = my_duplicate_elimination
                    )
            else:
                # Algorithms with population size
                algorithm = algdict[configs.algorithm] (
                        pop_size = configs.pop_size,
                        ref_dirs = ref_dirs,
                        sampling = my_sampling,
                        crossover = my_crossover,
                        mutation = my_mutation,
                        repair = my_repair,
                        callback = my_callback,
                        eliminate_duplicates = my_duplicate_elimination
                    )

        res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=configs.seed,
            verbose=configs.verbose,
            save_tracker = True
        )

        # Infeasible
        if res.CV is None:
            print("ERROR: No solution could be found.")
            return None


        file_output_tracker = "%s_tracker.csv"%".".join(configs.output.name.split(".")[:-1])
        file_tracker = open(file_output_tracker, "wb+") #Fix the name in the config. parametgers
        coreDF = pd.DataFrame(columns=algorithm.record_tracker_columns)
        coreDF.to_csv(file_tracker, index=False, encoding='utf-8')
        
        for igen in range(len(res.history_pop)):
            f_idx_objectives = list()

            for o in res.history_pop[igen]["opt"]:
                 f_idx_objectives.append(o.data["idx"])
                
            data = []
            for individual in res.history_pop[igen]["pop"]:
                ind_data = individual.data 
                if ind_data["idx"] in f_idx_objectives:
                      ind_data["isF"] = True
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


        if configs.save_history:
            return res.algorithm.callback.string_history
        else:
            return res.algorithm.callback.string_solution