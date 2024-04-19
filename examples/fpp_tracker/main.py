"""
export PYTHONPATH=${PYTHONPATH}:~/Projects/pymoo_tracker
python -u "/Users/isaac/Projects/pymoo_tracker/examples/lineage_tracking/eval_CTPproblems.py"
"""

from network import Network
from parameters import configs
from solve_problem import solve
from plot import plot_convergence, plot_scatter_legend
from arrange import get_pareto_front_from_files
from analyze import get_table
from file_utils import solutions_to_string
from get_ref_points import solutions_to_ref_points, lazy_ref_points

import random
import numpy as np
import sys

def paint_graph(ntw, seed=1):
    try:
        while True:
            print("Showing graph painted with seed = {}".format(seed))
            ntw.displayGraph(seed)
            seed += 1
    except KeyboardInterrupt:
        pass

# MAIN
# ==============================================================================

# print(configs)
# print(configs.output.name)
# exit()


if False:
    ##
    # DEBUG
    configs.seed = 1

    if configs.command != "solve":
        configs.input = open("/Users/isaac/Projects/pymoo_tracker/examples/fpp_tracker/data/networks/ntw_722_040-040-020_C","r")
        # configs.output=open("/Users/isaac/Projects/pymoo_tracker/examples/fpp_tracker/data/solutions/ntw_722_040-040-020_C/obj_distance-occupation-ntw_utilization/Replicas040/Genetics/CTAEA_1_010-010_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt","w")
    else:
        configs.command = "solve"
        configs.input = open("/Users/isaac/Projects/pymoo_tracker/examples/fpp_tracker/data/networks/ntw_722_040-040-020_C","rb")
        # configs.output=open("/Users/isaac/Projects/pymoo_tracker/examples/fpp_tracker/data/solutions/ntw_722_040-040-020_C/obj_distance-occupation-ntw_utilization/Replicas040/Genetics/CTAEA_1_010-010_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt","w")
    print(type(configs.input))
    configs.objectives = ["distance" ,"occupation" ,"ntw_utilization"]
    configs.pop_size = 10
    configs.n_gen = 10
    configs.n_replicas = 40
    configs.n_partitions = 16
    configs.crossover_version = 2
    configs.sampling_version = 0
    configs.mutation_version = 1
    configs.mutation_prob_move = 0.2
    configs.mutation_prob_change = 0.1
    configs.mutation_prob_binomial = 0.1
    configs.save_history = True
    configs.algorithm="CTAEA"

##

if __name__ == "__main__":

    random.seed(configs.seed)

    if configs.command == 'generate':
        print("GENERATE")
        # Generate the network
        ntw = Network(configs)

        if configs.print:
            print(ntw.getTotalNodeMemory())
            print(ntw.getTotalTaskMemory())
            print(ntw.memory)
            print(ntw.getMinimumNNodesNeeded())

        if configs.paint:
            paint_graph(ntw, configs.paint_seed)

        if configs.output:
            import pickle
            pickle.dump(ntw, configs.output)
            configs.output.close()

        if not ntw.checkMemoryRequirements():
            print('WARNING: Memory requirements could not be satisfied.')
            print('- Total node memory: {}'.format(
                    ntw.getTotalNodeMemory()))
            print('- Total task memory: {}'.format(
                    ntw.getTotalTaskMemory()))
            print('Change memory limit or amount of tasks and try again')
            sys.exit(1)

    elif configs.command == 'modify':
        import pickle
        ntw = pickle.load(configs.input)

    elif configs.command == 'solve':
        # Solve a problem using a network and an optimization algorithm
        import pickle
        ntw = pickle.load(configs.input)

        solution = solve(ntw, configs)
        if solution is None:
            if configs.output:
                configs.output.close()
            sys.exit(1)

        if configs.print:
            print(solution)

        if configs.output:
            configs.output.write(solution)
            configs.output.close()

    elif configs.command == 'arrange':
        # Arrange files of solutions and prepare them for ploting
        array = get_pareto_front_from_files(configs)

        s = solutions_to_string(array)

        if configs.print:
            print(s)

        if configs.output:
            configs.output.write(s)
            configs.output.close()

    elif configs.command == 'get_ref_points':
        if configs.lazy:
            s = lazy_ref_points(configs).tolist().__repr__()
        else:
            s = solutions_to_ref_points(configs)

        if configs.output:
            configs.output.write(s)
            configs.output.close()
             
    elif configs.command == 'analyze':
        # Analyze the solutions
        table = get_table(configs)

        if configs.print:
            print(table)

        if configs.output:
            configs.output.write(table)
            configs.output.close()
    
    elif configs.command == 'plot':
        # Plot files
        if configs.comparison:
            plot_scatter_legend(configs)
        if configs.history:
            plot_convergence(configs)



