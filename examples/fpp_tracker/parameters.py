import argparse
import sys
import ast
from default import *

class Range(object):
    """Used for ranges of values that are not integers"""
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__ (self, other):
        return self.start <= other <= self.end
    def __repr__(self):
        return "[{}, {}]".format(self.start, self.end)

def type_point_list(arg):
    return ast.literal_eval('%s' % arg)

parser = argparse.ArgumentParser(description="Arguments")
subparsers = parser.add_subparsers(dest='command')

# Random
parser.add_argument('--seed', type=int, default=SEED, help='Seed used for random generation')

# Network generation
parser_generate = subparsers.add_parser('generate', help='Generate the network')

parser_generate.add_argument('-e', '--edges', type=int, default=E, help='Number of edges to attach from a new node to existing nodes. Conflicts with --probability.')
parser_generate.add_argument('--min_weight', type=float, default=MIN_WEIGHT, help='Minimum weight that an edge can have')
parser_generate.add_argument('--max_weight', type=float, default=MAX_WEIGHT, help='Maximum weight that an edge can have')
parser_generate.add_argument('--edge_min_bandwidth', type=float, default=EDGE_MAX_BANDWIDTH, help='Minimum bandwidth that an edge can have')
parser_generate.add_argument('--edge_max_bandwidth', type=float, default=EDGE_MIN_BANDWIDTH, help='Minimum bandwidth that an edge can have')

parser_generate.add_argument('--n_nodes', type=int, default=N_NODES, help='Number of server nodes in the network')

parser_generate.add_argument('--n_tasks', type=int, default=N_TASKS, help='Number of tasks to be executed in the network')
parser_generate.add_argument('--task_min_memory', type=float, default=TASK_MIN_MEMORY, help='Minimum memory that a task requires to execute')
parser_generate.add_argument('--task_max_memory', type=float, default=TASK_MAX_MEMORY, help='Maximum memory that a task requires to execute')
parser_generate.add_argument('--task_min_cpu_usage', type=float, default=TASK_MIN_CPU_USAGE, help='Minimum CPU usage that a task requires to execute')
parser_generate.add_argument('--task_max_cpu_usage', type=float, default=TASK_MAX_CPU_USAGE, help='Maximum CPU usage that a task requires to execute')

parser_generate.add_argument('--n_users', type=int, default=N_USERS, help='Number of users in the network')
parser_generate.add_argument('--user_request_size', type=int, default=USER_REQUEST_SIZE, help='Size of user request to be sended/received')
parser_generate.add_argument('--user_min_pps', type=int, default=USER_MIN_PPS, help='Minimum petitions per second')
parser_generate.add_argument('--user_max_pps', type=int, default=USER_MAX_PPS, help='Maximum petitions per second')

parser_generate.add_argument('-p', '--probability', type=float, choices=[Range(0.0, 1.0)], default=P, help='Probability of a user requesting a service. This probability will only take effect if communities option is not enabled.')

parser_generate.add_argument('--communities', action='store_true', help='Distribute task/user assignments based on communities.')
parser_generate.add_argument('--group_size', type=int, default=GROUP_SIZE, help='Size of node grouping. This will determine the number of partitions, so that number of groups scales with the problem.')
parser_generate.add_argument('--popularity', type=float, choices=[Range(0.0, 1.0)], default=POPULARITY, help='Probability of a task being requested by users within the same community.')
parser_generate.add_argument('--spreadness', type=float, choices=[Range(0.0, 1.0)], default=SPREADNESS, help='Ratio at which the probability of a task being requested decreases for users from different communities.')

parser_generate.add_argument('--paint', action='store_true', help='Paint the graph with Matplotlib')
parser_generate.add_argument('--paint_seed', type=int, default=1, help='Seed used for graph painting')
parser_generate.add_argument('--print', action='store_true', help='Print on console useful information about the generated network')
parser_generate.add_argument('-o', '--output', type=argparse.FileType('wb'), help='Output file path used for storing the network data')

# Network modification (after it was generated)

# Ideas para opciones TODO:
# - Analizar memoria de los nodos
# - Analizar memoria de las tareas
# - Pasar una matriz de asignación de tareas/nodos, si no viene ya incorporada:
#     - Analizar memoria ocupada
#     - Analizar distancia entre usuarios
# - Permitir obtener undm, tuam y tudm
# - Analizar grado de centralidad

# Pymoo optimization problem solving
parser_solve = subparsers.add_parser('solve', help='Solve optimization problem')

parser_solve.add_argument('-i', '--input', type=argparse.FileType('rb'), help='Input file path used for generating the network')
parser_solve.add_argument('--objectives', type=str, nargs='+', default=['distance','nodes'], help='Objectives to use in the algorithm')

parser_solve.add_argument('--pop_size', type=int, default=POP_SIZE, help='Population size')
parser_solve.add_argument('--algorithm', type=str, choices=ALGORITHMS, default=ALGORITHM, help='Name of the algorithm to be used to solve the problem')

parser_solve.add_argument('--termination_type', type=str, default=TERMINATION_TYPE, help='Termination type for the algorithm')
parser_solve.add_argument('--n_gen', type=int, default=N_GEN, help='Number of generations as termination parameter')

parser_solve.add_argument('--n_replicas', type=int, default=N_REPLICAS, help='Max number of replicas for service assignment')
parser_solve.add_argument('--mutation_prob_move', type=float, choices=[Range(0.0, 1.0)], default=MUTATION_PROB_MOVE, help='Probability of mutation of type move in task assignment to nodes')
parser_solve.add_argument('--mutation_prob_change', type=float, choices=[Range(0.0, 1.0)], default=MUTATION_PROB_CHANGE, help='Probability of mutation of type change in task assignment to nodes')
parser_solve.add_argument('--mutation_prob_binomial', type=float, choices=[Range(0.0, 1.0)], default=MUTATION_PROB_BINOMIAL, help='Binomial probability which will give the amount of mutations for each individual depending on infrastructure size. Only for mutation versions that support it (currently, only Mutation_v4)')

parser_solve.add_argument('-v', '--verbose', action='store_true')
parser_solve.add_argument('--print', action='store_true', help='Print on console useful information about the generated network')

parser_solve.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file path used for storing the solution data')
parser_solve.add_argument('--save_history', action='store_true', help='Will save history to retrieve the evolution of the solutions')

parser_solve.add_argument('--single_mode', action='store_true', help='Will force the algorithm to use single mode optimization')
parser_solve.add_argument('-l', '--lmb', type=float, choices=[Range(0.0, 1.0)], default=LAMBDA, help='Parameter used for weighted average when converting bimode to single-mode.')

# Helper for ILP constraint handling
parser_solve.add_argument('--o1_max', type=float, default=None, help='Will use this value as a max value constraint for O1')
parser_solve.add_argument('--o2_min', type=float, default=None, help='Will use this value as a min value constraint for O2')

# Specific parameter for algorithms that need reference directions
parser_solve.add_argument('--n_partitions', type=int, default=N_PARTITIONS, help='Specific parameter for algorithm NSGA3. Set the number of partitions for reference directions.')

# Specific parameter for algorithms that need reference point
parser_solve.add_argument('--ref_points', type=type_point_list, default=REF_POINTS, help='Specific parameter for algorithms that requiere reference points')

# Problem operators
parser_solve.add_argument('--sampling_version', type=int, default=SAMPLING_VERSION, help='Sampling version number to be used')
parser_solve.add_argument('--crossover_version', type=int, default=CROSSOVER_VERSION, help='Crossover version number to be used')
parser_solve.add_argument('--repair_version', type=int, default=0, help='Repair version number to be used')
parser_solve.add_argument('--mutation_version', type=int, default=MUTATION_VERSION, help='Mutation version number to be used')



# Solution arrange
parser_arrange = subparsers.add_parser('arrange', help='Arrange solution files')

parser_arrange.add_argument('-i', '--input', nargs='+', type=argparse.FileType('r'), help='List of input file paths used for arranging the solutions')
parser_arrange.add_argument('--print', action='store_true', help='Print on console the resulting arrangement')
parser_arrange.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file path used for storing the arranged solution data')

parser_arrange.add_argument('--n_objectives', type=int, default=2, help='Number of objectives within the solution file.')



# Reference point calculator
parser_get_ref_points = subparsers.add_parser('get_ref_points', help='Get reference points')

parser_get_ref_points.add_argument('-i', '--input', type=argparse.FileType('r'), help='Reference point file')
parser_get_ref_points.add_argument('--ntw_file', type=argparse.FileType('rb'), help='Network file used to find normalized values')
parser_get_ref_points.add_argument('--objectives', type=str, nargs='+', default=['distance','nodes'], help='Objectives to use in the algorithm')

parser_get_ref_points.add_argument('--lazy', action='store_true', help='Lazy way of getting reference points. It gets a solution file, normalizes the points, scales by a float between 0 and 1 and denormalizes them.')
parser_get_ref_points.add_argument('-l', '--lmb', type=float, choices=[Range(0.0, 1.0)], default=LAMBDA, help='If used with lazy option, will act as a scalar constant.')

parser_get_ref_points.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file path used for storing the reference point formated solution')



# Solution analysis
parser_analyze = subparsers.add_parser('analyze', help='Analyze the generated solutions using several tools')

parser_analyze.add_argument('-i', '--input', nargs='+', type=argparse.FileType('r'), help='List of input file paths used for analyzing the solutions')
parser_analyze.add_argument('--ref_points', type=type_point_list, default=REF_POINTS, help='Specific parameter for algorithms that requiere reference points')

parser_analyze.add_argument('--alg_names', nargs='+', type=str, help='Algorithm names to use on the table')
parser_analyze.add_argument('--gen_step', type=int, default=0, help='Generation step used for printing the evolution of each indicator')

parser_analyze.add_argument('--network', type=argparse.FileType('rb'), help='Input file path used for retrieving the network, useful for calculating several needed parameters, like O1 and O2 max values')

parser_analyze.add_argument('--n_objectives', type=int, default=2, help='Number of objectives within the solution file.')
parser_analyze.add_argument('--objectives', type=str, nargs='+', default=['distance','nodes'], help='Objectives to use in the algorithm')

parser_analyze.add_argument('--print', action='store_true', help='Print on console the table')
parser_analyze.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file path used for saving generated table')


# Solution ploting
parser_plot = subparsers.add_parser('plot', help='Plot the resulting data from the solution')

parser_plot.add_argument('-i', '--input', nargs='+', type=argparse.FileType('r'), help='List of input file paths used for plotting the solutions')
parser_plot.add_argument('--ref_points', type=type_point_list, default=None, help='Specific parameter for plotting reference points')
parser_plot.add_argument('--ref_points_legend', type=str, help='Names for the reference points to be plotted and added to the legend')

parser_plot.add_argument('--n_objectives', type=int, default=2, help='Number of objectives within the solution file.')
parser_plot.add_argument('--objectives', type=str, nargs='+', default=['distance','nodes'], help='Objectives to be plotted')

parser_plot.add_argument('--history', action='store_true', help='Plot a single solution including the history representing the evolution with the form of a scatter plot')
parser_plot.add_argument('--trim_gen', action='store_true', help='Plots generations until convergence')

parser_plot.add_argument('--comparison', action='store_true', help='Plot multiple solutions comparing them in the same graph with the form of a scatter plot')
parser_plot.add_argument('--legend', nargs='*', type=str, help='List of names for the different inputs to be plotted and added to the legend')

parser_plot.add_argument('--title', type=str, default='', help='Title of the plot')
parser_plot.add_argument('--x_label', type=str, default=None, help='Label for X axis')
parser_plot.add_argument('--y_label', type=str, default=None, help='Label for Y axis')
parser_plot.add_argument('--z_label', type=str, default=None, help='Label for Z axis')

parser_plot.add_argument('-o', '--output', type=argparse.FileType('wb'), help='Output file path used for saving plot result')

# TODO: pasándole un network por parámetro, permitir imprimir memoria/nodo para ver frente pareto
#   - Frente pareto memoria de los nodos
#   - Frente pareto memoria de las tareas


# CONFIG GENERATOR
# ==============================================================================
configs = parser.parse_args()
configs.node_memory_choice = NODE_MEMORY_CHOICE
configs.node_max_tasks_choice = NODE_MAX_TASKS_CHOICE
configs.node_n_cpus_choice = NODE_N_CPUS_CHOICE
configs.node_min_pw_choice = NODE_MIN_PW_CHOICE
configs.node_cpu_pw_r_choice = NODE_CPU_PW_R_CHOICE
configs.node_mem_pw_r_choice = NODE_CPU_PW_R_CHOICE
configs.node_memory_pareto_shape = NODE_MEMORY_PARETO_SHAPE
configs.node_n_cpus_pareto_shape = NODE_N_CPUS_PARETO_SHAPE
configs.task_memory_pareto_shape = TASK_MEMORY_PARETO_SHAPE
configs.task_cpu_usage_pareto_shape = TASK_CPU_USAGE_PARETO_SHAPE
configs.user_request_size = USER_REQUEST_SIZE

if __name__ == '__main__':
    print(configs)


