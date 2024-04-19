import numpy as np

# Random
SEED = 722
RANDOM_KEYS = [ 
        'graph_nodes',
        'graph_weights',
        'node_memory',
        'node_cpu',
        'task_memory',
        'tu_assignment',
        'node_power',
        'task_cpu_usage',
        'user_data'
    ]

# Network generation
E = 2
MIN_WEIGHT = 5.
MAX_WEIGHT = 20.
EDGE_MIN_LATENCY = 5.
EDGE_MAX_LATENCY = 20.
EDGE_MIN_BANDWIDTH = 500.
EDGE_MAX_BANDWIDTH = 8000.

N_NODES = 10
NODE_MEMORY_CHOICE = [512, 1024, 2048, 4096]
NODE_MEMORY_PARETO_SHAPE = 1.16
NODE_MAX_TASKS_CHOICE = list(range(100,101))
NODE_N_CPUS_CHOICE = [1, 2, 4, 6, 8]
NODE_N_CPUS_PARETO_SHAPE = 1.16

# Power consumption
NODE_MIN_PW_CHOICE = np.arange(3.5, 20.5, 0.5)
NODE_CPU_PW_R_CHOICE = np.arange(0.5, 1.75, 0.25)
NODE_CPU_PW_MODEL_LIST = [
        (np.array([0.0, 0.2, 0.8, 1.0]), np.array([0.0, 0.3, 0.7, 1.0])),
        (np.array([0.0, 0.3, 0.7, 1.0]), np.array([0.0, 0.2, 0.8, 1.0])),
        (np.array([0.0, 0.2, 0.6, 1.0]), np.array([0.0, 0.4, 0.8, 1.0])),
        (np.array([0.0, 0.4, 0.8, 1.0]), np.array([0.0, 0.2, 0.6, 1.0]))
    ] # (x[i], y[i]) segment points for piecewise linear functions

NODE_MEM_PW_R_CHOICE = np.arange(0.2, 1.1, 0.1)

N_TASKS = 20
TASK_MEMORY_PARETO_SHAPE = 0.8
TASK_MIN_MEMORY = 30
TASK_MAX_MEMORY = 1500
TASK_CPU_USAGE_PARETO_SHAPE = 0.7
TASK_MIN_CPU_USAGE = 0.01
TASK_MAX_CPU_USAGE = 1.0

N_USERS = 3
P = 0.3

USER_REQUEST_SIZE = 50
USER_MIN_PPS = 0.2
USER_MAX_PPS = 5.0

GROUP_SIZE = 5
POPULARITY = 0.5
SPREADNESS = 0.5

OUTPUT_FILE = 'graph01.gefx'

# Pymoo optimization problem solving
POP_SIZE = 100
ALGORITHM = 'NSGA2'
N_GEN = 100
TERMINATION_TYPE = 'n_gen'

N_REPLICAS = 1
MUTATION_PROB_MOVE     = 0.1
MUTATION_PROB_CHANGE   = 0.1
MUTATION_PROB_BINOMIAL = 0.1

N_PARTITIONS = 16
REF_POINTS = '[[18., 6.], [15., 8.], [21., 5.]]'

LAMBDA = 0.5 # used for converting bimode to single-mode

OBJ_LIST = [
        'distance',
        'nodes',
        'hops',
        'occupation',
        'occ_variance',
        'pw_consumption',
        'ntw_utilization'
    ]

OBJ_DESCRIPTION = [
        'Mean latency between users/services',
        'Occupied nodes',
        'Mean hops to service',
        'Mean node occupation ratio',
        'Node occupation ratio variance',
        'Power consumption',
        'Network utilization'
    ]

ALGORITHMS = [
        'NSGA2',
        'RNSGA2',
        'NSGA3',
        'UNSGA3',
        'RNSGA3',
        'AGEMOEA',
        'CTAEA',
        'SMSEMOA',
        'RVEA',
        'ILP'
    ]

SAMPLING_VERSION = 0
CROSSOVER_VERSION = 2
MUTATION_VERSION = 1



