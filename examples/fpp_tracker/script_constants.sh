#!/bin/bash

# ==============================================================================
# CONSTANTS
# ==============================================================================
SEED=722

NODES=40
TASKS=40
USERS=20

COMMUNITIES=true

N_REPLICAS=$NODES
MUTATION_PROB_MOVE=0.2
MUTATION_PROB_MOVE_LIST=(0.1 0.2)
MUTATION_PROB_CHANGE=0.1
MUTATION_PROB_CHANGE_LIST=(0.1 0.2)
MUTATION_PROB_BINOMIAL=0.1
MUTATION_PROB_BINOMIAL_LIST=(0.025 0.05)

POP_SIZE=200
N_GEN=200


ALGORITHM='NSGA2'
# ALGORITHMS=('NSGA2' 'NSGA3' 'UNSGA3' 'CTAEA' 'SMSEMOA') # 'RVEA')
# ALGORITHMS=('CTAEA')
ALGORITHMS=('SMSEMOA')
# Available: 'NSGA2' 'NSGA3' 'UNSGA3' 'CTAEA' 'SMSEMOA' 'RVEA' 'RNSGA2' 'RNSGA3'
# Not implemented: 'AGEMOEA'

SAMPLING_VERSION=0
SAMPLING_VERSION_LIST=(0)
CROSSOVER_VERSION=2
CROSSOVER_VERSION_LIST=(2)
MUTATION_VERSION=1
# MUTATION_VERSION_LIST=(1 2 3 4)
MUTATION_VERSION_LIST=(0 1 2 3)

N_PARTITIONS=16

REF_POINTS_ALGORITHM='NSGA2'
LAMBDA_LIST=($(LANG=en_US seq 0.1 0.2 1))

OBJECTIVES=('distance' 'occupation' 'ntw_utilization')
N_OBJECTIVES=3

N_EXECUTIONS=1
SEED2=A

NTW_PREFIX="data/networks"
SOL_PREFIX="data/solutions" 
PREFIX="data/solutions/P$POP_SIZE-G$N_GEN/MM$MUTATION_PROB_MOVE-MC$MUTATION_PROB_CHANGE/new_crossover"
PREFIX2="data/solutions/P$POP_SIZE-G$N_GEN/MM$MUTATION_PROB_MOVE-MC$MUTATION_PROB_CHANGE/communities"

