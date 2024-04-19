#!/bin/bash

source script_constants.sh
source script_functions.sh

# ==============================================================================
# PROGRAM
# ==============================================================================

### Network
# generate $SEED $NODES $TASKS $USERS $COMMUNITIES

### Problem solving
# solve
# arrange

### Plotting
# plot_comparison
# plot_convergence

### Helpers
# send_telegram_message

### Looping networks
#for NODES in $(seq 20 20 40); do
#for TASKS in $(seq 20 20 $((NODES*2))); do
#for USERS in $(seq 20 20 $NODES); do
#	:
#done
#done
#done

### Looping algorithms
#for ALGORITHM in ${ALGORITHMS[*]}; do
#	:
#done

### Looping seeds + thread handling
#for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
#	:
#	pids[${SEED2}]=$!
#done
#
#for pid in ${pids[*]}; do
#	wait $pid
#done

plot_comparison algorithms

#for MUTATION_PROB_MOVE in ${MUTATION_PROB_MOVE_LIST[*]}; do
#for MUTATION_PROB_CHANGE in ${MUTATION_PROB_CHANGE_LIST[*]}; do
#for MUTATION_PROB_BINOMIAL in ${MUTATION_PROB_BINOMIAL_LIST[*]}; do
#
#for SAMPLING_VERSION in ${SAMPLING_VERSION_LIST[*]}; do
#for CROSSOVER_VERSION in ${CROSSOVER_VERSION_LIST[*]}; do
#for MUTATION_VERSION in ${MUTATION_VERSION_LIST[*]}; do
#
SEED2=2
plot_convergence
#
#done
#done
#done
#
#done
#done
#done



