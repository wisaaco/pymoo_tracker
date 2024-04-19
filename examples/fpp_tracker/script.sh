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

### Looping operator versions
#for SAMPLING_VERSION in ${SAMPLING_VERSION_LIST[*]}; do
#for CROSSOVER_VERSION in ${CROSSOVER_VERSION_LIST[*]}; do
for MUTATION_VERSION in ${MUTATION_VERSION_LIST[*]}; do
	echo $MUTATION_VERSION
done
#done
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

#TODO: https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html

# generate

for ALGORITHM in ${ALGORITHMS[*]}; do
    echo "$ALGORITHM"
    for MUTATION_VERSION in ${MUTATION_VERSION_LIST[*]}; do
        echo " Mutation $MUTATION_VERSION"

        for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
            echo "   Seed  $SEED2"
            solve &
            pids[${SEED2}]=$!
        done

        for pid in ${pids[*]}; do
            wait $pid
        done

        arrange
    
    done

done

# arrange_all
#solution_to_ref_points 0.9
