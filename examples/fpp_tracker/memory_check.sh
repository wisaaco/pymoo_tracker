#!/bin/bash

SEED=722

for MIN_MEM in $(seq 300 -5 30); do

	b=0

	for NODES in $(seq 20 20 40); do
	for TASKS in $(seq 20 20 $((NODES*2))); do
	for USERS in $(seq 20 20 $NODES); do

		echo "    $MIN_MEM: $NODES-$TASKS-$USERS"

		python3 main.py --seed $SEED generate \
			--n_nodes $NODES --n_tasks $TASKS --n_users $USERS \
			--task_min_memory $MIN_MEM
		retVal=$?

		if [[ $retVal -ne 0 ]]; then
			b=1
			break
		fi

	done
	if [[ b -eq 1 ]]; then break; fi
	done
	if [[ b -eq 1 ]]; then break; fi
	done

	if [[ b -eq 0 ]]; then
		echo "--> $MIN_MEM: success with all configurations"
		break
	fi

done

# Success for MIN_MEM < 155
# but ILP might return unfeasible solutions

