#!/bin/bash

SEED=722
SEED2=1

NODES=20
TASKS=20
USERS=20

POP_SIZE=500
N_GEN=200
MUTATION_PROB=0.1
ALGORITHM='ILP'

N_POINTS=5
N_PARTITIONS=$((N_POINTS+1))

LAMBDA_LIST='0.1 0.3 0.5 0.7 0.9'
#LAMBDA_LIST=$(python3 -c "for i in range(1,$N_REF_POINTS+1): print(i/($N_REF_POINTS+1))")

PREFIX="data/solutions/P$POP_SIZE-G$N_GEN/M$MUTATION_PROB"

for NODES in $(seq 20 20 40); do
for TASKS in $(seq 20 20 $((NODES*2))); do
for USERS in $(seq 10 20 $NODES); do
	echo $NODES $TASKS $USERS

	mkdir -p "$PREFIX/$NODES-$TASKS-$USERS/ref_points"
	mkdir -p "$PREFIX/$NODES-$TASKS-$USERS/tmp"
	mkdir -p "$PREFIX/$NODES-$TASKS-$USERS/time"

	i=1
	for l in $LAMBDA_LIST; do
		echo "  $NODES $TASKS $USERS: $i/$N_POINTS: lambda=$l"

		# Async call
		{ time python3 main.py --seed $SEED2 solve \
			-i "data/network/ntw_"$SEED"_"$NODES"-"$TASKS"-"$USERS \
			--pop_size $POP_SIZE --n_gen $N_GEN --mutation_prob $MUTATION_PROB \
			--algorithm $ALGORITHM --n_partitions $N_PARTITIONS --single_mode --lmb $l \
			--output "$PREFIX/$NODES-$TASKS-$USERS/tmp/ref_$i"
		} 2> "$PREFIX/$NODES-$TASKS-$USERS/time/time_$i" &
		pids[${i}]=$!
		i=$((i+1))
	done

	i=1
	for pid in ${pids[*]}; do
		wait $pid
		if [ $? -eq 0 ]; then
			array[${i}]="$(cat "$PREFIX/$NODES-$TASKS-$USERS/tmp/ref_$i" | grep . | awk '{print "[" $1 "," $2 "]"}')"
		fi
		i=$((i+1))
	done

	echo "[${array[*]}]" | tr -s '[:blank:]' ',' > "$PREFIX/$NODES-$TASKS-$USERS/ref_points/rp_"$ALGORITHM"_"$SEED"-"$SEED2

done
done
done

