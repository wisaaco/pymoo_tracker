#!/bin/bash

SEED=722

POP_SIZE=500
N_GEN=200
MUTATION_PROB=0.1

NODES=40
TASKS=60
USERS=20

PREFIX1="data/solutions/P$POP_SIZE-G$N_GEN/M$MUTATION_PROB/$NODES-$TASKS-$USERS"
PREFIX2="data/analysis/P$POP_SIZE-G$N_GEN/M$MUTATION_PROB/$NODES-$TASKS-$USERS"

mkdir -p "$PREFIX2"

for SEED2 in {1..4}; do
python3 main.py --seed $SEED analyze \
	-i \
		"$PREFIX1/NSGA2_"$SEED"-"$SEED2"_"$NODES"-"$TASKS"-"$USERS"_"$POP_SIZE"-"$N_GEN \
		"$PREFIX1/NSGA3_"$SEED"-"$SEED2"_"$NODES"-"$TASKS"-"$USERS"_"$POP_SIZE"-"$N_GEN \
		"$PREFIX1/UNSGA3_"$SEED"-"$SEED2"_"$NODES"-"$TASKS"-"$USERS"_"$POP_SIZE"-"$N_GEN \
		"$PREFIX1/CTAEA_"$SEED"-"$SEED2"_"$NODES"-"$TASKS"-"$USERS"_"$POP_SIZE"-"$N_GEN \
		"$PREFIX1/SMSEMOA_"$SEED"-"$SEED2"_"$NODES"-"$TASKS"-"$USERS"_"$POP_SIZE"-"$N_GEN \
		"$PREFIX1/RVEA_"$SEED"-"$SEED2"_"$NODES"-"$TASKS"-"$USERS"_"$POP_SIZE"-"$N_GEN \
	--alg_name \
		"NSGA2" \
		"NSGA3" \
		"UNSGA3" \
		"CTAEA" \
		"SMSEMOA" \
		"RVEA" \
	--ref_points \
		$(cat "$PREFIX1/ref_points/rp_ILP_"$SEED"-1") \
	--network \
		"data/network/ntw_"$SEED"_$NODES-$TASKS-$USERS" \
	--output \
		"$PREFIX2/table_"$SEED"-"$SEED2"_$NODES-$TASKS-$USERS"
done

output="$PREFIX2/table_"$SEED"-M_$NODES-$TASKS-$USERS"
rm -f $output

head -n1 "$PREFIX2/table_"$SEED"-1_$NODES-$TASKS-$USERS" >> $output

for i in {2..7}; do

	printf '%-12s' $(cat "$PREFIX2/table_"$SEED"-1_$NODES-$TASKS-$USERS" | sed -n "${i}p" | awk '{print $1}') >> $output

	for j in {2..6}; do

		result=0.0

		for SEED2 in {1..4}; do

			cell=$(cat "$PREFIX2/table_"$SEED"-"$SEED2"_$NODES-$TASKS-$USERS" | sed -n "${i}p" | awk '{print $'$j'}')
			result=$(echo "scale=10; $result + $cell" | bc)

		done

		result=$(echo "scale=10; $result / 4" | bc)
		LC_ALL=C printf '%012.10f   ' $result >> $output

	done

	echo >> $output
done
