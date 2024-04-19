#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:~/Projects/pymoo_tracker

# ==============================================================================
# FUNCTIONS
# ==============================================================================
zeropad_left() {
	printf "%0$2d\n" $1
}

get_network_filename() {
	local SEED=$(zeropad_left $1 3)
	local NODES=$(zeropad_left $2 3)
	local TASKS=$(zeropad_left $3 3)
	local USERS=$(zeropad_left $4 3)
	local COMMUNITIES=$5

	if [ $COMMUNITIES = true ]; then
		local SUFFIX="C"
	else
		local SUFFIX="H"
	fi

	echo "ntw_"$SEED"_"$NODES"-"$TASKS"-"$USERS"_"$SUFFIX
}

generate() {
	mkdir -p "$NTW_PREFIX"

	if [ $COMMUNITIES = true ]; then
		local C_OPT=--communities
	else
		local C_OPT=
	fi

	local NTW_FILENAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"

	python3 main.py --seed $SEED generate \
		--n_nodes $NODES --n_tasks $TASKS --n_users $USERS \
		$C_OPT -o "$NTW_PREFIX/$NTW_FILENAME"
}

get_solution_path() {
	local NTW_NAME=$1
	local N_REPLICAS=$(zeropad_left $2 3)
	local ALGORITHM=$3
	shift; shift; shift
	local OBJECTIVES=("$@")

	if [ $ALGORITHM = "ILP" ]; then
		local ALG_TYPE="ILP"
	else
		local ALG_TYPE="Genetics"
	fi
	
	local prev_ifs=$IFS
	IFS='-'; local OBJ_STRING="${OBJECTIVES[*]}"; IFS=$prev_ifs

	echo "$NTW_NAME/obj_$OBJ_STRING/Replicas$N_REPLICAS/$ALG_TYPE"
}

get_solution_filename() {
	local ALGORITHM=$1
	if [ $ALGORITHM = "ILP" ]; then
		echo "ILP.txt"
	else
		local SEED=$2
		local POP_SIZE=$(zeropad_left $3 3)
		local N_GEN=$(zeropad_left $4 3)
		local SV=$5
		local CV=$6
		local MV=$7
		local MM=$8
		local MC=$9
		local MB=${10}
		echo $ALGORITHM"_"$SEED"_"$POP_SIZE"-"$N_GEN"_SV"$SV"-CV"$CV"-MV"$MV"_MM"$MM"-MC"$MC"-MB"$MB".txt"
	fi
}

get_ref_points_path() {
	local NTW_NAME=$1
	local N_REPLICAS=$(zeropad_left $2 3)
	shift; shift
	local OBJECTIVES=("$@")

	local prev_ifs=$IFS
	IFS='-'; local OBJ_STRING="${OBJECTIVES[*]}"; IFS=$prev_ifs

	echo "$NTW_NAME/obj_$OBJ_STRING/Replicas$N_REPLICAS/RefPoints"
}

get_ref_points_filename() {
	local ALGORITHM=$1
	echo "rp_$ALGORITHM.txt"
}

solve() {
	local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	local SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
	local SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	mkdir -p "$SOL_PREFIX/$SOL_PATH" 

	python3 main.py --seed $SEED2 solve \
		-i "$NTW_PREFIX/$NTW_NAME" \
		--objectives ${OBJECTIVES[*]} $REF_POINTS_OPT $REF_POINTS_STRING \
		--pop_size $POP_SIZE --n_gen $N_GEN \
		--n_replicas $N_REPLICAS \
		--n_partitions $N_PARTITIONS \
		--sampling_version $SAMPLING_VERSION \
		--crossover_version $CROSSOVER_VERSION \
		--mutation_version $MUTATION_VERSION \
		--mutation_prob_move $MUTATION_PROB_MOVE \
		--mutation_prob_change $MUTATION_PROB_CHANGE \
		--mutation_prob_binomial $MUTATION_PROB_BINOMIAL \
		--save_history \
		--algorithm $ALGORITHM \
		-o "$SOL_PREFIX/$SOL_PATH/$SOL_NAME"

}

arrange() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	for SEED2 in $(seq 1 1 $N_EXECUTIONS); do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		ALG_FILES[${SEED2}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
	done

	SOL_NAME="$(get_solution_filename $ALGORITHM "A" $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	python3 main.py arrange \
		--n_objectives $N_OBJECTIVES \
		-i ${ALG_FILES[*]} \
		-o "$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
}

arrange_all(){
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	i=1
	for ALG in ${ALGORITHMS[*]}; do
		SOL_NAME="$(get_solution_filename $ALG 'A' $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		ALG_FILES[$i]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done

	SOL_NAME="$(get_solution_filename "ALL" "A" $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	python3 main.py arrange \
		--n_objectives $N_OBJECTIVES \
		-i ${ALG_FILES[*]} \
		-o "$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
}

solution_to_ref_points() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $REF_POINTS_ALGORITHM ${OBJECTIVES[@]})"
	SOL_NAME="$(get_solution_filename $REF_POINTS_ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
	REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	REF_FILENAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"

	mkdir -p "$SOL_PREFIX/$REF_PATH" 

	FACTOR=$1
	if [ -z ${FACTOR} ]; then
		local FACTOR_OPT=()
	else
		local FACTOR_OPT=(--lazy --lmb $FACTOR)
	fi

	python3 main.py --seed $SEED get_ref_points \
		--objectives ${OBJECTIVES[*]} \
		--ntw_file "$NTW_PREFIX/$NTW_NAME" \
		-i "$SOL_PREFIX/$SOL_PATH/$SOL_NAME" ${FACTOR_OPT[*]} \
		-o "$SOL_PREFIX/$REF_PATH/$REF_FILENAME"
}

solve_ilp() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS 'ILP' ${OBJECTIVES[@]})"

	rm -rf "$SOL_PREFIX/$SOL_PATH/*"

	mkdir -p "$SOL_PREFIX/$SOL_PATH/tmp"
	mkdir -p "$SOL_PREFIX/$SOL_PATH/log"

	i=1
	for l in ${LAMBDA_LIST[*]}; do
		# Async call
		{ time python3 main.py --seed $SEED2 solve \
			-i "$NTW_PREFIX/$NTW_NAME" \
			--algorithm "ILP" --n_partitions $N_PARTITIONS --single_mode --lmb $l \
			--n_replicas $N_REPLICAS \
			--verbose \
			--output "$SOL_PREFIX/$SOL_PATH/tmp/ref_$i"
		} &> "$SOL_PREFIX/$SOL_PATH/log/log_$i" &
		pids[${i}]=$!
		i=$((i+1))
	done

	i=1
	for pid in ${pids[*]}; do
		wait $pid
		if [ $? -eq 0 ]; then
			cat "$SOL_PREFIX/$SOL_PATH/tmp/ref_$i" | grep . >> "$SOL_PREFIX/$SOL_PATH/tmp.txt"
		fi
		i=$((i+1))
	done

	sort -f "$SOL_PREFIX/$SOL_PATH/tmp.txt" | uniq > "$SOL_PREFIX/$SOL_PATH/ILP.txt"
	rm "$SOL_PREFIX/$SOL_PATH/tmp.txt" 
}

plot_convergence() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"
	SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"

	echo $SOL_NAME
	
	python3 main.py --seed $SEED plot \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES \
		-i "$SOL_PREFIX/$SOL_PATH/$SOL_NAME" \
		--history \
		--title "Objective space - Convergence ($ALGORITHM) - $NODES:$TASKS:$USERS" \
		--trim_gen
}

get_algorithm_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	ALG_FILES=()
	i=0
	for ALG in ${ALGORITHMS[*]}; do
		SOL_NAME="$(get_solution_filename $ALG $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		ALG_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME" 
		i=$((i+1))
	done
	echo ${ALG_FILES[*]}
}

get_operator_version_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	OP_FILES=()
	i=0
	for SV in ${SAMPLING_VERSION_LIST[*]}; do
	for CV in ${CROSSOVER_VERSION_LIST[*]}; do
	for MV in ${MUTATION_VERSION_LIST[*]}; do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SV $CV $MV $MUTATION_PROB_MOVE $MUTATION_PROB_CHANGE $MUTATION_PROB_BINOMIAL)"
		OP_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done
	done
	done

	echo ${OP_FILES[*]}
}

get_operator_version_legend() {
	OP_LEGEND=()
	i=0
	for SV in ${SAMPLING_VERSION_LIST[*]}; do
	for CV in ${CROSSOVER_VERSION_LIST[*]}; do
	for MV in ${MUTATION_VERSION_LIST[*]}; do
		OP_LEGEND[${i}]="SV$SV:CV$CV:MV$MV"
		i=$((i+1))
	done
	done
	done

	echo ${OP_LEGEND[*]}
}

get_mutation_files() {
	NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	SOL_PATH="$(get_solution_path $NTW_NAME $N_REPLICAS $ALGORITHM ${OBJECTIVES[@]})"

	MUT_FILES=()
	i=0
	for MM in ${MUTATION_PROB_MOVE_LIST[*]}; do
	for MC in ${MUTATION_PROB_CHANGE_LIST[*]}; do
	for MB in ${MUTATION_PROB_BINOMIAL_LIST[*]}; do
		SOL_NAME="$(get_solution_filename $ALGORITHM $SEED2 $POP_SIZE $N_GEN $SAMPLING_VERSION $CROSSOVER_VERSION $MUTATION_VERSION $MM $MC $MB)"
		MUT_FILES[${i}]="$SOL_PREFIX/$SOL_PATH/$SOL_NAME"
		i=$((i+1))
	done
	done
	done

	echo ${MUT_FILES[*]}
}

get_mutation_legend() {
	MUT_LEGEND=()
	i=0
	for MM in ${MUTATION_PROB_MOVE_LIST[*]}; do
	for MC in ${MUTATION_PROB_CHANGE_LIST[*]}; do
	for MB in ${MUTATION_PROB_BINOMIAL_LIST[*]}; do
		MUT_LEGEND[${i}]="MM$MM:MC$MC:MB$MB"
		i=$((i+1))
	done
	done
	done

	echo ${MUT_LEGEND[*]}
}

plot_comparison() {
	local VAR1=$1
	local INPUT="${VAR1:=algorithms}"
	local VAR2=$2
	local SUFFIX="${VAR2:=}"
	if [ $INPUT = "algorithms" ]; then
		FILES=$(get_algorithm_files)
		FILE_LEGEND=${ALGORITHMS[*]}
		TITLE="algorithms"
	elif [ $INPUT = "operator_versions" ]; then
		FILES=$(get_operator_version_files)
		FILE_LEGEND=$(get_operator_version_legend)
		TITLE="operator versions"
	elif [ $INPUT = "mutations" ]; then
		FILES=$(get_mutation_files)
		FILE_LEGEND=$(get_mutation_legend)
		TITLE="mutations"
	fi

	local NTW_NAME="$(get_network_filename $SEED $NODES $TASKS $USERS $COMMUNITIES)"
	local REF_PATH="$(get_ref_points_path $NTW_NAME $N_REPLICAS ${OBJECTIVES[@]})"
	local REF_NAME="$(get_ref_points_filename $REF_POINTS_ALGORITHM)"
	local REF_FILE="$SOL_PREFIX/$REF_PATH/$REF_NAME"
	if [ -f "$REF_FILE" ]; then
		local REF_POINTS_STRING="$(cat $REF_FILE | tr -d '[:space:]')"
		local REF_POINTS_OPT=--ref_points
	else
		local REF_POINTS_STRING=
		local REF_POINTS_OPT=
	fi

	python3 main.py --seed $SEED plot \
		--objectives ${OBJECTIVES[*]} \
		--n_objectives $N_OBJECTIVES $REF_POINTS_OPT $REF_POINTS_STRING \
		--ref_points_legend "$REF_POINTS_ALGORITHM RP" \
		-i $FILES \
		--comparison \
		--legend $FILE_LEGEND \
		--title "Objective space - Comparison between $TITLE $SUFFIX - $NODES:$TASKS:$USERS"
}

send_telegram_message() {

	# Send message using Telegram Bot API to notify that the process has finished
	ME=$(basename "$0")
	TOKEN=$(cat ../token.txt)
	CHAT_ID=$(cat ../chat_id.txt)
	HOSTNAME=$(hostname)
	curl -X POST -H 'Content-Type: application/json' \
		-d '{"chat_id": '$CHAT_ID', "text": "Script '$ME' has finished executing on server '$HOSTNAME'"}' \
		"https://api.telegram.org/bot$TOKEN/sendMessage"
	echo
}



