#!/bin/bash
export OPENAI_API_KEY="your key here"

## exp utils
script_name=`basename "$0"`
id=${script_name%.*}
entry=${entry:-'src'}
tag=${tag:-"none"}
override=${override:-""}
group=${group:-"default"}
## data
benchmark=${benchmark:-'ii'}
datasets=${datasets:-'24'}
## method
method=${method:-'mop'}
num_exps=${num_exps:-3}
n_experts=${n_experts:-10}

## collect command line parameters
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

IFS=',' read -ra dataset_array <<< "$datasets"

#### run experiment
cd ../

## src directory management
entry_copy="$entry-$(date +"%Y-%m-%d_%H-%M-%S")"
cp -r "$entry" "$entry_copy"

# Iterate over the items and echo each one
for dataset in "${dataset_array[@]}";
do
    echo $dataset
    python ${entry_copy}/run_mop.py \
        --group $group --save $id --tag $tag --num_exps $num_exps \
        --benchmark $benchmark --task $dataset \
        --method $method --n_experts $n_experts
done
