#!/bin/bash

# Run all jobs in parallel
# the experiments were on the zero-shot setting for all the models and the datasets
for RunId in {1..9}; do
    echo "Starting RunId: $RunId"
    python3.9 clean_inference_code.py \
        --dataset_name 'GTSRB' \
        --model 'gpt-4o' \
        --disable_tree \
        --RunId $RunId &
done

# Wait for all background jobs to complete
wait
echo "All runs completed!"