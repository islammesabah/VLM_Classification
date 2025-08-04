we want all the variations for the arguments
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='GTSRB',
        choices=['GTSRB', 'CIFAR-10'],
        help='Dataset to use for inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['openai/gpt-4o', 'qwen-vl-max', 'meta-llama/llama-3.2-11b-vision-instruct', 'meta-llama/llama-3.2-90b-vision-instruct'],
        help='Model to use for inference'
    )    
    parser.add_argument(
        '--include_memory',
        action='store_true',
        help='Include history chat (questions and answers) in the LLM input of the tree inference'
    )

    parser.add_argument(
        '--include_description',
        action='store_true',
        help='Include description of the class in the LLM input'
    )
    parser.add_argument(
        '--include_zero_shot_label',
        action='store_true',
        help='Include zero-shot label in the LLM input for the tree inference'
    )

Done
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'openai/gpt-4o' \
    --include_memory

Done
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'qwen-vl-max' \
    --include_memory

Done
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --include_memory

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'openai/gpt-4o' \

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'qwen-vl-max' \

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'openai/gpt-4o' \
    --include_memory

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'qwen-vl-max' \
    --include_memory

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --include_memory

Done
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'openai/gpt-4o' \
    --include_description

DONE
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'qwen-vl-max' \
    --include_description

Running
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --include_description

Done
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'openai/gpt-4o' \
    --include_description

Done
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'qwen-vl-max' \
    --include_description

Running
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --include_description

Running
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'openai/gpt-4o' \
    --disable_tree

Running
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'qwen-vl-max' \
    --disable_tree

Running
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --disable_tree

Running
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'openai/gpt-4o' \
    --disable_tree

Running
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'qwen-vl-max' \
    --disable_tree

Running
python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --disable_tree

----------------------------------------

# description for the zero-shot label
python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'gpt-4o' \
    --disable_tree \
    --include_description 2

python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'qwen-vl-max' \
    --disable_tree \
    --include_description 2

python3.9 clean_inference_code.py \
    --dataset_name 'CIFAR-10' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --disable_tree \
    --include_description 2

python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'gpt-4o' \
    --disable_tree \
    --include_description 2

python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'qwen-vl-max' \
    --disable_tree \
    --include_description 2

python3.9 clean_inference_code.py \
    --dataset_name 'GTSRB' \
    --model 'meta-llama/llama-3.2-11b-vision-instruct' \
    --disable_tree \
    --include_description 2

GTSRB gpt 0.7 done
GTSRB qwen 0.7 done
CIFAR qwen 0.7 done
CIFAR gpt 0.7 running
GTSRB llama 0.7 running
CIFAR llama 0.7 running

