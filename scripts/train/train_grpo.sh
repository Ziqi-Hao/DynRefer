#! /bin/bash

# Train view selector via GRPO
echo "config: $1"
python grpo_train.py --cfg-path $1
