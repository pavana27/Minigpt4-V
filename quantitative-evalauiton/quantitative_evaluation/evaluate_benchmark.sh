#!/bin/bash

echo $OPENAI_API_KEY

# Define common arguments
PRED_GENERIC="/home/pavana/MiniGPT4-video/evaluation/mistral_engagenet_base_config_eval.json"
OUTPUT_DIR="/home/pavana/MiniGPT4-video/evaluation/mistral_engagenet_base_config_eval"

# Module 1: Correctness
python evaluation_main.py \
  --module 1 \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/correctness_eval" \
  --output_json "${OUTPUT_DIR}/correctness_results.json" \
  --api_key $OPENAI_API_KEY \
  --num_tasks $NUM_TASKS

# Module 2: Detailed Orientation
python evaluation_main.py \
  --module 2 \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/detailed_eval" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_results.json" \
  --api_key $OPENAI_API_KEY \
  --num_tasks $NUM_TASKS

# Module 3: Contextual Understanding
python evaluation_main.py \
  --module 3 \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/context_eval" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_results.json" \
  --api_key $OPENAI_API_KEY \
  --num_tasks $NUM_TASKS

# Module 4: Temporal Understanding
python evaluation_main.py \
  --module 4 \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/temporal_eval" \
  --output_json "${OUTPUT_DIR}/temporal_understanding_results.json" \
  --api_key $OPENAI_API_KEY \
  --num_tasks $NUM_TASKS

# Module 5: Consistency
python evaluation_main.py \
  --module 5 \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/consistency_eval" \
  --output_json "${OUTPUT_DIR}/consistency_results.json" \
  --api_key $OPENAI_API_KEY \
  --num_tasks $NUM_TASKS

echo "All evaluations completed!"

