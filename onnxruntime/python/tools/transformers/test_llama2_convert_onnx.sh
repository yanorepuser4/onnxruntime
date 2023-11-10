#!/bin/bash
ep=$1
precision=$2
python3 -m models.llama.benchmark \
    --benchmark-type ort-convert-to-onnx \
    --ort-model-path ./llama2-7b/rank_0_Llama-2-7b-hf_decoder_merged_model_fp32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision $precision \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device $ep