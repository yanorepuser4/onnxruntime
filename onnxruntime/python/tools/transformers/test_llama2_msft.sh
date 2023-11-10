#!/bin/bash
ep=$1
precision=$2
python3 -m models.llama.benchmark \
    --benchmark-type ort-msft \
    --ort-model-path ./Llama-2-Onnx/7B_float32/ONNX/LlamaV2_7B_float32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision $precision \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device $ep