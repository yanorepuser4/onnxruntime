#!/bin/bash
ep=$1
python3 -m models.llama.benchmark \
    --benchmark-type hf-ort \
    --hf-ort-dir-path ./Llama-2-7b-hf-onnx/ \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device $ep \
    --auth