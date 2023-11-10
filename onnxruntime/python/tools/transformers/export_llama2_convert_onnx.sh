#!/bin/bash
ep=$1
precision=$2
python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-$ep --precision $precision --execution_provider $ep