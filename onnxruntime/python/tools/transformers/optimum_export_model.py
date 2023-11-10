from optimum.onnxruntime import ORTModelForCausalLM

name = "meta-llama/Llama-2-7b-hf"
model = ORTModelForCausalLM.from_pretrained(
    name,
    export=True,
    use_auth_token=True,
)
model.save_pretrained(name.split("/")[-1] + "-onnx")