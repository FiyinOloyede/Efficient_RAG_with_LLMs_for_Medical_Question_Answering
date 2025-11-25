from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

def login_hf(token):
    login(token=token)

# Load model and setup generation pipeline
def load_model_and_tokenizer(model_name, quantization=None):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
        # load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    elif quantization == "4bit":
        load_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Setup generation pipeline
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        return_full_text=False
    )
    return model_pipeline