######### Run following module from the projecte-aina ###########

# https://huggingface.co/projecte-aina/FLOR-6.3B

## --- console
## Gated model: Login with a HF token with gated access permission

# MODULES

# 	$	pip install transformers

# 	$	pip install torch

# 	$	pip install tensorflow

# 	$	pip install flax

#  Use a pipeline as a high-level helper

# 	$	pip install git+https://github.com/huggingface/accelerate

# 	$	huggingface-cli login

import os

is_transformer = False

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if is_transformer:

    # Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("text-generation", model="projecte-aina/FLOR-6.3B")

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("projecte-aina/FLOR-6.3B")
    model = AutoModelForCausalLM.from_pretrained("projecte-aina/FLOR-6.3B")

else:

    # With no transformers version

    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    input_text = "Sovint em trobo pensant en tot allò que"

    model_id  = "projecte-aina/FLOR-6.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generator = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        #device="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map=0,   # Usa device_map="auto" para mapeo automático con Accelerate
        max_new_tokens = 100,
        # Omitir 'max_model_input_sizes' si no es válido max_model_input_sizes = 20
        clean_up_tokenization_spaces=True,
    )
    generation = generator(
        input_text,
        do_sample=True,
        top_k=10,
        eos_token_id=tokenizer.eos_token_id,
    )
    #print(tokenizer.max_model_input_sizes)
    print(f"Result: {generation[0]['generated_text']}")