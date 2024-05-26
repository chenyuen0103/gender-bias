import os

from unsloth import FastLanguageModel
import torch

max_seq_length = 1 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

HF_TOKEN = os.environ.get("HF_TOKEN")


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth


print(f"Downloading model: unsloth/llama-3-8b-bnb-4bit", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
print(f"Model downloaded: unsloth/llama-3-8b-bnb-4bit", flush=True)

model.eval()  # Set the model to evaluation mode
# Define the input text
input_text = "Once upon a time in a land far, far away,"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors='pt').to(model.device)

# Forward pass to get logits
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Compute log probabilities
log_probs = torch.log_softmax(logits, dim=-1)

# Extract log probabilities for each token
input_ids = inputs['input_ids']
sequence_log_probs = log_probs[0, range(len(input_ids[0])), input_ids[0]]

# Print the log probabilities
for token, log_prob in zip(tokenizer.convert_ids_to_tokens(input_ids[0]), sequence_log_probs):
    print(f"Token: {token}, Log Probability: {log_prob.item()}")