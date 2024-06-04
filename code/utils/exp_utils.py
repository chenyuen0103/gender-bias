import pandas as pd
import os
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import math
import torch
import numpy as np
import argparse
from itertools import product
from efficiency.function import set_seed


def get_logprobs(model, tokenizer, prompt):
    # Tokenize the prompt and convert to PyTorch tensors
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)


    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])

    # Compute log-probabilities of the token predictions
    logprobs = torch.log_softmax(outputs.logits, dim=-1)

    # Extract the log-probabilities of the input tokens
    input_token_ids = inputs['input_ids']
    token_logprobs = logprobs.gather(-1, input_token_ids.unsqueeze(-1)).squeeze(-1)

    return token_logprobs, input_token_ids


def get_probs(model, tokenizer, prompt):
    # Tokenize the prompt and convert to PyTorch tensors
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Perform a forward pass through the model without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    # Shift logits and labels to align them
    shift_probs = probs[:, :-1, :].contiguous()
    shift_input_ids = inputs['input_ids'][:, 1:].contiguous()

    # Gather the probabilities corresponding to the actual next tokens
    next_token_probs = shift_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)

    return next_token_probs, shift_input_ids

def setup_model(model_str):
    # Load the tokenizer and model from Hugging Face
    if model_str == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        return model, tokenizer

    if model_str == 'llama3-8b':
        model_id = "meta-llama/Meta-Llama-3-8B"
    elif model_str == 'llama3-8b-instruct':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_str == 'mistral-7b':
        model_id = "mistralai/Mistral-7B-v0.3"
    elif model_str == 'mistral-7b-instruct':
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model_str == 'llama2-7b':
        model_id = "meta-llama/Llama-2-7b-hf"
    elif model_str == 'llama2-7b-chat':
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    elif model_str == 'llama2-7b-instruct':
        model_id = "togethercomputer/Llama-2-7B-32K-Instruct"
    elif model_str == 'llama3-70b':
        model_id = "meta-llama/Meta-Llama-3-70B"
    elif model_str == 'llama3-70b-instruct':
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif model_str == 'alpaca-7b':
        model_id = "allenai/open-instruct-stanford-alpaca-7b"
    HF_TOKEN = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 # device_map="auto",
                                                 # quantization_config=quantization_config,
                                                 token=HF_TOKEN)
    return model, tokenizer

def get_logprobs(model, tokenizer, prompt):
    # Tokenize the prompt and convert to PyTorch tensors
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)


    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])

    # Compute log-probabilities of the token predictions
    logprobs = torch.log_softmax(outputs.logits, dim=-1)

    # Extract the log-probabilities of the input tokens
    input_token_ids = inputs['input_ids']
    token_logprobs = logprobs.gather(-1, input_token_ids.unsqueeze(-1)).squeeze(-1)

    return token_logprobs, input_token_ids


def get_top_k(model, tokenizer, prompt, top_k=10):
    """
    Get the top K tokens with their probabilities given a model, tokenizer, and prompt.

    Args:
    model: The pre-trained language model.
    tokenizer: The tokenizer associated with the model.
    prompt: The input prompt (string).
    top_k: The number of top tokens to retrieve (default is 10).

    Returns:
    List of tuples containing top tokens and their probabilities.
    """

    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # Get the model outputs
    with torch.no_grad():
        outputs = model(input_ids)

    # Get the logits for the last token
    logits = outputs.logits[:, -1, :]

    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get the top K tokens with their probabilities
    top_probs, top_indices = torch.topk(probs, top_k)

    # Convert token indices to tokens
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices[0].tolist())

    # Prepare the list of tuples (token, probability)
    top_k_tokens_with_probs = [(token, prob.item()) for token, prob in zip(top_tokens, top_probs[0])]

    return top_k_tokens_with_probs