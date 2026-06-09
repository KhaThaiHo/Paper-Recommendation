#!/usr/bin/env python3
# debug_transformers.py — local Transformers smoke test for the extraction prompt

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_DO_SAMPLE,
    MODEL_MAX_NEW_TOKENS,
    MODEL_NAME,
    MODEL_REPETITION_PENALTY,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_TRUST_REMOTE_CODE,
)


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=MODEL_TRUST_REMOTE_CODE)
    model_kwargs = {"trust_remote_code": MODEL_TRUST_REMOTE_CODE, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs.update({"device_map": "auto", "torch_dtype": "auto"})
    else:
        model_kwargs.update({"torch_dtype": torch.float32})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate(tokenizer, model, messages):
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        prompt = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages) + "\nASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MODEL_MAX_NEW_TOKENS,
            do_sample=MODEL_DO_SAMPLE,
            temperature=MODEL_TEMPERATURE,
            top_p=MODEL_TOP_P,
            repetition_penalty=MODEL_REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    tokenizer, model = build_model()
    messages = [
        {
            "role": "user",
            "content": "Reply with this exact JSON: {\"hello\": \"world\"}",
        }
    ]
    output = generate(tokenizer, model, messages)
    print(json.dumps({"model": MODEL_NAME, "output": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
