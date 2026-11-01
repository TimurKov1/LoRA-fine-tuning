import re
import torch

from config import tokenizer, device
from data_utils import PROMPT

def generate_json(model, text, max_new_tokens=200):
    model.eval()
    messages = [{'role': 'user', 'content': PROMPT.format(text=text)}]
    inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def strip_fence(raw):
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.removeprefix("```json").removeprefix("```").strip()

@torch.no_grad()
def chat(model, question, max_new_tokens=250):
    model.eval()
    messages = [{'role': 'user', 'content': question}]
    inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
