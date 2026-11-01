import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM

from config import model_name, tokenizer, device

# в ноутбуке эта строка была закомментирована, а переменная `model` дальше использовалась
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

messages = [
    {'role': 'user', 'content': 'Напиши код на Python, который реализует калькулятор'}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_new_tokens=100, temperature=1.0, do_sample=True, top_p=0.8)

new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=False))

input_ids = inputs['input_ids']

for i in range(200):
    with torch.no_grad():
        outputs = model(input_ids)

    last_logit = outputs.logits[0, -1, :]
    next_token = torch.argmax(last_logit, dim=-1).view(1, 1)

    input_ids = torch.cat([input_ids, next_token], dim=-1)
    print(tokenizer.decode(next_token[0]), end='')
