from pathlib import Path

import torch
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
qlora_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# где лежит llama.cpp (нужен для конвертации в GGUF и квантизации)
llama_cpp_dir = Path.home() / 'llama.cpp'
