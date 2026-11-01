import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import llama_cpp_dir
from data_utils import PROMPT

OLLAMA_URL = 'http://localhost:11434/api/chat'

# системное сообщение, которое шаблон Qwen подставляет по умолчанию: при обучении оно было в каждом примере
SYSTEM_PROMPT = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

def merge_adapter(adapter_path, out_dir):
    # слияние в fp32, чтобы малые поправки адаптера не терялись при округлении
    base_name = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_name).save_pretrained(out_dir)

def convert_to_gguf(model_dir, gguf_path):
    script = Path(llama_cpp_dir) / 'convert_hf_to_gguf.py'
    subprocess.run([sys.executable, '-B', str(script), str(model_dir), '--outfile', str(gguf_path), '--outtype', 'f16'], check=True)

def quantize_gguf(src, dst, quant):
    quantize = Path(llama_cpp_dir) / 'build' / 'bin' / 'llama-quantize'
    subprocess.run([str(quantize), str(src), str(dst), quant.upper()], check=True)

def modelfile_text(gguf_path):
    # шаблон повторяет тот, на котором обучалась модель; если он отличается, качество падает без всяких ошибок
    return (
        f'FROM {Path(gguf_path).resolve()}\n'
        'TEMPLATE """<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"""\n'
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|endoftext|>"\n'
        'PARAMETER temperature 0\n'
        'PARAMETER num_ctx 4096\n'
    )

def create_ollama_model(tag, gguf_path):
    modelfile = Path(gguf_path).with_suffix('.Modelfile')
    modelfile.write_text(modelfile_text(gguf_path))
    subprocess.run(['ollama', 'create', tag, '-f', str(modelfile)], check=True)

def export_gguf(adapter_path, name, quants=('q8_0', 'q4_k_m'), out_dir='gguf', keep_intermediate=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = out_dir / f'{name}_merged'
    f16 = out_dir / f'{name}-f16.gguf'

    merge_adapter(adapter_path, merged)
    convert_to_gguf(merged, f16)

    for quant in quants:
        gguf = out_dir / f'{name}-{quant}.gguf'
        quantize_gguf(f16, gguf, quant)
        create_ollama_model(f'{name}-{quant}', gguf)

    if not keep_intermediate:
        shutil.rmtree(merged)
        f16.unlink()

def ollama_generate(tag, text, max_new_tokens=200):
    body = json.dumps({
        'model': tag,
        'messages': [{'role': 'user', 'content': PROMPT.format(text=text)}],
        'stream': False,
        'options': {'temperature': 0, 'num_predict': max_new_tokens, 'seed': 0},
    }).encode()
    request = urllib.request.Request(OLLAMA_URL, body, {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(request, timeout=600))['message']['content']
