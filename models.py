import gc
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from config import model_name, qlora_model_name, device

def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

def make_model(target_modules, r=16, alpha=32, dropout=0.05):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        task_type='CAUSAL_LM'
    )

    torch.manual_seed(0)
    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    return model

def make_qlora_model(model_name, target_modules, r=16, alpha=32, dropout=0.05):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config(), device_map={'': 0})
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        task_type='CAUSAL_LM'
    )
    torch.manual_seed(0)

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model

def load_lora_model(adapter_path):
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model

def load_qlora_model(adapter_path, quantized=True):
    """quantized=True: база в 4 бита, как при обучении; False: база в fp16 (быстрее генерация)."""
    if quantized:
        base = AutoModelForCausalLM.from_pretrained(
            qlora_model_name, quantization_config=bnb_config(), device_map={'': 0})
    else:
        base = AutoModelForCausalLM.from_pretrained(qlora_model_name, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
