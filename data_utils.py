import json
import torch
from torch.utils.data import DataLoader

from config import tokenizer

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]

PROMPT = "Извлеки данные из объявления о сдаче квартиры в JSON.\n\nОбъявление: {text}"

def build_message(example):
    user = PROMPT.format(text=example['text'])
    answer = json.dumps(example['expected'], ensure_ascii=False)

    return [
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': answer}
    ]

def encode(example):
    message = build_message(example)

    prompt_ids = tokenizer.apply_chat_template(
        message[:1], tokenize=True, add_generation_prompt=True,
    )['input_ids']

    full_ids = tokenizer.apply_chat_template(
        message, tokenize=True
    )['input_ids']

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {'input_ids': full_ids, 'labels': labels}

def collate(batch):
    max_len = max(len(ex['input_ids']) for ex in batch)
    input_ids, labels, attention_mask = [], [], []

    for ex in batch:
        pad = max_len - len(ex['input_ids'])
        input_ids.append(ex['input_ids'] + [tokenizer.pad_token_id] * pad)
        labels.append(ex['labels'] + [-100] * pad)
        attention_mask.append([1] * len(ex['input_ids']) + [0] * pad)

    return {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),
        'attention_mask': torch.tensor(attention_mask)
    }


def make_loader(dataset, shuffle, batch_size=4):
    encoded = [encode(e) for e in dataset]
    return DataLoader(encoded, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
