import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from config import tokenizer, device
from schema import Listing

lines = []
for name, field in Listing.model_fields.items():
    lines.append(f"- {name}: {field.description}")
fields = "\n".join(lines)

example_text = (
    "Сдаётся 1-комнатная квартира 38 кв.м на 6 этаже 12-этажного дома рядом с метро Таганская. 50 000 руб. в месяц. Животные только по договорённости. Звоните: +7 903 444-55-66, Павел."
)
example_json = {
    "rooms": 1, "area_m2": 38.0, "floor": 6, "total_floors": 12,
    "metro": "Таганская", "price_rub": 50000, "pets_allowed": None,
    "phone": "79034445566", "contact_name": "Павел",
}

SYSTEM_PROMPT = f"""
Извлеки данные из объявления о сдаче квартиры и верни ТОЛЬКО JSON, без пояснений, без markdown, без слов json, только значения в фигурных скобках.
Верни JSON-объект ровно с такими ключами, все ключи обязательны:
{fields}

Правила:
- Если значения нет в тексте, пиши null. Ничего не выдумывай.
- Числа пиши числами, без пробелов и единиц измерения.
- Не добавляй других ключей.

Пример.
Объявление: {example_text}
JSON: {example_json}
"""

def answers(prompt, model, max_new_tokens=500):
    prompt = f"""
    Объявление: {prompt}
    JSON:
    """
    messages = [
        {'role': 'system',
        'content': SYSTEM_PROMPT},
        {'role': 'user',
        'content': prompt},
    ]

    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
    output_ids = model.generate(**text, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(output_ids[0][text['input_ids'].shape[1]:], skip_special_tokens=True)
    return answer
