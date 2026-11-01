import json
import os
from collections import Counter
from pydantic import ValidationError

from schema import Listing
from inference import generate_json, strip_fence

def run_one(text, expected, model, generate=generate_json):
    raw = generate(model, text)
    clean = strip_fence(raw)

    result = {
        'raw': raw,
        'clean': clean,
        'fenced': clean != raw.strip(),
        'status': None,
        'error': None,
        'parsed': None,
        'match': 0
    }

    try:
        json.loads(clean)
    except json.JSONDecodeError as e:
        result["status"] = "not_json"
        result["error"] = str(e)
        return result

    try:
        obj = Listing.model_validate_json(clean)
    except ValidationError as e:
        result['status'] = 'bad_schema'
        result['error'] = str(e)
        return result

    result['status'] = 'ok'
    result['parsed'] = obj.model_dump()

    gold = Listing.model_validate(expected).model_dump()
    result['match'] = sum([result['parsed'][k] == gold[k] for k in gold])
    return result

def save_results(results, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_results(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def run_all(test_set, model, save_path=None, verbose=True, generate=generate_json):
    results = []
    for i, item in enumerate(test_set):
        r = run_one(item["text"], item["expected"], model, generate)
        r["text"] = item["text"]
        r["expected"] = item["expected"]
        results.append(r)

        if verbose:
            print(f'{i + 1}/{len(test_set)}', end=' ', flush=True)

    if save_path:
        save_results(results, save_path)

    return results

def field_errors(results):
    """Сколько раз ошибалось каждое поле. Ответы, не прошедшие схему, считаются отдельно."""
    errors = Counter()
    for r in results:
        if r['status'] != 'ok':
            errors[r['status']] += 1
            continue
        gold = Listing.model_validate(r['expected']).model_dump()
        for k in gold:
            if r['parsed'][k] != gold[k]:
                errors[k] += 1
    return errors

def metrics(results):
    n = len(results)
    n_fields = len(Listing.model_fields)

    statuses = Counter(r['status'] for r in results)

    return {
        'n': n,
        'statuses': dict(statuses),
        'valid_rate': f'{(statuses['ok'] / n):.2f}',
        'exact_match': f'{(sum(r['match'] == n_fields for r in results) / n):.2f}',
        'field_acc': f'{(sum(r['match'] for r in results) / (n * n_fields)):.2f}',
        'fenced_rate': f'{(sum(r['fenced'] for r in results) / n):.2f}'
    }
