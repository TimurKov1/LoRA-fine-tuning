import torch
from transformers import get_linear_schedule_with_warmup

from config import device

def compute_loss(model, batch):
    min_index = min((row != -100).nonzero()[0].item() for row in batch['labels'])
    logits = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        logits_to_keep=batch['labels'].shape[1] - min_index + 1
    ).logits

    logits = logits[:, :-1]
    labels = batch['labels'][:, min_index:]

    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1,)

    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

    return loss

def evaluate_loss(model, loader):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)
            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

def train(model, loader, epochs=2, lr=2e-4, accum=4, max_updates=None, val_loader=None,
          save_dir='models/adapter'):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    total_updates = len(loader) * epochs // accum
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_updates)

    model.train()
    n_updates = 0
    losses = []

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)
            losses.append(loss.item())

            (loss / accum).backward()

            if (i + 1) % accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                n_updates += 1

                if n_updates % 5 == 0:
                    print(n_updates, sum(losses[-5 * accum:]) / len(losses[-5 * accum:]))

                if max_updates and n_updates >= max_updates:
                    return losses

        if val_loader is not None:
            print('Эпоха', epoch + 1, 'val loss', evaluate_loss(model, val_loader))

        model.save_pretrained(f'{save_dir}/epoch{epoch + 1}')

    model.save_pretrained(save_dir)    # итоговый адаптер лежит прямо в save_dir, чекпойнты эпох в подпапках

    return losses
