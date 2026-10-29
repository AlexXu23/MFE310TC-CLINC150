"""Fine-tune BERT on CLINC150 in-scope intents and export validation/test logits.

The model is trained only on the 150 in-scope training classes. The best epoch
is selected using in-scope validation macro-F1. Exported validation and test
arrays include both in-scope and OOS examples so downstream rejection and
calibration experiments can use exactly the same frozen model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from common import load_dataset, validate_full_dataset


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    max_length: int
    learning_rate: float
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    weight_decay: float
    warmup_ratio: float
    seed: int
    gradient_clip_norm: float
    fp16: bool


class EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit; do not use for final experiments.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(rows: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    pairs = list(rows)
    return [row[0] for row in pairs], [row[1] for row in pairs]


def encode_dataset(tokenizer, texts: list[str], labels: list[int], max_length: int) -> EncodedDataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return EncodedDataset(dict(encodings), labels)


def make_loader(
    dataset: Dataset,
    collator: DataCollatorWithPadding,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.inference_mode()
def collect_logits(model, loader: DataLoader, device: torch.device, use_fp16: bool) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            logits = model(**batch).logits
        chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def save_export(
    name: str,
    logits: np.ndarray,
    texts: list[str],
    labels: list[str],
    source_splits: list[str],
    label_to_id: dict[str, int],
    output: Path,
) -> None:
    label_ids = np.array([label_to_id.get(label, -1) for label in labels], dtype=np.int64)
    is_oos = np.array([label == "oos" for label in labels], dtype=bool)
    if logits.shape != (len(texts), len(label_to_id)):
        raise ValueError(
            f"Unexpected {name} logits shape {logits.shape}; "
            f"expected {(len(texts), len(label_to_id))}"
        )

    np.save(output / f"{name}_logits.npy", logits, allow_pickle=False)
    np.savez_compressed(
        output / f"{name}_targets.npz",
        label_ids=label_ids,
        is_oos=is_oos,
    )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(texts), dtype=np.int64),
            "source_split": source_splits,
            "true_label": labels,
            "label_id": label_ids,
            "is_oos": is_oos,
            "text": texts,
        }
    )
    metadata.to_csv(output / f"{name}_metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if (output / "best_model").exists() or (output / "run_summary.json").exists():
        raise FileExistsError("Training outputs already exist; select a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "best_model"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Pass a CUDA-enabled PyTorch build or "
            "use a separate CPU-oriented configuration explicitly."
        )

    device = torch.device("cuda")
    use_fp16 = not args.no_fp16
    config = RunConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        gradient_clip_norm=args.gradient_clip_norm,
        fp16=use_fp16,
    )
    set_seed(config.seed)
    started = time.time()

    dataset = load_dataset(args.data)
    validation = validate_full_dataset(dataset)
    labels = validation["in_scope_intents"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    train_rows = dataset["train"]
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    train_texts, train_labels = unpack(train_rows)
    val_texts, val_labels = unpack(dataset["val"])
    oos_val_texts, oos_val_labels = unpack(dataset["oos_val"])
    test_texts, test_labels = unpack(dataset["test"])
    oos_test_texts, oos_test_labels = unpack(dataset["oos_test"])

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)

    train_data = encode_dataset(
        tokenizer,
        train_texts,
        [label_to_id[label] for label in train_labels],
        config.max_length,
    )
    val_data = encode_dataset(
        tokenizer,
        val_texts,
        [label_to_id[label] for label in val_labels],
        config.max_length,
    )
    validation_export_texts = val_texts + oos_val_texts
    validation_export_labels = val_labels + oos_val_labels
    validation_data = encode_dataset(
        tokenizer,
        validation_export_texts,
        [label_to_id.get(label, -1) for label in validation_export_labels],
        config.max_length,
    )
    test_export_texts = test_texts + oos_test_texts
    test_export_labels = test_labels + oos_test_labels
    test_data = encode_dataset(
        tokenizer,
        test_export_texts,
        [label_to_id.get(label, -1) for label in test_export_labels],
        config.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = make_loader(
        train_data, collator, config.train_batch_size, True, generator=generator
    )
    val_loader = make_loader(val_data, collator, config.eval_batch_size, False)

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(round(total_steps * config.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    best_macro_f1 = -1.0
    best_epoch = None
    history = []
    print(
        f"device={torch.cuda.get_device_name(0)} train={len(train_data)} "
        f"val={len(val_data)} classes={len(labels)} steps={total_steps}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler may skip an optimizer step after detecting overflow.
            # Advance the learning-rate schedule only when parameters changed.
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
            epoch_loss += float(loss.detach().cpu())
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch}/{config.epochs} step={step}/{len(train_loader)} "
                    f"loss={epoch_loss / step:.5f} lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )

        val_logits = collect_logits(model, val_loader, device, use_fp16)
        val_prediction = val_logits.argmax(axis=1)
        val_targets = np.array([label_to_id[label] for label in val_labels])
        val_accuracy = float(accuracy_score(val_targets, val_prediction))
        val_macro_f1 = float(f1_score(val_targets, val_prediction, average="macro"))
        epoch_result = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(epoch_result)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps(epoch_result), flush=True)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"saved new best checkpoint at epoch {epoch}", flush=True)

    del model
    torch.cuda.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)

    validation_loader = make_loader(
        validation_data, collator, config.eval_batch_size, False
    )
    test_loader = make_loader(test_data, collator, config.eval_batch_size, False)
    validation_logits = collect_logits(best_model, validation_loader, device, use_fp16)
    test_logits = collect_logits(best_model, test_loader, device, use_fp16)

    save_export(
        "validation",
        validation_logits,
        validation_export_texts,
        validation_export_labels,
        ["val"] * len(val_texts) + ["oos_val"] * len(oos_val_texts),
        label_to_id,
        output,
    )
    save_export(
        "test",
        test_logits,
        test_export_texts,
        test_export_labels,
        ["test"] * len(test_texts) + ["oos_test"] * len(oos_test_texts),
        label_to_id,
        output,
    )

    test_in_scope_logits = test_logits[: len(test_texts)]
    test_targets = np.array([label_to_id[label] for label in test_labels])
    test_prediction = test_in_scope_logits.argmax(axis=1)
    test_accuracy = float(accuracy_score(test_targets, test_prediction))
    test_macro_f1 = float(f1_score(test_targets, test_prediction, average="macro"))

    mapping = {
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
        "oos_label_id": -1,
    }
    (output / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "config": asdict(config),
        "data_sha256": sha256(args.data),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "test_in_scope_accuracy": test_accuracy,
        "test_in_scope_macro_f1": test_macro_f1,
        "validation_logits_shape": list(validation_logits.shape),
        "test_logits_shape": list(test_logits.shape),
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (output / "run_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

