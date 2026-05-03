from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"

RANDOM_SEED = 0

# OUTSIDE_ACTOR: token is not part of an actor's span
# ACTOR_SPAN_START: token is the first token in an actor span
# ACTOR_SPAN_INSIDE: token is inside an actor span but not the first token

OUTSIDE_ACTOR = "outside_actor"
ACTOR_SPAN_START = "actor_span_start"
ACTOR_SPAN_INSIDE = "actor_span_inside"

TAGS: List[str] = [OUTSIDE_ACTOR, ACTOR_SPAN_START, ACTOR_SPAN_INSIDE]
TAG2ID: Dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID2TAG: Dict[int, str] = {i: t for t, i in TAG2ID.items()}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_DATA_CSV = _project_root() / "data" / "gender_dataset.csv"
DEFAULT_MODEL_DIR = _project_root() / "models" / "actor_distilbert"

TRAIN_EPOCHS = 4.0
TRAIN_BATCH_SIZE = 16
TRAIN_LEARNING_RATE = 5e-5
MAX_SEQUENCE_LENGTH = 128


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def load_csv(csv_path: Path) -> pd.DataFrame:
    data_frame = pd.read_csv(csv_path)
    required_columns = ["sentence", "actor_start", "actor_end"]
    for column in required_columns:
        if column not in data_frame.columns:
            raise ValueError(f"CSV must contain column '{column}'.")
    data_frame = data_frame.dropna(subset=required_columns)
    data_frame["actor_start"] = data_frame["actor_start"].astype(int)
    data_frame["actor_end"] = data_frame["actor_end"].astype(int)
    return data_frame


def stratified_split(data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_values = data_frame["ambiguity_label"] if "ambiguity_label" in data_frame.columns else None
    train_data_frame, temp_data_frame = train_test_split(
        data_frame, test_size=0.2, stratify=stratify_values, random_state=RANDOM_SEED
    )
    stratify_temp_values = (
        temp_data_frame["ambiguity_label"] if "ambiguity_label" in temp_data_frame.columns else None
    )
    validation_data_frame, test_data_frame = train_test_split(
        temp_data_frame, test_size=0.5, stratify=stratify_temp_values, random_state=RANDOM_SEED
    )
    return (
        train_data_frame.reset_index(drop=True),
        validation_data_frame.reset_index(drop=True),
        test_data_frame.reset_index(drop=True),
    )


# Converts character-based actor spans to token-aligned label IDs for a sentence, using offset mappings.
def _align_labels_for_sentence(
    offset_mapping: List[Tuple[int, int]],
    actor_span_start_char: int,
    actor_span_end_char: int,
) -> List[int]:
    aligned_labels: List[int] = []
    found_actor_start = False
    for token_start_char, token_end_char in offset_mapping:
        if token_start_char == 0 and token_end_char == 0:
            aligned_labels.append(-100)
            continue
        token_overlaps_actor = (
            token_start_char < actor_span_end_char and token_end_char > actor_span_start_char
        )
        if not token_overlaps_actor:
            aligned_labels.append(TAG2ID[OUTSIDE_ACTOR])
        elif not found_actor_start:
            aligned_labels.append(TAG2ID[ACTOR_SPAN_START])
            found_actor_start = True
        else:
            aligned_labels.append(TAG2ID[ACTOR_SPAN_INSIDE])
    return aligned_labels


# Tokenizes and aligns labels for a batch of sentences, using offset mappings.
def _tokenize_and_align(
    examples_batch: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    tokenized_encoding = tokenizer(
        examples_batch["sentence"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    aligned_labels_batch: List[List[int]] = []
    for i in range(len(examples_batch["sentence"])):
        labels_for_sentence = _align_labels_for_sentence(
            tokenized_encoding["offset_mapping"][i],
            int(examples_batch["actor_start"][i]),
            int(examples_batch["actor_end"][i]),
        )
        aligned_labels_batch.append(labels_for_sentence)
    tokenized_encoding["labels"] = aligned_labels_batch
    del tokenized_encoding["offset_mapping"]
    return tokenized_encoding


def build_compute_metrics():
    def compute_metrics(eval_prediction):
        prediction_logits, reference_labels = eval_prediction
        predicted_labels = np.argmax(prediction_logits, axis=-1)
        valid_label_mask = reference_labels != -100
        filtered_predicted_labels = predicted_labels[valid_label_mask]
        filtered_reference_labels = reference_labels[valid_label_mask]
        return {
            "accuracy": float(accuracy_score(filtered_reference_labels, filtered_predicted_labels)),
            "f1_macro": float(
                f1_score(filtered_reference_labels, filtered_predicted_labels, average="macro", zero_division=0)
            ),
        }

    return compute_metrics


def decode_first_actor_span(
    predicted_label_ids: np.ndarray,
    offset_mapping: List[Tuple[int, int]],
    id_to_tag: Dict[int, str],
) -> Optional[Tuple[int, int]]:
    token_idx = 0
    num_tokens = len(predicted_label_ids)
    while token_idx < num_tokens:
        token_start_char, token_end_char = offset_mapping[token_idx]
        if token_start_char == 0 and token_end_char == 0:
            token_idx += 1
            continue
        tag_label = id_to_tag.get(int(predicted_label_ids[token_idx]), OUTSIDE_ACTOR)
        if tag_label == ACTOR_SPAN_START:
            span_start_char = token_start_char
            span_end_char = token_end_char
            token_idx += 1
            while token_idx < num_tokens:
                token_start_char_2, token_end_char_2 = offset_mapping[token_idx]
                if token_start_char_2 == 0 and token_end_char_2 == 0:
                    token_idx += 1
                    continue
                tag_label_2 = id_to_tag.get(int(predicted_label_ids[token_idx]), OUTSIDE_ACTOR)
                if tag_label_2 == ACTOR_SPAN_INSIDE:
                    span_end_char = token_end_char_2
                    token_idx += 1
                else:
                    break
            return (span_start_char, span_end_char)
        token_idx += 1
    return None


def span_exact_match_metrics(
    data_frame: pd.DataFrame,
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    max_length: int,
    split_name: str,
) -> Dict[str, float]:
    model = trainer.model
    device = next(model.parameters()).device
    model.eval()
    exact_match_count = 0
    total_examples = len(data_frame)
    for _, row in data_frame.iterrows():
        sentence = str(row["sentence"])
        gold_actor_start, gold_actor_end = int(row["actor_start"]), int(row["actor_end"])
        encoding = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offset_mapping_tensor = encoding.pop("offset_mapping")[0]
        if isinstance(offset_mapping_tensor, torch.Tensor):
            offset_mapping_list = offset_mapping_tensor.tolist()
        else:
            offset_mapping_list = list(offset_mapping_tensor)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = model(**encoding).logits
        predicted_label_ids = logits.argmax(-1)[0].cpu().numpy()
        id_to_tag_dict: Dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
        predicted_span = decode_first_actor_span(predicted_label_ids, offset_mapping_list, id_to_tag_dict)
        if predicted_span is not None and predicted_span[0] == gold_actor_start and predicted_span[1] == gold_actor_end:
            exact_match_count += 1
    exact_match_accuracy = exact_match_count / total_examples if total_examples else 0.0
    print(f"\n Span exact match ({split_name})")
    print(f"exact_match={exact_match_accuracy}  ({exact_match_count}/{total_examples})\n")
    return {"span_exact_match": float(exact_match_accuracy)}


def train_localizer() -> Dict[str, Any]:
    set_seed()
    data_frame = load_csv(DEFAULT_DATA_CSV)
    train_data_frame, validation_data_frame, test_data_frame = stratified_split(data_frame)

    raw_datasets = DatasetDict(
        train=Dataset.from_pandas(train_data_frame, preserve_index=False),
        validation=Dataset.from_pandas(validation_data_frame, preserve_index=False),
        test=Dataset.from_pandas(test_data_frame, preserve_index=False),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _tokenize_batch(batch_example):
        return _tokenize_and_align(batch_example, tokenizer, MAX_SEQUENCE_LENGTH)

    tokenized_datasets = DatasetDict(
        {
            split_key: raw_datasets[split_key].map(
                _tokenize_batch,
                batched=True,
                remove_columns=raw_datasets[split_key].column_names,
            )
            for split_key in raw_datasets
        }
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TAGS),
        id2label=ID2TAG,
        label2id=TAG2ID,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=str(DEFAULT_MODEL_DIR),
        learning_rate=TRAIN_LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        num_train_epochs=TRAIN_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        seed=RANDOM_SEED,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(DEFAULT_MODEL_DIR))
    tokenizer.save_pretrained(str(DEFAULT_MODEL_DIR))

    print("\nValidation metrics:")
    span_metrics_validation = span_exact_match_metrics(
        validation_data_frame, trainer, tokenizer, MAX_SEQUENCE_LENGTH, "validation"
    )
    span_metrics_test = span_exact_match_metrics(
        test_data_frame, trainer, tokenizer, MAX_SEQUENCE_LENGTH, "test"
    )

    return {
        "csv_path": str(DEFAULT_DATA_CSV),
        "model_dir": str(DEFAULT_MODEL_DIR),
        "rows_total": len(data_frame),
        "rows_train": len(train_data_frame),
        "rows_val": len(validation_data_frame),
        "rows_test": len(test_data_frame),
        "span_metrics_validation": span_metrics_validation,
        "span_metrics_test": span_metrics_test,
    }


if __name__ == "__main__":
    print("Training summary:", train_localizer())
