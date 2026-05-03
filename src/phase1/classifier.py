from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"

RANDOM_SEED = 0

LABEL2ID = {"ambiguous": 0, "unambiguous": 1}
ID2LABEL = {0: "ambiguous", 1: "unambiguous"}

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

DEFAULT_DATA_CSV = _project_root() / "data" / "gender_dataset.csv"
DEFAULT_MODEL_DIR = _project_root() / "models" / "ambiguity_distilbert"

TRAIN_EPOCHS = 4.0
TRAIN_BATCH_SIZE = 16
TRAIN_LEARNING_RATE = 5e-5
MAX_SEQUENCE_LENGTH = 128


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def load_csv(csv_file_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_file_path)
    if "ambiguity_label" not in dataframe.columns or "sentence" not in dataframe.columns:
        raise ValueError("CSV must contain columns 'sentence' and 'ambiguity_label'.")
    dataframe = dataframe.dropna(subset=["sentence", "ambiguity_label"])
    dataframe["labels"] = dataframe["ambiguity_label"].map(LABEL2ID)
    if dataframe["labels"].isna().any():
        unknown_labels = dataframe.loc[dataframe["labels"].isna(), "ambiguity_label"].unique()
        raise ValueError(f"Unknown ambiguity_label values: {unknown_labels}")
    return dataframe


def stratified_split(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_dataframe, temp_dataframe = train_test_split(
        dataframe,
        test_size=0.2,
        stratify=dataframe["labels"],
        random_state=RANDOM_SEED,
    )
    validation_dataframe, testing_dataframe = train_test_split(
        temp_dataframe,
        test_size=0.5,
        stratify=temp_dataframe["labels"],
        random_state=RANDOM_SEED,
    )
    return (
        training_dataframe.reset_index(drop=True),
        validation_dataframe.reset_index(drop=True),
        testing_dataframe.reset_index(drop=True),
    )


def _build_dataset_dict(
    training_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    testing_dataframe: pd.DataFrame
) -> DatasetDict:
    columns_to_keep = ["sentence", "labels"]
    return DatasetDict(
        train=Dataset.from_pandas(training_dataframe[columns_to_keep], preserve_index=False),
        validation=Dataset.from_pandas(validation_dataframe[columns_to_keep], preserve_index=False),
        test=Dataset.from_pandas(testing_dataframe[columns_to_keep], preserve_index=False),
    )


def build_compute_metrics():
    def compute_metrics(eval_prediction):
        prediction_logits, reference_labels = eval_prediction
        predicted_labels = np.argmax(prediction_logits, axis=-1)
        precision, recall, f1_score_macro, _ = precision_recall_fscore_support(
            reference_labels, predicted_labels, average="macro", zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(reference_labels, predicted_labels)),
            "f1_macro": float(f1_score(reference_labels, predicted_labels, average="macro", zero_division=0)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
        }

    return compute_metrics


def _format_confusion_matrix(conf_matrix: np.ndarray) -> str:
    s = ""
    s += "              pred amb  pred unamb\n"
    s += f"true amb      {conf_matrix[0, 0]}  {conf_matrix[0, 1]}\n"
    s += f"true unamb    {conf_matrix[1, 0]}  {conf_matrix[1, 1]}"
    return s


def evaluate_split(
    trainer: Trainer, dataset: Dataset, split_name: str
) -> Dict[str, Any]:
    prediction_output = trainer.predict(dataset)
    true_labels = np.array(dataset["labels"])
    predicted_labels = np.argmax(prediction_output.predictions, axis=-1)
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "f1_macro": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
    }
    print(f"\n--- {split_name} ---")
    print(_format_confusion_matrix(conf_matrix))
    print(
        f"accuracy={metrics['accuracy']}  macro-F1={metrics['f1_macro']}\n"
    )
    return metrics


def train_classifier() -> Dict[str, Any]:
    set_seed()
    dataframe = load_csv(DEFAULT_DATA_CSV)
    training_dataframe, validation_dataframe, testing_dataframe = stratified_split(dataframe)
    dataset_splits = _build_dataset_dict(training_dataframe, validation_dataframe, testing_dataframe)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example_batch: Dict[str, Any]) -> Dict[str, Any]:
        encoding = tokenizer(
            example_batch["sentence"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding=False,
        )
        return encoding

    tokenized_datasets = DatasetDict(
        {
            split_name: dataset_splits[split_name].map(
                tokenize,
                batched=True,
                remove_columns=["sentence"],
            )
            for split_name in dataset_splits
        }
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
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
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(DEFAULT_MODEL_DIR))
    tokenizer.save_pretrained(str(DEFAULT_MODEL_DIR))

    print("\nValidation metrics:")
    evaluate_split(trainer, tokenized_datasets["validation"], "validation")
    test_metrics = evaluate_split(trainer, tokenized_datasets["test"], "test")

    return {
        "csv_path": str(DEFAULT_DATA_CSV),
        "model_dir": str(DEFAULT_MODEL_DIR),
        "rows_total": len(dataframe),
        "rows_train": len(training_dataframe),
        "rows_val": len(validation_dataframe),
        "rows_test": len(testing_dataframe),
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    print("Training summary:", train_classifier())
