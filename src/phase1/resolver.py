from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_AMBIGUITY_MODEL_DIR = _project_root() / "models" / "ambiguity_distilbert"
DEFAULT_ACTOR_MODEL_DIR = _project_root() / "models" / "actor_distilbert"


class AmbiguityResolver:

    def __init__(
        self,
        ambiguity_model_dir: Optional[Path] = None,
        actor_model_dir: Optional[Path] = None,
    ):
        self.ambiguity_model_dir = ambiguity_model_dir or DEFAULT_AMBIGUITY_MODEL_DIR
        self.actor_model_dir = actor_model_dir or DEFAULT_ACTOR_MODEL_DIR
        self._ambiguity_pipeline = None
        self._actor_pipeline = None

    @staticmethod
    def _trained_model_path(model_dir: Path, role: str) -> Path:
        resolved = model_dir.resolve()
        if not resolved.is_dir() or not (resolved / "config.json").is_file():
            raise FileNotFoundError(
                f"{role} not found at {resolved}. Train first: "
                f"PYTHONPATH=src python -m phase1.classifier (and phase1.localizer)."
            )
        return resolved

    def _ambiguity_classifier(self):
        if self._ambiguity_pipeline is None:
            path = self._trained_model_path(self.ambiguity_model_dir, "Ambiguity classifier")
            key = str(path)
            self._ambiguity_pipeline = pipeline(
                "text-classification",
                model=key,
                tokenizer=key,
                device=-1,
            )
        return self._ambiguity_pipeline

    def _actor_tagger(self):
        if self._actor_pipeline is None:
            path = self._trained_model_path(self.actor_model_dir, "Actor localizer")
            key = str(path)
            self._actor_pipeline = pipeline(
                "token-classification",
                model=key,
                tokenizer=key,
                aggregation_strategy="simple",
                device=-1,
            )
        return self._actor_pipeline

    def is_ambiguous(self, sentence: str) -> bool:
        pipe = self._ambiguity_classifier()
        result = pipe(sentence)[0]
        label = result["label"]
        id2label = pipe.model.config.id2label
        if label.startswith("LABEL_"):
            idx = int(label.replace("LABEL_", ""))
            label = id2label.get(idx, id2label[str(idx)])
        return str(label).lower() == "ambiguous"

    def locate_occupation(self, sentence: str) -> Optional[Tuple[int, int]]:
        pipe = self._actor_tagger()
        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for entity in pipe(sentence):
            group = entity.get("entity_group") or entity.get("entity") or entity.get("label") or ""
            if "actor_span" not in str(group).lower():
                continue
            score = float(entity.get("score", 0.0))
            if score > best_score:
                best_score = score
                best = entity
        if best is None:
            return None
        return int(best["start"]), int(best["end"])

    def inject(self, sentence: str, span: Tuple[int, int], gender: str) -> str:
        start, end = span
        occupation = sentence[start:end]
        replacement = f"{gender} {occupation}"
        return sentence[:start] + replacement + sentence[end:]

    def resolve(self, sentence: str) -> List[str]:
        if not self.is_ambiguous(sentence):
            return [sentence]

        span = self.locate_occupation(sentence)
        if span is None:
            return [sentence]

        return [
            self.inject(sentence, span, "male"),
            self.inject(sentence, span, "female"),
        ]
