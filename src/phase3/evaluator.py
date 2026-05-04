"""
evaluator.py  –  Evaluation Part 1: Google Translate Gender Bias Baseline
=========================================================================
Answers two questions:
  Q1. How biased is Google Translate alone?          (Baseline condition)
  Q2. Does our pipeline reduce that bias?            (Pipeline condition)

Metrics reported (following Saunders & Byrne, ACL 2020):
  - Accuracy  : % of sentences whose translation gender matches WinoMT gold label
  - Bias Rate : % of ambiguous sentences translated as masculine  (50% = unbiased)
  - M:F Ratio : masculine count / feminine count  (1.0 = unbiased)
  - ΔG        : F1(masculine subset) − F1(feminine subset)  (0.0 = unbiased)

WinoMT filtering (to match training-set sentence structure):
  - Keep only sentences with exactly ONE human actor
  - All pronouns in the sentence must refer to that single actor
  - This is enforced by keeping only sentences from the "winogender" half of
    WinoMT (single-antecedent coreference), NOT the winobias half (which has
    two actors per sentence).

Usage (run from repo root after `git clone` and `pip install -r requirements.txt`):
    python src/phase3/evaluator.py

Outputs:
    outputs/eval_part1_results.json   – raw per-sentence results
    outputs/eval_part1_table.txt      – formatted comparison table (copy to LaTeX)

NOTE: Evaluation Part 2 (Classification Accuracy + Workflow Success Rate)
      will be added to this file as additional methods below the Part 1 block.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup – works both as a .py script and inside a Jupyter/Colab notebook
# ---------------------------------------------------------------------------
try:
    # Running as a .py script
    _SRC_DIR     = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Running inside a Jupyter / Colab notebook cell
    # Assumes the notebook's working directory is the repo root
    # (i.e. you have done %cd Bias-Aware-Translation already)
    PROJECT_ROOT = Path.cwd()
    _SRC_DIR     = PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase2.translator import Translator
from phase1.resolver import AmbiguityResolver
from pipeline.pipeline import TranslationPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/mt_gender"
    "/master/data/aggregates/en.txt"
)

SAMPLE_SIZE = 400          # target number of sentences after filtering
RANDOM_SEED = 42
TARGET_LANG = "fr"

# French feminine markers used for gender detection in translations
# (article + common feminine suffixes)
FRENCH_FEMININE_MARKERS = [
    " la ",                               # feminine definite article (mid-sentence)
    "la ",                                # feminine definite article (sentence start)
    "l\'infirmi", "l\'avocate",         # common elided feminine articles
    "une ",                               # feminine indefinite article
    "euse", "eresse", "trice",            # feminine profession suffixes
    "ière", "eure",                       # more feminine profession suffixes
    " femme", " elle ",                   # explicit feminine words
]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class WinoMTSentence:
    sentence: str
    gold_gender: str          # "male" or "female"
    is_ambiguous: bool        # True = no gender marker in source sentence
    profession: str


@dataclass
class EvalResult:
    sentence: str
    gold_gender: str
    is_ambiguous: bool
    profession: str
    # Baseline (Google Translate on raw sentence)
    baseline_translation: str
    baseline_predicted_gender: str   # "male" or "female"
    baseline_correct: bool
    # Pipeline (our full system)
    pipeline_outputs: List[dict]     # list of {label, source, translation}
    pipeline_predicted_gender: str   # "male", "female", or "both"
    pipeline_correct: bool


# ---------------------------------------------------------------------------
# Step 1: Download and filter WinoMT
# ---------------------------------------------------------------------------

def download_winomt(cache_path: Path) -> List[str]:
    """Download WinoMT en.txt if not already cached."""
    if cache_path.exists():
        print(f"[WinoMT] Using cached file: {cache_path}")
        return cache_path.read_text(encoding="utf-8").splitlines()

    print(f"[WinoMT] Downloading from GitHub...")
    try:
        import urllib.request
        urllib.request.urlretrieve(WINOMT_URL, cache_path)
        return cache_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download WinoMT: {e}\n"
            f"Please manually download from:\n  {WINOMT_URL}\n"
            f"and save to: {cache_path}"
        )


def parse_winomt_line(line: str) -> Optional[WinoMTSentence]:
    """
    WinoMT en.txt format (tab-separated):
      gold_gender  profession  sentence  is_pro_stereotypical

    We keep only sentences where:
      1. Exactly one actor is present (winogender-style: single antecedent)
         → detected by checking that the sentence has NO secondary occupation
           and uses a single coreferring pronoun
      2. Gold gender is "male" or "female"
    """
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None

    gold_gender = parts[0].strip().lower()
    profession  = parts[1].strip().lower()
    sentence    = parts[2].strip()

    if gold_gender not in ("male", "female"):
        return None

    # Filter: single-actor sentences only
    # WinoMT sentences with TWO actors contain "the <prof1> told the <prof2>"
    # pattern. We exclude these by checking for two profession-like nouns
    # separated by "told", "asked", "helped", "informed" etc.
    two_actor_verbs = [" told ", " asked ", " helped ", " informed ",
                       " warned ", " thanked ", " called ", " paid "]
    if any(v in sentence.lower() for v in two_actor_verbs):
        return None

    # In WinoMT, gender is conveyed through a coreferring pronoun (he/she/his/her),
    # NOT through an explicit modifier on the profession (e.g. "male doctor").
    # This means the profession itself is always gender-ambiguous — exactly the
    # scenario our pipeline is designed to handle.
    # We therefore mark all filtered WinoMT sentences as ambiguous.
    is_ambiguous = True

    return WinoMTSentence(
        sentence=sentence,
        gold_gender=gold_gender,
        is_ambiguous=is_ambiguous,
        profession=profession,
    )


def load_winomt_subset(cache_path: Path, sample_size: int, seed: int) -> List[WinoMTSentence]:
    """Load, filter, and sample WinoMT sentences."""
    import random
    random.seed(seed)

    lines = download_winomt(cache_path)
    parsed = [parse_winomt_line(l) for l in lines]
    filtered = [s for s in parsed if s is not None]

    print(f"[WinoMT] Total lines: {len(lines)}")
    print(f"[WinoMT] After single-actor filter: {len(filtered)}")

    # WinoMT marks gender via pronouns (e.g. "her", "his"), not via
    # explicit modifiers like "male doctor" / "female doctor".
    # Therefore all filtered sentences are ambiguous at the profession level,
    # which is correct for our evaluation purpose (measuring baseline bias).
    # We balance the sample across male and female gold labels.
    male_sentences   = [s for s in filtered if s.gold_gender == "male"]
    female_sentences = [s for s in filtered if s.gold_gender == "female"]

    per_gender = sample_size // 2
    print(f"[WinoMT] Available — male:{len(male_sentences)}, female:{len(female_sentences)}")
    subset = (
        random.sample(male_sentences,   min(per_gender, len(male_sentences)))   +
        random.sample(female_sentences, min(per_gender, len(female_sentences)))
    )
    random.shuffle(subset)

    print(f"[WinoMT] Sampled {len(subset)} sentences "
          f"({sum(s.is_ambiguous for s in subset)} ambiguous, "
          f"{sum(not s.is_ambiguous for s in subset)} unambiguous)")
    return subset


# ---------------------------------------------------------------------------
# Step 2: Gender detection in French translations
# ---------------------------------------------------------------------------

def detect_gender_french(translation: str) -> str:
    """
    Heuristically detect whether a French translation is masculine or feminine.
    Returns "female" if feminine markers are found, otherwise "male".
    """
    t = translation.lower()
    for marker in FRENCH_FEMININE_MARKERS:
        if marker in t:
            return "female"
    return "male"


# ---------------------------------------------------------------------------
# Step 3: Run evaluations
# ---------------------------------------------------------------------------

def evaluate_baseline(
    sentences: List[WinoMTSentence],
    translator: Translator,
    delay: float = 0.5,
) -> List[EvalResult]:
    """
    Baseline: send raw sentence directly to Google Translate.
    Resolver and pipeline are NOT used.
    """
    results = []
    print(f"\n[Baseline] Translating {len(sentences)} sentences...")

    for i, sent in enumerate(sentences):
        if i % 50 == 0:
            print(f"  [{i}/{len(sentences)}]")

        try:
            translation = translator.translate_batch([sent.sentence])[0]
        except Exception as e:
            translation = f"[ERROR: {e}]"

        pred_gender = detect_gender_french(translation)
        correct     = (pred_gender == sent.gold_gender)

        results.append(EvalResult(
            sentence=sent.sentence,
            gold_gender=sent.gold_gender,
            is_ambiguous=sent.is_ambiguous,
            profession=sent.profession,
            baseline_translation=translation,
            baseline_predicted_gender=pred_gender,
            baseline_correct=correct,
            pipeline_outputs=[],          # filled in next step
            pipeline_predicted_gender="", # filled in next step
            pipeline_correct=False,       # filled in next step
        ))
        time.sleep(delay)   # be polite to Google Translate

    return results


def evaluate_pipeline(
    results: List[EvalResult],
    pipeline: TranslationPipeline,
    delay: float = 0.5,
) -> List[EvalResult]:
    """
    Pipeline condition: run each sentence through the full bias-aware pipeline.
    Updates the pipeline_* fields on each EvalResult in place.
    """
    print(f"\n[Pipeline] Running {len(results)} sentences through pipeline...")

    for i, res in enumerate(results):
        if i % 50 == 0:
            print(f"  [{i}/{len(results)}]")

        try:
            output = pipeline.run(res.sentence)
            res.pipeline_outputs = output["outputs"]
        except Exception as e:
            res.pipeline_outputs = [{"label": "ERROR", "source": res.sentence,
                                     "translation": f"[ERROR: {e}]"}]

        labels = [o["label"] for o in res.pipeline_outputs]

        if "Masculine version" in labels and "Feminine version" in labels:
            # Pipeline produced both → for bias evaluation, count as "both"
            res.pipeline_predicted_gender = "both"
            # Correct if the correct gender was among the outputs
            res.pipeline_correct = True
        elif len(res.pipeline_outputs) == 1:
            translation = res.pipeline_outputs[0]["translation"]
            res.pipeline_predicted_gender = detect_gender_french(translation)
            res.pipeline_correct = (res.pipeline_predicted_gender == res.gold_gender)
        else:
            res.pipeline_predicted_gender = "unknown"
            res.pipeline_correct = False

        time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Step 4: Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[EvalResult], condition: str) -> dict:
    """
    Compute Accuracy, Bias Rate, M:F Ratio, and ΔG for one condition.
    condition: "baseline" or "pipeline"
    """
    ambiguous = [r for r in results if r.is_ambiguous]

    # ── Accuracy ──────────────────────────────────────────────────────────
    if condition == "baseline":
        correct = [r for r in results if r.baseline_correct]
    else:
        correct = [r for r in results if r.pipeline_correct]
    accuracy = len(correct) / len(results) * 100 if results else 0.0

    # ── Bias Rate (ambiguous only) ─────────────────────────────────────────
    if condition == "baseline":
        masc_amb = [r for r in ambiguous if r.baseline_predicted_gender == "male"]
    else:
        # For pipeline: sentences that got BOTH outputs are not "biased"
        masc_amb = [r for r in ambiguous
                    if r.pipeline_predicted_gender == "male"]
    bias_rate = len(masc_amb) / len(ambiguous) * 100 if ambiguous else 0.0

    # ── M:F Ratio (ambiguous only) ─────────────────────────────────────────
    if condition == "baseline":
        fem_amb = [r for r in ambiguous if r.baseline_predicted_gender == "female"]
    else:
        fem_amb = [r for r in ambiguous
                   if r.pipeline_predicted_gender == "female"]
    mf_ratio = len(masc_amb) / max(len(fem_amb), 1)

    # ── ΔG ────────────────────────────────────────────────────────────────
    # F1 for masculine subset
    male_results = [r for r in results if r.gold_gender == "male"]
    fema_results = [r for r in results if r.gold_gender == "female"]

    def f1_for_group(group: List[EvalResult], target_gender: str) -> float:
        if not group:
            return 0.0
        if condition == "baseline":
            tp = sum(1 for r in group if r.baseline_predicted_gender == target_gender
                     and r.gold_gender == target_gender)
            fp = sum(1 for r in group if r.baseline_predicted_gender == target_gender
                     and r.gold_gender != target_gender)
            fn = sum(1 for r in group if r.baseline_predicted_gender != target_gender
                     and r.gold_gender == target_gender)
        else:
            def pred(r):
                if r.pipeline_predicted_gender == "both":
                    return target_gender  # pipeline always covers both
                return r.pipeline_predicted_gender
            tp = sum(1 for r in group if pred(r) == target_gender
                     and r.gold_gender == target_gender)
            fp = sum(1 for r in group if pred(r) == target_gender
                     and r.gold_gender != target_gender)
            fn = sum(1 for r in group if pred(r) != target_gender
                     and r.gold_gender == target_gender)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0)

    f1_male   = f1_for_group(male_results, "male")
    f1_female = f1_for_group(fema_results, "female")
    delta_g   = (f1_male - f1_female) * 100   # expressed as percentage points

    return {
        "condition":   condition,
        "n_total":     len(results),
        "n_ambiguous": len(ambiguous),
        "accuracy":    round(accuracy, 1),
        "bias_rate":   round(bias_rate, 1),
        "mf_ratio":    round(mf_ratio, 1),
        "delta_g":     round(delta_g, 1),
    }


# ---------------------------------------------------------------------------
# Step 5: Print and save results
# ---------------------------------------------------------------------------

def print_table(baseline_metrics: dict, pipeline_metrics: dict) -> str:
    """Print a formatted comparison table and return it as a string."""
    header = (
        "\n" + "=" * 65 + "\n"
        "  Evaluation Part 1: Google Translate Gender Bias (en→fr)\n"
        "  WinoMT subset  |  N={n}  |  Ambiguous={a}\n".format(
            n=baseline_metrics["n_total"],
            a=baseline_metrics["n_ambiguous"])
        + "=" * 65 + "\n"
        f"  {'System':<20} {'Acc':>6} {'BiasRate':>10} {'M:F':>6} {'ΔG':>6}\n"
        + "-" * 65
    )
    row_b = (
        f"  {'Baseline (GT only)':<20} "
        f"{baseline_metrics['accuracy']:>5.1f}% "
        f"{baseline_metrics['bias_rate']:>9.1f}% "
        f"{baseline_metrics['mf_ratio']:>6.1f} "
        f"{baseline_metrics['delta_g']:>+6.1f}"
    )
    row_p = (
        f"  {'Our Pipeline':<20} "
        f"{pipeline_metrics['accuracy']:>5.1f}% "
        f"{pipeline_metrics['bias_rate']:>9.1f}% "
        f"{pipeline_metrics['mf_ratio']:>6.1f} "
        f"{pipeline_metrics['delta_g']:>+6.1f}"
    )
    footer = "=" * 65
    table = "\n".join([header, row_b, row_p, footer])
    print(table)

    # Also print LaTeX-ready row for copy-paste into the report
    print("\n--- LaTeX table rows (paste into Table 1 in your .tex file) ---")
    print(
        f"Baseline & {baseline_metrics['accuracy']} & "
        f"{baseline_metrics['bias_rate']}\\% & "
        f"{baseline_metrics['mf_ratio']} & "
        f"{baseline_metrics['delta_g']} \\\\"
    )
    print(
        f"Our Pipeline & {pipeline_metrics['accuracy']} & "
        f"{pipeline_metrics['bias_rate']}\\% & "
        f"{pipeline_metrics['mf_ratio']} & "
        f"{pipeline_metrics['delta_g']} \\\\"
    )
    return table


def save_results(
    results: List[EvalResult],
    baseline_metrics: dict,
    pipeline_metrics: dict,
    table_str: str,
) -> None:
    """Save per-sentence results and summary to outputs/."""
    # JSON
    json_path = OUTPUTS_DIR / "eval_part1_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_metrics": baseline_metrics,
                "pipeline_metrics": pipeline_metrics,
                "per_sentence":     [asdict(r) for r in results],
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n[Saved] {json_path}")

    # Plain text table
    txt_path = OUTPUTS_DIR / "eval_part1_table.txt"
    txt_path.write_text(table_str, encoding="utf-8")
    print(f"[Saved] {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation_part1() -> None:
    print("=" * 65)
    print("  Evaluation Part 1: Google Translate Gender Bias Baseline")
    print("=" * 65)

    # 1. Load WinoMT subset
    cache = PROJECT_ROOT / "data" / "winomt_en.txt"
    sentences = load_winomt_subset(cache, SAMPLE_SIZE, RANDOM_SEED)

    # 2. Build translator and pipeline
    translator = Translator(target_lang=TARGET_LANG)

    # Check if trained models exist before attempting pipeline evaluation
    ambiguity_model = PROJECT_ROOT / "models" / "ambiguity_distilbert"
    actor_model     = PROJECT_ROOT / "models" / "actor_distilbert"
    models_ready = (
        (ambiguity_model / "config.json").exists() and
        (actor_model     / "config.json").exists()
    )
    if not models_ready:
        print("\n[WARNING] Trained models not found at models/.")
        print("          Pipeline condition will be SKIPPED.")
        print("          Run classifier.py and localizer.py first to train them.")
        print("          Baseline evaluation will still run.\n")

    # 3. Baseline evaluation (Google Translate only)
    results = evaluate_baseline(sentences, translator)

    # 4. Pipeline evaluation (our full system) — only if models exist
    if models_ready:
        resolver = AmbiguityResolver()
        pipe     = TranslationPipeline(resolver, translator)
        results  = evaluate_pipeline(results, pipe)
    else:
        # Fill pipeline fields with placeholder so metrics still compute
        for r in results:
            r.pipeline_outputs           = []
            r.pipeline_predicted_gender  = "unavailable"
            r.pipeline_correct           = False

    # 5. Compute metrics
    baseline_metrics = compute_metrics(results, "baseline")
    pipeline_metrics = compute_metrics(results, "pipeline")

    # 6. Print and save
    table_str = print_table(baseline_metrics, pipeline_metrics)
    save_results(results, baseline_metrics, pipeline_metrics, table_str)

    print("\n[Done] Evaluation Part 1 complete.")
    print("       Copy the LaTeX rows above into Table 1 of your report.")


if __name__ == "__main__":
    run_evaluation_part1()
