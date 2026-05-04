# Bias-Aware Translation: Generating Gender-Consistent Alternatives for Ambiguous Inputs

## Overview

Machine translation systems often introduce **gender bias** when translating from gender-neutral languages or sentences into gendered languages. Even when gender is not specified in the source sentence, models frequently default to a single gendered interpretation.

This project introduces a **bias-aware translation pipeline** that:
- Detects gender ambiguity in sentences
- Identifies the occupational **actor** span in the text
- For ambiguous inputs, injects `male` / `female` before the actor and translates both variants
- For unambiguous inputs, produces a **single** translation
- Exposes how machine translation can differ under explicit gender disambiguation

---

## Key idea

Instead of producing a single translation, our system generates an un-biased translation containing:

- A **masculine version** of the input
- A **feminine version** of the input 

for gender-ambiguous occupational sentences. Note, for feasibility, we constrain our use-case for sentences containing a single occupational term and a single gendered entity. This constraint avoids combinatorial growth in possible gender–occupation pairings and ensures controlled, interpretable evaluation of translation outputs.

---

## System architecture

```
Input sentence (English)
    ↓
Phase 1: Ambiguity classification + actor span tagging (DistilBERT, fine-tuned)
    ↓
If ambiguous: inject "male" / "female" before the actor → two source strings
If unambiguous: one source string
    ↓
Phase 2: English → French (Google Translate via `deep-translator`)
    ↓
One or two French outputs (labeled in the CLI)
```

Phase 3 (evaluation / batch logging) exists in the repo but is **not** used by the default command-line program.

---

## Models

Phase 1 uses **two small DistilBERT models** trained on `data/gender_dataset.csv`: one decides whether the occupational actor is **gender-ambiguous** in English, and one finds **where the actor is** in the sentence so the pipeline can insert `male` / `female` before translation.

Weights are produced by the training scripts and stored under **`models/ambiguity_distilbert/`** and **`models/actor_distilbert/`**. Those folders are not tracked in git—you generate them locally by training (see below).

---

## Project structure

```
Bias-Aware-Translation/
├── data/
│   └── gender_dataset.csv    # Training/eval data (sentences, ambiguity, actor spans)
├── models/                    # Created by training; not in git (see .gitignore)
│   ├── ambiguity_distilbert/
│   └── actor_distilbert/
├── src/
│   ├── main.py                # Interactive CLI (entry point)
│   ├── phase1/
│   │   ├── classifier.py    # Train ambiguity model
│   │   ├── localizer.py     # Train actor-span model
│   │   └── resolver.py      # Load models, resolve, inject
│   ├── phase2/
│   │   └── translator.py     # Google Translate wrapper
│   ├── phase3/
│   │   └── evaluator.py      # Evaluation and batch logging tools
│   └── pipeline/
│       └── pipeline.py       # Pipeline logic: drives resolver and translator
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3 -m venv venv_nlp
source venv_nlp/bin/activate
pip install -r requirements.txt
```

Translation uses **Google’s service** through `deep-translator`; you need a **network connection** when running the app.

---

## Training the models

From the **repository root** (the folder that contains `data/` and `src/`), with `data/gender_dataset.csv` in place:

```bash
# Ambiguity classifier → models/ambiguity_distilbert/
PYTHONPATH=src python3 -m phase1.classifier

# Actor localizer → models/actor_distilbert/
PYTHONPATH=src python3 -m phase1.localizer
```

Each command fine-tunes on CPU (as configured), prints validation metrics, and saves the best checkpoint to the corresponding `models/...` directory. Training hyperparameters (epochs, batch size, learning rate, max sequence length) are set as constants at the top of `classifier.py` and `localizer.py`.

---

## Running the program end to end

1. **Train** both models (or ensure `models/ambiguity_distilbert/config.json` and `models/actor_distilbert/config.json` exist).
2. From the **repository root**:

```bash
PYTHONPATH=src python3 src/main.py
```

3. Enter English sentences at the prompt. Type **`exit`** to quit.

- If the sentence is classified as **unambiguous**, you get **one** French translation.
- If **ambiguous**, the resolver produces **two** English variants (with `male` / `female` injected) and you get **two** French lines (masculine / feminine versions in the UI labels).

If a model directory is missing, the program raises a clear `FileNotFoundError` pointing you back to the training commands above.