# Bias-Aware Translation: Generating Gender-Consistent Alternatives for Ambiguous Inputs

## Overview

Machine translation systems often introduce **gender bias** when translating from gender-neutral languages or sentences into gendered languages. Even when gender is not specified in the source sentence, models frequently default to a single gendered interpretation.

This project introduces a **bias-aware translation pipeline** that:
- Detects gender ambiguity in sentences
- Identifies occupational terms
- Generates both masculine and feminine variants
- Translates both versions independently
- Exposes hidden gender bias in machine translation systems

---

## Key Idea

Instead of producing a single translation, our system generates an un-biased translation containing:

- A **masculine version** of the input
- A **feminine version** of the input 

for gender-ambiguous occupational sentences. Note, for feasibility, we constrain our use-case for sentences containing a single occupational term and a single gendered entity. This constraint avoids combinatorial growth in possible gender–occupation pairings and ensures controlled, interpretable evaluation of translation outputs.

---

## System Architecture
```
Input Sentence
↓
Phase 1: Ambiguity Detection + Occupation Localization
↓
Gender Injection (if ambiguous)
↓
Phase 2: Machine Translation (Google Translate backend)
↓
Phase 3: Evaluation + Logging
```

---

## Project Structure

---

## 🚀 Getting Started

### 1. Setup environment

```
python3 -m venv venv_nlp
source venv_nlp/bin/activate  # (Mac/Linux)
pip install -r requirements.txt
```

### 2. Run DEMO mode or Batched Experiments

```
python src/main.py --mode demo
```
or
```
python src/main.py --mode experiment --input_file data/eval_inputs.txt
```

## DEMO Example


