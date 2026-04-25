import argparse
from tqdm import tqdm

from phase1.resolver import AmbiguityResolver
from phase2.translator import Translator
from phase3.evaluator import Evaluator
from pipeline.pipeline import TranslationPipeline


def build_pipeline():
    resolver = AmbiguityResolver()
    translator = Translator(target_lang="fr")

    return TranslationPipeline(resolver, translator)


def run_demo():
    pipeline = build_pipeline()

    print("\nBias-Aware Translation Demo\n")

    while True:
        sentence = input("Enter sentence (or 'exit'): ")
        if sentence == "exit":
            break

        result = pipeline.run(sentence)

        print("\n--- RESULT ---")
        print("Input:", result["input"])

        for item in result["outputs"]:
            print(f"\n[{item['label']}]")
            print("Source:", item["source"])
            print("Translation:", item["translation"])

        print("--------------")


def run_experiment(file_path: str):
    pipeline = build_pipeline()
    evaluator = Evaluator()

    with open(file_path) as f:
        sentences = [l.strip() for l in f if l.strip()]

    for s in tqdm(sentences):
        result = pipeline.run(s)
        evaluator.log(result)

    evaluator.save()

    print("\n--- SUMMARY ---")
    print(evaluator.compute_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "experiment"], default="demo")
    parser.add_argument("--input_file", default="data/eval_inputs.txt")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        run_experiment(args.input_file)
