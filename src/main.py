from phase1.resolver import AmbiguityResolver
from phase2.translator import Translator
from pipeline.pipeline import TranslationPipeline


def build_pipeline() -> TranslationPipeline:
    resolver = AmbiguityResolver()
    translator = Translator(target_lang="fr")
    return TranslationPipeline(resolver, translator)


def main() -> None:
    pipeline = build_pipeline()

    print("\nBias-aware English to French translation\n")

    while True:
        sentence = input("Enter an English sentence (or 'exit'): ").strip()
        if sentence.lower() == "exit":
            break
        if not sentence:
            continue

        result = pipeline.run(sentence)

        print("\n--- Results ---")
        print("Input:", result["input"])

        for item in result["outputs"]:
            print(f"\n[{item['label']}]")
            print("Source:", item["source"])
            print("French:", item["translation"])

        print("---------------\n")


if __name__ == "__main__":
    main()
