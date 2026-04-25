from typing import Dict, List


class TranslationPipeline:
    def __init__(self, resolver, translator):
        self.resolver = resolver
        self.translator = translator

    def run(self, sentence: str) -> Dict:

        resolved: List[str] = self.resolver.resolve(sentence)
        translations: List[str] = self.translator.translate_batch(resolved)

        labeled_outputs = []

        if len(resolved) == 1:
            labeled_outputs.append(
                {
                    "label": "Unambiguous",
                    "source": resolved[0],
                    "translation": translations[0],
                }
            )
        else:
            labeled_outputs.append(
                {
                    "label": "Masculine version",
                    "source": resolved[0],
                    "translation": translations[0],
                }
            )

            labeled_outputs.append(
                {
                    "label": "Feminine version",
                    "source": resolved[1],
                    "translation": translations[1],
                }
            )

        return {"input": sentence, "outputs": labeled_outputs}
