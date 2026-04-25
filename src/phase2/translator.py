from typing import List
from deep_translator import GoogleTranslator


class Translator:
    def __init__(self, target_lang: str = "fr"):
        self.target_lang = target_lang
        self.translator = GoogleTranslator(source="auto", target=target_lang)

    def translate_batch(self, sentences: List[str]) -> List[str]:
        outputs = []

        for s in sentences:
            try:
                outputs.append(self.translator.translate(s))
            except Exception as e:
                outputs.append(f"[TRANSLATION_ERROR]: {str(e)}")

        return outputs
