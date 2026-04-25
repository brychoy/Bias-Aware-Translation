from typing import List, Optional, Tuple


class AmbiguityResolver:
    """
    Phase 1 wrapper:
    - detects ambiguity (mock logic for now / placeholder)
    - finds occupation span
    - injects gendered variants
    """

    def __init__(self):
        self.occupations = ["professor", "doctor", "nurse", "engineer", "teacher"]

        # gender indicators (simple heuristic baseline)
        self.gender_markers = ["male", "female", "man", "woman", "he", "she"]

    # mock classifier
    def is_ambiguous(self, sentence: str) -> bool:
        lower = sentence.lower()
        has_occupation = any(o in lower for o in self.occupations)
        has_gender = any(g in lower for g in self.gender_markers)
        return has_occupation and not has_gender

    # mock localizer
    def locate_occupation(self, sentence: str) -> Optional[Tuple[int, int]]:
        lower = sentence.lower()

        for occ in self.occupations:
            if occ in lower:
                start = lower.index(occ)
                end = start + len(occ)
                return (start, end)

        return None

    # gender injection
    def inject(self, sentence: str, span: Tuple[int, int], gender: str) -> str:
        start, end = span
        occupation = sentence[start:end]

        replacement = f"{gender} {occupation}"

        return sentence[:start] + replacement + sentence[end:]

    # main interface
    def resolve(self, sentence: str) -> List[str]:

        # 1. if not ambiguous -> return as-is
        if not self.is_ambiguous(sentence):
            return [sentence]

        span = self.locate_occupation(sentence)

        if span is None:
            return [sentence]

        return [
            self.inject(sentence, span, "male"),
            self.inject(sentence, span, "female"),
        ]
