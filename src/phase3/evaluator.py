import json


class Evaluator:
    def __init__(self):
        self.results = []

    def log(self, result: dict):
        self.results.append(result)

    def compute_summary(self):
        total = len(self.results)
        ambiguous = sum(1 for r in self.results if len(r["resolved"]) == 2)

        return {
            "total_inputs": total,
            "ambiguous_cases_detected": ambiguous,
            "ambiguity_rate": ambiguous / total if total else 0,
        }

    def save(self, path="outputs/results.json"):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
