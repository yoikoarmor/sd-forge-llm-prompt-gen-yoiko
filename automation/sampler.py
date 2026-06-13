"""Random Gen Prompt sampling from a category word pool."""

import json
import random
from pathlib import Path

# Join order for sampled parts. Categories absent from the pool are skipped.
CATEGORY_ORDER = [
    "subject",
    "setting",
    "lighting",
    "style",
    "mood",
    "composition",
    "extra",
]


class GenPromptSampler:
    def __init__(self, wordpool_path, seed=None):
        with open(wordpool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        self.required = pool.get("required", {})
        self.optional = pool.get("optional", {})
        self.rng = random.Random(seed)

    def sample(self):
        """Returns (gen_prompt, parts) where parts maps category -> chosen item."""
        parts = {}
        for category, items in self.required.items():
            parts[category] = self.rng.choice(items)
        for category, spec in self.optional.items():
            if self.rng.random() < spec.get("probability", 0.5):
                parts[category] = self.rng.choice(spec["items"])

        ordered = [parts[c] for c in CATEGORY_ORDER if c in parts]
        # Any category not in CATEGORY_ORDER still gets appended at the end.
        ordered += [v for k, v in parts.items() if k not in CATEGORY_ORDER]
        return ", ".join(ordered), parts


if __name__ == "__main__":
    pool_path = Path(__file__).resolve().parent / "wordpool.json"
    sampler = GenPromptSampler(pool_path)
    for _ in range(10):
        prompt, _ = sampler.sample()
        print(prompt)
