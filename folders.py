from pathlib import Path

folders = [
    # DistilBERT
    "models/distilbert/mrpc",
    "models/distilbert/qqp",
    "models/distilbert/paws",

    "checkpoints/distilbert/mrpc",
    "checkpoints/distilbert/qqp",
    "checkpoints/distilbert/paws",

    # T5 (future work)
    "models/t5/mrpc",
    "models/t5/qqp",
    "models/t5/paws",

    "checkpoints/t5/mrpc",
    "checkpoints/t5/qqp",
    "checkpoints/t5/paws",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("Project folder structure initialized.")