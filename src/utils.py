from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import time
import pandas as pd
import random
import numpy as np
import torch
from openpyxl import load_workbook

def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def dataset_info(dataset, name):
    print("="*70)
    print(name)
    for split in dataset.keys():
        print(split)
        print(len(dataset[split]))
        print(dataset[split].column_names)
        print()

def label_distribution(dataset):
    labels = dataset["label"]
    return Counter(labels)

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self,*args):
        self.end = time.time()
        self.elapsed = self.end-self.start
        print(f"Elapsed: {self.elapsed:.2f} sec")


def save_metrics(metrics, filename):
    df = pd.DataFrame([metrics])
    df.to_csv(filename,index=False)

def save_predictions(df,filename):
    df.to_csv(filename,index=False)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_device():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

def print_model_summary(model):
    print(model)
    total = sum(
        p.numel()
        for p in model.parameters()
    )
    print(f"Parameters: {total:,}")

def log_experiment(
    excel_path,
    row
):
    wb = load_workbook(excel_path)
    ws = wb.active
    ws.append(row)
    wb.save(excel_path)