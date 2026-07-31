import os
import random
import numpy as np
import torch

# ==========================================================
# Random Seeds (Reproducibility)
# ==========================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

NUM_EPOCHS = 3

MAX_SEQUENCE_LENGTH = 128

# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"