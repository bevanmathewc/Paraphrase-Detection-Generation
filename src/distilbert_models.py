import numpy as np

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

def load_tokenizer(
    model_name="distilbert-base-uncased"
):
    """
    Load the Hugging Face tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    return tokenizer

def load_model(
    model_name="distilbert-base-uncased",
    num_labels=2
):
    """
    Load DistilBERT for sequence classification.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    return model

def tokenize_dataset(
    dataset,
    tokenizer,
    text1_column,
    text2_column,
    max_length=128
):
    """
    Tokenize sentence pairs.
    """

    return tokenizer(
        dataset[text1_column],
        dataset[text2_column],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

def dataframe_to_dataset(df):
    """
    Convert pandas DataFrame
    to HuggingFace Dataset.
    """

    return Dataset.from_pandas(
        df,
        preserve_index=False
    )

def compute_metrics(eval_pred):
    """
    Metrics used by Trainer.
    """

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    import torch

    probabilities = (
        torch.softmax(
            torch.tensor(logits),
            dim=1
        )[:, 1]
        .numpy()
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

def create_training_arguments(
    output_dir
):
    """
    Create HuggingFace
    TrainingArguments.
    """

    return TrainingArguments(

        output_dir=output_dir,

        learning_rate=2e-5,

        per_device_train_batch_size=16,

        per_device_eval_batch_size=16,

        num_train_epochs=3,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        logging_steps=100,

        report_to="none"
    )

def create_trainer(
    model,
    args,
    train_dataset,
    validation_dataset
):
    """
    Create Trainer.
    """

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        compute_metrics=compute_metrics

    )

    return trainer

def save_transformer(
    trainer,
    path
):
    """
    Save model and tokenizer.
    """

    trainer.save_model(path)

def load_saved_model(
    path
):
    """
    Load a saved DistilBERT model.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        path
    )

def load_saved_tokenizer(path):
    """
    Load a saved tokenizer.
    """

    return AutoTokenizer.from_pretrained(path)



