from sentence_transformers import SentenceTransformer
import numpy as np
import joblib


def load_sbert_model(model_name="all-MiniLM-L6-v2"):
    """
    Load a pretrained Sentence-BERT model.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(model, sentences, batch_size=32):
    """
    Generate sentence embeddings.

    Parameters
    ----------
    model : SentenceTransformer
    sentences : iterable of strings
    batch_size : int

    Returns
    -------
    numpy.ndarray
        Shape = (num_sentences, embedding_dim)
    """

    embeddings = model.encode(
        sentences.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def concatenate_embeddings(embeddings1, embeddings2):
    """
    Concatenate two embedding matrices.

    Output shape:
    (N, 768)
    """

    return np.concatenate(
        [embeddings1, embeddings2],
        axis=1
    )


def save_embeddings(embeddings, path):
    """
    Save embeddings.
    """
    joblib.dump(embeddings, path)


def load_embeddings(path):
    """
    Load saved embeddings.
    """
    return joblib.load(path)