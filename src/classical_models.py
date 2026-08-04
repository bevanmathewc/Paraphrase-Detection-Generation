from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


def combine_sentences(df, col1="sentence1", col2="sentence2"):
    """
    Combine two text columns into a single input string.
    """
    return df[col1] + " [SEP] " + df[col2]


def create_tfidf_vectorizer(
    max_features=10000,
    ngram_range=(1, 2)
):
    """
    Create a TF-IDF vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )


def vectorize_data(vectorizer, X_train, X_validation, X_test=None):
    """
    Fit on the training set and transform the remaining splits.
    """
    X_train_vector = vectorizer.fit_transform(X_train)

    X_validation_vector = vectorizer.transform(X_validation)

    X_test_vector = None

    if X_test is not None:
        X_test_vector = vectorizer.transform(X_test)

    return (
        X_train_vector,
        X_validation_vector,
        X_test_vector
    )


def train_logistic_regression(
    X_train,
    y_train,
    random_state=42,
    max_iter=1000
):
    """
    Train a Logistic Regression classifier.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter
    )

    model.fit(X_train, y_train)

    return model


def predict(model, X):
    """
    Generate predictions and probabilities.
    """
    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


def save_classical_model(model, vectorizer, model_path, vectorizer_path):
    """
    Save the trained model and vectorizer.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load_classical_model(model_path, vectorizer_path):
    """
    Load a saved model and vectorizer.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer