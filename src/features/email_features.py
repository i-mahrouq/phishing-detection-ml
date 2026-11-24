# Email features: combine training-fitted TF-IDF with six scaled numerical indicators.

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from src.preprocessing.emails import urls
from .common import suspicious_count

NUMERIC_COLUMNS = (
    'text_length',
    'word_count',
    'url_count',
    'suspicious_keyword_count',
    'subject_length',
    'exclamation_count',
)


# Feature stage: calculate six email indicators in the saved column order.
def numerical_features(frame: pd.DataFrame) -> np.ndarray:
    result = np.empty((len(frame), len(NUMERIC_COLUMNS)), dtype=np.float32)
    subjects = frame['subject'].fillna('') if 'subject' in frame else [''] * len(frame)
    counts = frame['url_count'] if 'url_count' in frame else [None] * len(frame)
    for i, (text, subject, count) in enumerate(zip(frame.text.fillna(''), subjects, counts)):
        text, subject = str(text), str(subject)
        result[i] = (
            len(text),
            len(text.split()),
            len(urls(text)) if count is None or pd.isna(count) else count,
            suspicious_count(text),
            len(subject),
            text.count('!'),
        )
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Email numerical features must be finite and nonnegative')
    return result


# Email transformer: keep the training vocabulary and numeric scales together.
class EmailFeatures:

    # Setup stage: create the training-only text vectorizer and numeric scaler.
    def __init__(self, max_features=50000, min_df=2):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.scaler = MaxAbsScaler()

    # Fitting stage: learn the email vocabulary and scales from training data only.
    def fit_transform(self, training):
        if hasattr(self, 'n_features_'):
            raise RuntimeError('Already fitted; create a new instance for a new training run')
        text = self.vectorizer.fit_transform(training.text.fillna(''))
        numeric = self.scaler.fit_transform(numerical_features(training))
        self.n_features_ = text.shape[1] + len(NUMERIC_COLUMNS)
        return sparse.hstack((text, sparse.csr_matrix(numeric)), format='csr', dtype=np.float32)

    # Email feature stage: reuse the training vocabulary and scales without fitting.
    def transform(self, messages: pd.DataFrame) -> sparse.csr_matrix:
        if not hasattr(self, 'n_features_'):
            raise RuntimeError('Fit on training first')
        text = self.vectorizer.transform(messages.text.fillna(''))
        numeric = self.scaler.transform(numerical_features(messages))
        return sparse.hstack((text, sparse.csr_matrix(numeric)), format='csr', dtype=np.float32)

    # Schema stage: describe the exact feature order, types and representation.
    def schema(self):
        return dict(
            dtype='float32',
            format='CSR',
            n_features=self.n_features_,
            tfidf_columns=[0, self.n_features_ - len(NUMERIC_COLUMNS)],
            tfidf='word unigrams/bigrams; vocabulary/order in email_transformer.joblib',
            numeric_columns=list(NUMERIC_COLUMNS),
            numeric_offset=self.n_features_ - len(NUMERIC_COLUMNS),
            numeric_scaling='MaxAbsScaler fitted on training only',
            character_ngrams=False,
            reason='Omitted to limit memory and computation',
        )
