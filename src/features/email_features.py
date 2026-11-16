"""Sparse word TF-IDF plus numerical metadata, fitted exclusively on training.

Inference accepts a DataFrame with `text` (same cleaned subject+body representation
as preprocessing), optional `subject`, and optional `url_count`. Supply URL counts
from MIME preprocessing when available, including links from HTML alternatives.
Without them, explicit links are counted in text. No source/label features enter X.
"""
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from src.preprocessing.emails import urls
from .common import suspicious_count

NUMERIC_COLUMNS = ('text_length', 'word_count', 'url_count',
                   'suspicious_keyword_count', 'subject_length', 'exclamation_count')


def numerical_features(frame: pd.DataFrame) -> np.ndarray:
    """Return six nonnegative float32 columns in NUMERIC_COLUMNS order.

    Prefer the pre-cleaning URL count supplied by MIME ingestion; otherwise
    count visible links. These raw values are scaled only by the fitted scaler.
    """
    result = np.empty((len(frame), len(NUMERIC_COLUMNS)), dtype=np.float32)
    subjects = frame['subject'].fillna('') if 'subject' in frame else [''] * len(frame)
    counts = frame['url_count'] if 'url_count' in frame else [None] * len(frame)
    for i, (text, subject, count) in enumerate(zip(frame.text.fillna(''), subjects, counts)):
        text, subject = str(text), str(subject)
        result[i] = (len(text), len(text.split()), len(urls(text)) if count is None or pd.isna(count) else count,
                     suspicious_count(text), len(subject), text.count('!'))
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Email numerical features must be finite and nonnegative')
    return result


class EmailFeatures:
    """Call fit_transform on training once; transform is inference-only thereafter."""
    def __init__(self, max_features=50000, min_df=2):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                                        min_df=min_df, sublinear_tf=True, dtype=np.float32)
        self.scaler = MaxAbsScaler()

    def fit_transform(self, training):
        if hasattr(self, 'n_features_'):
            raise RuntimeError('Already fitted; create a new instance for a new training run')
        text = self.vectorizer.fit_transform(training.text.fillna(''))
        numeric = self.scaler.fit_transform(numerical_features(training))
        self.n_features_ = text.shape[1] + len(NUMERIC_COLUMNS)
        return sparse.hstack((text, sparse.csr_matrix(numeric)), format='csr', dtype=np.float32)

    def transform(self, messages: pd.DataFrame) -> sparse.csr_matrix:
        """Reuse training vocabulary/IDF and MaxAbs scales; never densify TF-IDF.

        Word unigrams and bigrams capture individual terms and adjacent phrases.
        Sparse CSR storage avoids allocating one value per vocabulary term.
        """
        if not hasattr(self, 'n_features_'):
            raise RuntimeError('Fit on training first')
        text = self.vectorizer.transform(messages.text.fillna(''))
        numeric = self.scaler.transform(numerical_features(messages))
        return sparse.hstack((text, sparse.csr_matrix(numeric)), format='csr', dtype=np.float32)

    def schema(self):
        return dict(dtype='float32', format='CSR', n_features=self.n_features_,
                    tfidf_columns=[0, self.n_features_ - len(NUMERIC_COLUMNS)],
                    tfidf='word unigrams/bigrams; vocabulary/order in email_transformer.joblib',
                    numeric_columns=list(NUMERIC_COLUMNS), numeric_offset=self.n_features_ - len(NUMERIC_COLUMNS),
                    numeric_scaling='MaxAbsScaler fitted on training only',
                    character_ngrams=False, reason='Omitted to limit memory and computation')
