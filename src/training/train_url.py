"""Independent URL V1 training: scaled Logistic Regression and Random Forest.

Uses saved URL train/validation features only; no email artifacts are required.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import train_kind, training_main


def model_configs(seed=42):
    return {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, n_jobs=2, random_state=seed),
    }


def train(root, seed=42):
    return train_kind(root, 'url', seed, model_configs(seed))


def main(argv=None):
    training_main('url', model_configs, argv)


if __name__ == '__main__':
    main()
