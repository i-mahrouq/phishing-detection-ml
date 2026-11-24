"""Independent Email V1 training: Logistic Regression and Linear SVM.

Retain the original unweighted/balanced configurations of these two families.
Uses saved email train/validation features only; no URL artifacts are required.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .common import train_kind, training_main


def model_configs(seed=42):
    configurations = {}
    for weight in (None, 'balanced'):
        suffix = weight or 'none'
        configurations[f'logistic_regression_{suffix}'] = LogisticRegression(
            solver='liblinear', max_iter=1000, random_state=seed, class_weight=weight)
    for weight in (None, 'balanced'):
        suffix = weight or 'none'
        configurations[f'linear_svm_{suffix}'] = LinearSVC(
            max_iter=10000, dual='auto', random_state=seed, class_weight=weight)
    return configurations


def train(root, seed=42):
    return train_kind(root, 'email', seed, model_configs(seed))


def main(argv=None):
    training_main('email', model_configs, argv)


if __name__ == '__main__':
    main()
