import matplotlib.pyplot as plt
import numpy as np
from better_regressions.classifier import AutoClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def plot_decision_boundary(clf, X, y, ax, title, proba=True):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if proba:
        Z = clf.predict_proba(grid)[:, 1]
        Z = Z.reshape(xx.shape)
        cs = ax.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm, levels=np.linspace(0, 1, 21))
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    Zb = clf.predict(grid)
    Zb = Zb.reshape(xx.shape)
    ax.contour(xx, yy, Zb, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=10, edgecolor="k")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def test_moons():
    sample_sizes = [100, 300, 1000]
    settings = [(None, "depth=None"), (2, "depth=2"), ("auto", "depth='auto'")]
    fig, axs = plt.subplots(len(sample_sizes), len(settings), figsize=(14, 10))
    for i, n_samples in enumerate(sample_sizes):
        X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        for j, (depth, label) in enumerate(settings):
            clf = AutoClassifier(depth=depth)
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict_proba(X_train)[:, 1]
            y_val_pred = clf.predict_proba(X_val)[:, 1]
            train_loss = log_loss(y_train, y_train_pred)
            val_loss = log_loss(y_val, y_val_pred)
            title = f"n={n_samples}, {label}\ntrain logloss={train_loss:.3f}, val logloss={val_loss:.3f}"
            plot_decision_boundary(clf, X_val, y_val, axs[i, j], title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_moons()
