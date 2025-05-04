import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from better_regressions.utils import format_array


def _repr_logistic_regression(estimator: LogisticRegression, var_name: str) -> str:
    lines = []
    lines.append(f"{var_name} = LogisticRegression()")
    lines.append(f"{var_name}.coef_ = {format_array(estimator.coef_)}")
    lines.append(f"{var_name}.intercept_ = {format_array(estimator.intercept_)}")
    lines.append(f"{var_name}.classes_ = {format_array(estimator.classes_)}")
    return "\n".join(lines)


@typed
class AutoClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, depth: int | None | Literal["auto"] = "auto", val_size: float = 0.5, random_state: int = 42):
        super().__init__()
        self.depth = depth
        self.val_size = val_size
        self.random_state = random_state

    def _get_model(self, depth: int | None) -> BaseEstimator:
        if depth is None:
            return LogisticRegression()
        else:
            return XGBClassifier(max_depth=depth)

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: ND) -> "AutoClassifier":
        if self.depth == "auto":
            best_score = float("inf")
            best_depth = None
            best_model = None
            for d in [None, 1, 2, 3]:
                model = self._get_model(d)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)
                score = log_loss(y_val, y_pred)
                if score < best_score:
                    best_score = score
                    best_depth = d
                    best_model = clone(model)
            self.depth_ = best_depth
            self.model_ = best_model
            self.model_.fit(X, y)
        else:
            self.model_ = self._get_model(self.depth)
            self.model_.fit(X, y)
            self.depth_ = self.depth
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> ND:
        return self.model_.predict(X)

    @typed
    def predict_proba(self, X: Float[ND, "n_samples n_features"]) -> ND:
        return self.model_.predict_proba(X)

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "model_"):
            return f"{var_name} = AutoClassifier(depth={repr(self.depth)})"
        if isinstance(self.model_, LogisticRegression):
            model_init = f"{var_name} = AutoClassifier(depth=None)"
            model_repr = _repr_logistic_regression(self.model_, f"{var_name}_logreg")
            set_model = f"{var_name}.model_ = {var_name}_logreg"
            set_depth = f"{var_name}.depth_ = None"
            return "\n".join([model_init, model_repr, set_model, set_depth])
        elif isinstance(self.model_, XGBClassifier):
            raise NotImplementedError("XGBClassifier __repr__ is not implemented")
        else:
            raise NotImplementedError(f"Unsupported model: {type(self.model_)}")
