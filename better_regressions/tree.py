"""Tree-based linear regression."""

import lightgbm as lgb
import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from better_regressions.linear import Linear
from better_regressions.scaling import Scaler


@typed
class TreeLinear(RegressorMixin, BaseEstimator):
    """Combines tree-based models with linear regression.

    This model:
    1. Trains a tree-based model (LGBM, RandomForest, or ExtraTrees)
    2. Extracts leaf indices for each input point
    3. One-hot encodes and concatenates these indices as embeddings
    4. Optionally reduces dimensionality with PCA
    5. Fits a linear regression on the resulting embeddings

    Args:
        tree_type: Type of tree model to use ("rf" for RandomForest, "et" for ExtraTrees, "lgbm" for LightGBM)
        n_estimators: Number of trees in the ensemble
        max_depth: Maximum depth of each tree
        alpha: Regularization parameter for the linear model
        better_bias: Whether to use better bias handling in linear model
        min_samples_leaf: Minimum number of samples required at a leaf node
        random_state: Random state for reproducibility
        use_sparse: Whether to use sparse matrices internally (faster but may not work with all estimators)
        hidden_dim: Number of dimensions to reduce embeddings to with PCA (None=no reduction)
        pca_whiten: Whether to whiten the PCA components
    """

    def __init__(
        self,
        tree_type: Literal["rf", "et", "lgbm"] = "et",
        n_estimators: int = 100,
        max_depth: int | None = 3,
        alpha: float | Literal["bayes", "ard"] = "bayes",
        better_bias: bool = True,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        use_sparse: bool = False,
        hidden_dim: int | None = None,
        pca_whiten: bool = False,
    ):
        super().__init__()
        self.tree_type = tree_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha
        self.better_bias = better_bias
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.use_sparse = use_sparse
        self.hidden_dim = hidden_dim
        self.pca_whiten = pca_whiten

    def _get_tree_model(self):
        """Create and return the appropriate tree model."""
        if self.tree_type == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        elif self.tree_type == "et":
            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        elif self.tree_type == "lgbm":
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth else -1,
                min_child_samples=self.min_samples_leaf,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unknown tree type: {self.tree_type}")

    @typed
    def _get_leaf_embeddings(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples n_leaves"] | csr_matrix:
        """Extract leaf indices and convert to one-hot encoded embeddings."""
        # Get the leaf indices for each tree in the ensemble
        if self.tree_type == "lgbm":
            # LGBM's predict with pred_leaf=True returns leaf indices directly
            leaf_indices = self.tree_model_.predict(X, pred_leaf=True)
        else:
            leaf_indices = self.tree_model_.apply(X)  # Shape: (n_samples, n_estimators)

        # Create a unique identifier for each leaf
        n_samples, n_estimators = leaf_indices.shape

        # Get the number of unique leaves per tree
        tree_node_counts = []
        if self.tree_type in ["rf", "et"]:
            for tree_idx in range(n_estimators):
                tree = self.tree_model_.estimators_[tree_idx].tree_
                tree_node_counts.append(tree.node_count)
        elif self.tree_type == "lgbm":
            # For LGBM, we need an upper bound on leaf indices
            # We'll use max leaf index + 1 for each tree
            for tree_idx in range(n_estimators):
                max_leaf_idx = np.max(leaf_indices[:, tree_idx]) + 1
                tree_node_counts.append(max_leaf_idx)

        # Compute offset for each tree
        tree_offsets = np.zeros(n_estimators, dtype=np.int32)
        for i in range(1, n_estimators):
            tree_offsets[i] = tree_offsets[i - 1] + tree_node_counts[i - 1]

        # Apply offset to get unique leaf indices across all trees
        leaf_indices_with_offset = leaf_indices + tree_offsets[np.newaxis, :]

        # Flatten to create row indices for sparse matrix
        row_indices = np.repeat(np.arange(n_samples), n_estimators)
        col_indices = leaf_indices_with_offset.ravel()

        # Create sparse one-hot encoding
        data = np.ones_like(col_indices, dtype=np.float32)
        total_nodes = tree_offsets[-1] + tree_node_counts[-1]

        # Create sparse matrix
        leaf_embeddings = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, total_nodes))

        # Convert to dense array if not using sparse
        if not self.use_sparse:
            return leaf_embeddings.toarray()
        return leaf_embeddings

    @typed
    def _apply_pca(self, embeddings: Float[ND, "n_samples n_leaves"]) -> Float[ND, "n_samples hidden_dim"]:
        """Apply PCA dimensionality reduction to leaf embeddings."""
        # If no PCA or insufficient samples, return original embeddings
        if self.hidden_dim is None:
            return embeddings

        # Apply PCA transformation if fitted, otherwise fit and transform
        if hasattr(self, "pca_"):
            return self.pca_.transform(embeddings)
        else:
            # Calculate effective hidden_dim based on data
            effective_dim = min(self.hidden_dim, embeddings.shape[0], embeddings.shape[1])
            # Initialize and fit PCA
            self.pca_ = PCA(n_components=effective_dim, whiten=self.pca_whiten, random_state=self.random_state)
            return self.pca_.fit_transform(embeddings)

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "TreeLinear":
        """Fit the TreeLinear model.

        First trains the tree model, then extracts leaf embeddings,
        applies PCA if specified, and finally fits a linear model.
        """
        # Fit tree model and get leaf embeddings
        self.tree_model_ = self._get_tree_model()
        self.tree_model_.fit(X, y)
        leaf_embeddings = self._get_leaf_embeddings(X)
        self.n_leaves_ = leaf_embeddings.shape[1]

        # Log information about dimensions
        logger.info(f"Original dims: {self.n_leaves_} | n_samples: {leaf_embeddings.shape[0]} | method: {self.tree_type}")

        # Apply dimensionality reduction if specified
        if self.hidden_dim is not None:
            reduced_embeddings = self._apply_pca(leaf_embeddings)
            self.reduced_dim_ = reduced_embeddings.shape[1]
            logger.info(f"After PCA: {self.reduced_dim_} dimensions")
        else:
            reduced_embeddings = leaf_embeddings

        # Fit linear model on the (potentially reduced) embeddings
        linear = Linear(alpha=self.alpha, better_bias=self.better_bias)
        self.linear_model_ = Scaler(estimator=linear, x_method="standard", y_method="standard", use_feature_variance=True)
        self.linear_model_.fit(reduced_embeddings, y)

        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        """Predict using the TreeLinear model."""
        # Get leaf embeddings for test data
        leaf_embeddings = self._get_leaf_embeddings(X)

        # Handle potential dimension mismatch (test data might have fewer unique leaves)
        if leaf_embeddings.shape[1] < self.n_leaves_:
            # Pad with zeros to match training dimensions
            padded = np.zeros((leaf_embeddings.shape[0], self.n_leaves_), dtype=leaf_embeddings.dtype)
            padded[:, : leaf_embeddings.shape[1]] = leaf_embeddings
            leaf_embeddings = padded
        elif leaf_embeddings.shape[1] > self.n_leaves_:
            # Truncate to match training dimensions
            leaf_embeddings = leaf_embeddings[:, : self.n_leaves_]

        # Apply PCA if it was used during training
        if hasattr(self, "pca_"):
            leaf_embeddings = self.pca_.transform(leaf_embeddings)

        # Predict using the linear model
        return self.linear_model_.predict(leaf_embeddings)
