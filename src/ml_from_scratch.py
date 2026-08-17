"""
From-scratch Machine Learning primitives (NumPy implementation).
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between rows of matrix a and matrix b.
    Returns matrix of shape (n_rows_a, n_rows_b).
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm = np.where(a_norm == 0, 1.0, a_norm)
    b_norm = np.where(b_norm == 0, 1.0, b_norm)

    a_unit = a / a_norm
    b_unit = b / b_norm

    return a_unit @ b_unit.T


class NeuralNetwork:
    """
    Single hidden-layer Multi-Layer Perceptron (Linear -> ReLU -> Linear -> Softmax).
    Trained via mini-batch gradient descent with manual backpropagation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He (Kaiming) weight initialization for ReLU
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    # -- Forward Pass ---------------------------------------------------------

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(z.dtype)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - z.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute layer activations: returns (z1, a1, probabilities)."""
        z1 = X @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        probs = self._softmax(z2)
        return z1, a1, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Output class probability distribution, shape (n_samples, output_dim)."""
        _, _, probs = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Output predicted class index for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    # -- Training -------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.05,
        class_weight: str | None = None,
        seed: int = 0,
        log_every: int = 10,
        logger=None,
    ) -> dict:
        """
        Train the model using mini-batch gradient descent and early checkpointing.
        Restores best validation weights upon completion.
        """
        n_samples, input_dim = X.shape
        n_classes = self.w2.shape[1]
        y_onehot = np.eye(n_classes)[y]

        sample_weight = np.ones(n_samples)
        if class_weight == "balanced":
            class_counts = np.bincount(y, minlength=n_classes)
            class_counts = np.where(class_counts == 0, 1, class_counts)
            weight_per_class = n_samples / (n_classes * class_counts)
            sample_weight = weight_per_class[y]

        rng = np.random.default_rng(seed)
        best_val_accuracy = -1.0
        best_weights = None
        val_accuracy_history = []

        for epoch in range(1, epochs + 1):
            shuffled_order = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_idx = shuffled_order[start : start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]
                weight_batch = sample_weight[batch_idx]
                batch_size_actual = len(batch_idx)

                z1, a1, probs = self._forward(X_batch)

                dz2 = (probs - y_batch) * weight_batch[:, None] / batch_size_actual
                dw2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.w2.T
                dz1 = da1 * self._relu_grad(z1)
                dw1 = X_batch.T @ dz1
                db1 = dz1.sum(axis=0)

                self.w2 -= learning_rate * dw2
                self.b2 -= learning_rate * db2
                self.w1 -= learning_rate * dw1
                self.b1 -= learning_rate * db1

            val_predictions = self.predict(X_val)
            val_accuracy = float(np.mean(val_predictions == y_val))
            val_accuracy_history.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

            if logger is not None and (epoch % log_every == 0 or epoch == epochs):
                logger.info(
                    "epoch %d/%d - val_accuracy=%.4f (best=%.4f)",
                    epoch, epochs, val_accuracy, best_val_accuracy,
                )

        if best_weights is not None:
            self.w1, self.b1, self.w2, self.b2 = best_weights

        return {
            "best_val_accuracy": best_val_accuracy,
            "val_accuracy_history": val_accuracy_history,
        }

    # -- Persistence ----------------------------------------------------------

    def save(self, path) -> None:
        """Save weights to a .npz file."""
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path) -> "NeuralNetwork":
        """Load weights from a .npz file into a NeuralNetwork instance."""
        weights = np.load(path)
        input_dim, hidden_dim = weights["w1"].shape
        output_dim = weights["w2"].shape[1]

        model = cls(input_dim, hidden_dim, output_dim)
        model.w1 = weights["w1"]
        model.b1 = weights["b1"]
        model.w2 = weights["w2"]
        model.b2 = weights["b2"]
        return model
