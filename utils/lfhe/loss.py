"""LFHE frozen loss functions: MSE (control) and DA-MSE (treatment)."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

try:
    from keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.saving import register_keras_serializable


def _per_sequence_std(x: tf.Tensor, epsilon: float) -> tf.Tensor:
    """Population std (ddof=0) over the last axis (horizon dimension)."""
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=-1)
    return tf.sqrt(var + epsilon)


def make_mse_loss() -> str:
    """Standard MSE — identical to CSRLE Stage 2 control."""
    return "mse"


def make_da_mse_loss(
    lambda_disp: float = 1.0,
    epsilon: float = 1e-6,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Dispersion-Augmented MSE (frozen LFHE v1.1).

    L = MSE + λ · max(0, log(σ_y + ε) - log(σ_ŷ + ε))²

    Per-sequence σ computed over the 96-step horizon (ddof=0 via population variance).
    One-sided penalty: activates only when σ_ŷ < σ_y.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse = tf.reduce_mean(tf.square(y_pred - y_true))

        sigma_y = _per_sequence_std(y_true, epsilon)
        sigma_y_hat = _per_sequence_std(y_pred, epsilon)

        delta = tf.math.log(sigma_y) - tf.math.log(sigma_y_hat)
        disp = tf.square(tf.maximum(0.0, delta))
        return mse + lambda_disp * tf.reduce_mean(disp)

    return loss


@register_keras_serializable(package="lfhe", name="da_mse_loss")
def da_mse_loss_registered(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    lambda_disp: float = 1.0,
    epsilon: float = 1e-6,
) -> tf.Tensor:
    """Registered DA-MSE for model save/load (frozen λ=1.0 default)."""
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    sigma_y = _per_sequence_std(y_true, epsilon)
    sigma_y_hat = _per_sequence_std(y_pred, epsilon)
    delta = tf.math.log(sigma_y) - tf.math.log(sigma_y_hat)
    disp = tf.square(tf.maximum(0.0, delta))
    return mse + lambda_disp * tf.reduce_mean(disp)


def da_mse_components(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    lambda_disp: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return (total, mse, dispersion) for testing and logging."""
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    sigma_y = _per_sequence_std(y_true, epsilon)
    sigma_y_hat = _per_sequence_std(y_pred, epsilon)
    delta = tf.math.log(sigma_y) - tf.math.log(sigma_y_hat)
    disp = tf.reduce_mean(tf.square(tf.maximum(0.0, delta)))
    total = mse + lambda_disp * disp
    return total, mse, disp
