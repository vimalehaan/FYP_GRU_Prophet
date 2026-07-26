"""Unit tests for frozen LFHE DA-MSE loss."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from utils.lfhe.loss import da_mse_components, make_da_mse_loss


@pytest.fixture
def loss_fn():
    return make_da_mse_loss(lambda_disp=1.0, epsilon=1e-6)


def test_equals_mse_when_pred_variance_ge_target(loss_fn):
    """When σ_ŷ >= σ_y, dispersion penalty is zero; loss equals MSE."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=(8, 96)).astype(np.float32)
    # Scale predictions up so pred std >= true std per sequence
    y_pred = (y_true * 2.5).astype(np.float32)

    total, mse, disp = da_mse_components(
        tf.constant(y_true), tf.constant(y_pred),
    )
    assert float(disp.numpy()) == pytest.approx(0.0, abs=1e-6)
    assert float(total.numpy()) == pytest.approx(float(mse.numpy()), rel=1e-5)


def test_dispersion_penalty_when_under_dispersed(loss_fn):
    """Penalty activates when prediction std < target std."""
    rng = np.random.default_rng(7)
    y_true = rng.normal(size=(4, 96)).astype(np.float32)
    y_pred = (y_true * 0.05).astype(np.float32)

    total, mse, disp = da_mse_components(
        tf.constant(y_true), tf.constant(y_pred),
    )
    assert float(disp.numpy()) > 0.0
    assert float(total.numpy()) > float(mse.numpy())


def test_one_sided_no_penalty_over_dispersed(loss_fn):
    """Over-dispersed predictions: max(0, delta)=0."""
    rng = np.random.default_rng(3)
    y_true = rng.normal(scale=0.1, size=(4, 96)).astype(np.float32)
    y_pred = rng.normal(scale=1.0, size=(4, 96)).astype(np.float32)

    _, _, disp = da_mse_components(tf.constant(y_true), tf.constant(y_pred))
    assert float(disp.numpy()) == pytest.approx(0.0, abs=1e-6)


def test_no_nan_finite_loss(loss_fn):
    y_true = tf.random.normal((16, 96))
    y_pred = tf.random.normal((16, 96)) * 0.01
    val = float(loss_fn(y_true, y_pred).numpy())
    assert np.isfinite(val)


def test_gradient_no_explosion(loss_fn):
    y_true = tf.Variable(tf.random.normal((4, 96)))
    y_pred = tf.Variable(tf.random.normal((4, 96)) * 0.05)
    with tf.GradientTape() as tape:
        loss = loss_fn(y_true, y_pred)
    grads = tape.gradient(loss, y_pred)
    gnorm = float(tf.norm(grads).numpy())
    assert np.isfinite(gnorm)
    assert gnorm < 1e6
