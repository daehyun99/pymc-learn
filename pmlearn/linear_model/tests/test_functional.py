"""Tests for user-defined functional regression models."""

import numpy as np
import pytest
import pymc as pm

from pmlearn.exceptions import NotFittedError
from pmlearn.linear_model import FunctionalRegression


def quadratic_formula(X):
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    return alpha + beta[0] * X[:, 0] + beta[1] * X[:, 0] ** 2


def test_functional_regression_builds_user_formula():
    model = FunctionalRegression(quadratic_formula)
    model.num_training_samples = 4
    model.num_pred = 1

    pymc_model = model.create_model()

    assert set(pymc_model.named_vars) == {'alpha', 'beta', 'sigma', 'y'}


def test_functional_regression_fits_and_predicts():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(25, 1))
    y = 1.5 + 2 * X[:, 0] - 0.5 * X[:, 0] ** 2 + rng.normal(scale=.1, size=25)
    model = FunctionalRegression(quadratic_formula)
    model.default_advi_sample_draws = 30

    model.fit(X, y, inference_args={'n': 250, 'progressbar': False})
    predictions, std = model.predict_proba(X[:3], return_std=True)

    assert predictions.shape == (3,)
    assert std.shape == (3,)
    np.testing.assert_equal(model.predict(X[:3]).shape, (3,))


def test_functional_regression_validates_inputs_and_fit_state():
    with pytest.raises(TypeError):
        FunctionalRegression(None)
    with pytest.raises(ValueError):
        FunctionalRegression(quadratic_formula, sigma=0)
    with pytest.raises(NotFittedError):
        FunctionalRegression(quadratic_formula).predict_proba(np.ones((2, 1)))
