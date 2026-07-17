"""Regression models defined by user-supplied PyTensor formulas."""

# Authors: pymc-learn Developers
# License: BSD 3 clause

import numpy as np
import pymc as pm
import pytensor

from ..base import BayesianModel
from ..exceptions import NotFittedError


class FunctionalRegression(BayesianModel):
    """Bayesian regression with a user-defined mean formula.

    Parameters
    ----------
    formula : callable
        Callable receiving the two-dimensional PyTensor input matrix and
        returning the mean of the response distribution. The callable is run
        inside a PyMC model context, so it can declare any priors needed by the
        formula using PyMC random variables. For example, a quadratic model can
        be defined as::

            def quadratic(X):
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
                return alpha + beta[0] * X[:, 0] + beta[1] * X[:, 0] ** 2

    sigma : float, optional
        Scale of the HalfNormal prior used for the observation noise. Defaults
        to 1.

    Notes
    -----
    The formula must use PyTensor-compatible operations (for example,
    ``pytensor.tensor.exp`` rather than ``numpy.exp``). ``fit`` uses a Normal
    observation likelihood around the value returned by ``formula``.
    """

    def __init__(self, formula, sigma=1):
        super(FunctionalRegression, self).__init__()
        if not callable(formula):
            raise TypeError('`formula` must be callable.')
        if not np.isscalar(sigma) or sigma <= 0:
            raise ValueError('`sigma` must be a positive scalar.')

        self.formula = formula
        self.sigma = sigma
        self.n_features_in_ = None

    def _validate_X(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('`X` must be a two-dimensional array.')
        if X.shape[1] == 0:
            raise ValueError('`X` must contain at least one feature.')
        return X

    def create_model(self):
        """Create the PyMC model for the currently configured feature count."""
        model_input = pytensor.shared(
            np.zeros((self.num_training_samples, self.num_pred), dtype=float))
        model_output = pytensor.shared(
            np.zeros(self.num_training_samples, dtype=float))
        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        model = pm.Model()
        with model:
            mean = self.formula(model_input)
            observation_sigma = pm.HalfNormal('sigma', sigma=self.sigma)
            pm.Normal('y', mu=mean, sigma=observation_sigma,
                      observed=model_output)

        return model

    def fit(self, X, y, inference_type='advi', minibatch_size=None,
            inference_args=None):
        """Fit the priors declared by ``formula`` to ``X`` and ``y``.

        Parameters match :meth:`pmlearn.linear_model.LinearRegression.fit`.
        Minibatches are supported for ADVI just as they are for
        ``LinearRegression``.
        """
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            y = np.squeeze(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('`y` must be one-dimensional and match `X`.')

        self.num_training_samples, self.num_pred = X.shape
        self.n_features_in_ = self.num_pred
        self.inference_type = inference_type
        if not inference_args:
            inference_args = self._set_default_inference_args()

        # Formula priors are created when the model is built. Rebuild on every
        # fit so refitting cannot retain priors or observations from old data.
        self.cached_model = self.create_model()

        if minibatch_size:
            with self.cached_model:
                inference_args['more_replacements'] = {
                    self.shared_vars['model_input']: pm.Minibatch(
                        X, batch_size=minibatch_size),
                    self.shared_vars['model_output']: pm.Minibatch(
                        y, batch_size=minibatch_size),
                }
        else:
            self._set_shared_vars({'model_input': X, 'model_output': y})

        self._inference(inference_type, inference_args)
        return self

    def predict_proba(self, X, return_std=False):
        """Return posterior-predictive response means for ``X``.

        ``return_std=True`` additionally returns posterior-predictive standard
        deviations. The method is named ``predict_proba`` to provide the
        probability-oriented API of functional models; :meth:`predict` is an
        alias for the posterior-predictive mean.
        """
        if self.trace is None:
            raise NotFittedError('Run fit on the model before predict.')

        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('`X` has a different number of features than fit.')

        self._set_shared_vars({
            'model_input': X,
            'model_output': np.zeros(X.shape[0], dtype=float),
        })
        ppc = pm.sample_posterior_predictive(
            self.trace, model=self.cached_model, return_inferencedata=False)
        predictions = np.asarray(ppc['y'])
        # PyMC returns ``(chain, draw, observation)`` for an
        # ``InferenceData`` trace and ``(draw, observation)`` for a
        # ``MultiTrace``. Treat both leading sampling dimensions as draws.
        if predictions.ndim > 2:
            predictions = predictions.reshape(
                (-1,) + predictions.shape[2:])
        if return_std:
            return predictions.mean(axis=0), predictions.std(axis=0)
        return predictions.mean(axis=0)

    def predict(self, X, return_std=False):
        """Alias for :meth:`predict_proba`."""
        return self.predict_proba(X, return_std=return_std)
