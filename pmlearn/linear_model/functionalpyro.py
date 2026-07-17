"""Regression models defined by user-supplied Pyro formulas."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(RuntimeError):
        pass


ArrayLike = Union[np.ndarray, torch.Tensor]
Formula = Callable[[torch.Tensor], torch.Tensor]


class FunctionalRegression:
    """Bayesian regression with a user-defined Pyro mean formula.

    Parameters
    ----------
    formula
        Callable receiving a two-dimensional PyTorch tensor ``X`` and
        returning a one-dimensional response mean.

        The callable executes inside a Pyro model, so it may declare priors
        using ``pyro.sample``.

    sigma
        Scale of the HalfNormal prior used for observation noise.

    device
        PyTorch device. Examples are ``"cpu"``, ``"cuda"`` and
        ``torch.device("cuda:0")``.

    dtype
        Floating-point dtype used for input data.

    random_state
        Random seed passed to Pyro.

    Notes
    -----
    Formula operations must be PyTorch-compatible. For example, use
    ``torch.exp`` rather than ``numpy.exp``.

    Priors declared by ``formula`` are assumed to be global parameters.
    Per-observation local latent variables require a different minibatch
    construction.
    """

    def __init__(
        self,
        formula: Formula,
        sigma: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        random_state: Optional[int] = None,
    ):
        if not callable(formula):
            raise TypeError("`formula` must be callable.")

        if not np.isscalar(sigma) or float(sigma) <= 0:
            raise ValueError("`sigma` must be a positive scalar.")

        if not dtype.is_floating_point:
            raise TypeError("`dtype` must be a floating-point torch dtype.")

        self.formula = formula
        self.sigma = float(sigma)
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.random_state = random_state

        self.n_features_in_: Optional[int] = None
        self.inference_type_: Optional[str] = None

        self.guide_: Optional[AutoNormal] = None
        self.svi_: Optional[SVI] = None
        self.mcmc_: Optional[MCMC] = None
        self.posterior_samples_: Optional[dict[str, torch.Tensor]] = None

        self.loss_history_: list[float] = []
        self.is_fitted_: bool = False

    def _validate_X(
        self,
        X: ArrayLike,
        *,
        check_features: bool = False,
    ) -> torch.Tensor:
        X = torch.as_tensor(
            X,
            dtype=self.dtype,
            device=self.device,
        )

        if X.ndim != 2:
            raise ValueError("`X` must be a two-dimensional array.")

        if X.shape[1] == 0:
            raise ValueError("`X` must contain at least one feature.")

        if not torch.isfinite(X).all().item():
            raise ValueError("`X` contains NaN or infinite values.")

        if (
            check_features
            and self.n_features_in_ is not None
            and X.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                "`X` has a different number of features than the fitted data: "
                f"expected {self.n_features_in_}, received {X.shape[1]}."
            )

        return X

    def _validate_y(
        self,
        y: ArrayLike,
        n_samples: int,
    ) -> torch.Tensor:
        y = torch.as_tensor(
            y,
            dtype=self.dtype,
            device=self.device,
        )

        if y.ndim != 1:
            y = y.squeeze()

        if y.ndim != 1:
            raise ValueError("`y` must be one-dimensional.")

        if y.shape[0] != n_samples:
            raise ValueError(
                "`y` must have the same number of samples as `X`."
            )

        if not torch.isfinite(y).all().item():
            raise ValueError("`y` contains NaN or infinite values.")

        return y

    def _model(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        total_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Internal Pyro probabilistic model."""

        # Formula-defined global priors are sampled here.
        mean = self.formula(X)

        if not torch.is_tensor(mean):
            raise TypeError(
                "`formula(X)` must return a torch.Tensor."
            )

        if mean.shape != (X.shape[0],):
            raise ValueError(
                "`formula(X)` must return shape "
                f"({X.shape[0]},), but returned {tuple(mean.shape)}."
            )

        observation_sigma = pyro.sample(
            "sigma",
            dist.HalfNormal(X.new_tensor(self.sigma)),
        )

        batch_size = X.shape[0]

        if total_size is None:
            likelihood_scale = 1.0
        else:
            if total_size < batch_size:
                raise ValueError(
                    "`total_size` cannot be smaller than the batch size."
                )
            likelihood_scale = float(total_size) / float(batch_size)

        # Only the likelihood is rescaled for minibatches.
        # Global priors above must not be scaled by N / batch_size.
        with pyro.plate("data", batch_size):
            with poutine.scale(scale=likelihood_scale):
                pyro.sample(
                    "y",
                    dist.Normal(mean, observation_sigma),
                    obs=y,
                )

        return mean

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        inference_type: str = "advi",
        minibatch_size: Optional[int] = None,
        inference_args: Optional[dict] = None,
    ) -> "FunctionalRegression":
        """Fit the priors declared by ``formula``.

        Parameters
        ----------
        X
            Feature matrix with shape ``(n_samples, n_features)``.

        y
            Response vector with shape ``(n_samples,)``.

        inference_type
            One of:

            - ``"advi"``, ``"svi"`` or ``"vi"``:
              SVI with an AutoNormal guide.
            - ``"nuts"`` or ``"mcmc"``:
              NUTS sampling.

        minibatch_size
            Minibatch size for SVI. NUTS requires full-batch data.

        inference_args
            For SVI, supported entries are:

            - ``num_steps``: default 3000
            - ``lr``: default 0.01
            - ``num_particles``: default 1
            - ``clip_norm``: default 10.0

            For NUTS, supported entries are:

            - ``num_samples``: default 1000
            - ``warmup_steps``: default 500
            - ``num_chains``: default 1
            - ``target_accept_prob``: default 0.8
            - ``max_tree_depth``: default 10
            - ``disable_progbar``: default False
        """
        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0])

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.loss_history_ = []
        self.is_fitted_ = False

        inference_args = dict(inference_args or {})
        inference_type = inference_type.lower()

        if self.random_state is not None:
            pyro.set_rng_seed(self.random_state)

        # Pyro parameters, including AutoNormal parameters, are stored in a
        # global ParamStore. Clear it to ensure that refitting starts cleanly.
        pyro.clear_param_store()

        if inference_type in {"advi", "svi", "vi"}:
            self._fit_svi(
                X=X,
                y=y,
                minibatch_size=minibatch_size,
                inference_args=inference_args,
            )
            self.inference_type_ = "svi"

        elif inference_type in {"nuts", "mcmc"}:
            if minibatch_size is not None:
                raise ValueError(
                    "NUTS does not support minibatches in this implementation."
                )

            self._fit_nuts(
                X=X,
                y=y,
                inference_args=inference_args,
            )
            self.inference_type_ = "nuts"

        else:
            raise ValueError(
                "`inference_type` must be one of "
                "{'advi', 'svi', 'vi', 'nuts', 'mcmc'}."
            )

        self.is_fitted_ = True
        return self

    def _fit_svi(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        minibatch_size: Optional[int],
        inference_args: dict,
    ) -> None:
        num_steps = int(inference_args.pop("num_steps", 3000))
        learning_rate = float(inference_args.pop("lr", 0.01))
        num_particles = int(inference_args.pop("num_particles", 1))
        clip_norm = float(inference_args.pop("clip_norm", 10.0))

        if inference_args:
            unknown = ", ".join(sorted(inference_args))
            raise TypeError(
                f"Unsupported SVI inference arguments: {unknown}."
            )

        if num_steps <= 0:
            raise ValueError("`num_steps` must be positive.")

        if learning_rate <= 0:
            raise ValueError("`lr` must be positive.")

        if num_particles <= 0:
            raise ValueError("`num_particles` must be positive.")

        n_samples = X.shape[0]

        if minibatch_size is None:
            batch_size = n_samples
        else:
            if not isinstance(minibatch_size, int):
                raise TypeError("`minibatch_size` must be an integer.")

            if minibatch_size <= 0:
                raise ValueError("`minibatch_size` must be positive.")

            batch_size = min(minibatch_size, n_samples)

        self.guide_ = AutoNormal(self._model)

        optimizer = ClippedAdam(
            {
                "lr": learning_rate,
                "clip_norm": clip_norm,
            }
        )

        self.svi_ = SVI(
            model=self._model,
            guide=self.guide_,
            optim=optimizer,
            loss=Trace_ELBO(num_particles=num_particles),
        )

        self.mcmc_ = None
        self.posterior_samples_ = None

        for _ in range(num_steps):
            if batch_size == n_samples:
                X_batch = X
                y_batch = y
            else:
                # Sample without replacement for the current optimization step.
                indices = torch.randperm(
                    n_samples,
                    device=X.device,
                )[:batch_size]

                X_batch = X[indices]
                y_batch = y[indices]

            loss = self.svi_.step(
                X_batch,
                y_batch,
                total_size=n_samples,
            )

            self.loss_history_.append(float(loss))

    def _fit_nuts(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        inference_args: dict,
    ) -> None:
        num_samples = int(inference_args.pop("num_samples", 1000))
        warmup_steps = int(inference_args.pop("warmup_steps", 500))
        num_chains = int(inference_args.pop("num_chains", 1))
        target_accept_prob = float(
            inference_args.pop("target_accept_prob", 0.8)
        )
        max_tree_depth = int(
            inference_args.pop("max_tree_depth", 10)
        )
        disable_progbar = bool(
            inference_args.pop("disable_progbar", False)
        )

        if inference_args:
            unknown = ", ".join(sorted(inference_args))
            raise TypeError(
                f"Unsupported NUTS inference arguments: {unknown}."
            )

        kernel = NUTS(
            self._model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )

        self.mcmc_ = MCMC(
            kernel=kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
            disable_progbar=disable_progbar,
        )

        self.mcmc_.run(
            X,
            y,
            total_size=X.shape[0],
        )

        self.posterior_samples_ = self.mcmc_.get_samples(
            group_by_chain=False
        )

        self.guide_ = None
        self.svi_ = None

    def predict_proba(
        self,
        X: ArrayLike,
        return_std: bool = False,
        num_samples: int = 1000,
    ):
        """Return posterior-predictive response means.

        Parameters
        ----------
        X
            Prediction feature matrix.

        return_std
            Return posterior-predictive standard deviations together with
            the means.

        num_samples
            Number of posterior predictive samples when SVI was used.

            For NUTS, the number is determined by the available posterior
            samples.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "Run `fit` on the model before prediction."
            )

        X = self._validate_X(X, check_features=True)

        if num_samples <= 0:
            raise ValueError("`num_samples` must be positive.")

        if self.inference_type_ == "svi":
            if self.guide_ is None:
                raise RuntimeError("The fitted SVI guide is unavailable.")

            predictive = Predictive(
                model=self._model,
                guide=self.guide_,
                num_samples=num_samples,
                return_sites=("y",),
            )

        elif self.inference_type_ == "nuts":
            if self.posterior_samples_ is None:
                raise RuntimeError(
                    "The fitted NUTS posterior samples are unavailable."
                )

            predictive = Predictive(
                model=self._model,
                posterior_samples=self.posterior_samples_,
                return_sites=("y",),
            )

        else:
            raise RuntimeError(
                "The fitted inference method is not recognized."
            )

        with torch.no_grad():
            posterior_predictive = predictive(
                X,
                y=None,
                total_size=X.shape[0],
            )

        predictions = posterior_predictive["y"]

        # Expected shape is (posterior_samples, observations). Flatten any
        # additional leading sampling dimensions, such as chain dimensions.
        predictions = predictions.reshape(-1, X.shape[0])

        prediction_mean = predictions.mean(dim=0)

        if return_std:
            prediction_std = predictions.std(
                dim=0,
                unbiased=False,
            )

            return (
                prediction_mean.cpu().numpy(),
                prediction_std.cpu().numpy(),
            )

        return prediction_mean.cpu().numpy()

    def predict(
        self,
        X: ArrayLike,
        return_std: bool = False,
        num_samples: int = 1000,
    ):
        """Alias for :meth:`predict_proba`."""
        return self.predict_proba(
            X,
            return_std=return_std,
            num_samples=num_samples,
        )
