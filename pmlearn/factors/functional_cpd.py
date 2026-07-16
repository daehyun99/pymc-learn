"""Functional conditional probability distributions built with PyMC."""

# Authors: pymc-learn Developers
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pymc as pm


class FunctionalCPD(object):
    """Defines a functional conditional probability distribution using PyMC.

    A ``FunctionalCPD`` represents an arbitrary conditional probability
    distribution whose probability law is supplied as a Python callable. The
    callable receives a dictionary of parent values and must return a PyMC
    distribution created with the ``.dist`` API, for example
    ``pm.Normal.dist(mu=..., sigma=...)``.

    Parameters
    ----------
    variable : str
        Name of the variable for which this CPD is defined.
    fn : callable
        Function that receives parent values and returns a PyMC distribution.
    parents : list of str, optional
        List of parent variable names. Defaults to no parents.
    vectorized : bool, optional
        Whether ``fn`` accepts the full parent sample DataFrame and returns a
        vectorized distribution. Defaults to False.

    Examples
    --------
    >>> import pandas as pd
    >>> import pymc as pm
    >>> from pmlearn.factors import FunctionalCPD
    >>> cpd = FunctionalCPD(
    ...     variable='x3',
    ...     fn=lambda parent_sample: pm.Normal.dist(
    ...         mu=1.0 + 0.2 * parent_sample['x1'] + 0.3 * parent_sample['x2'],
    ...         sigma=1.0,
    ...     ),
    ...     parents=['x1', 'x2'],
    ... )
    >>> parent_samples = pd.DataFrame({'x1': [5, 10], 'x2': [1, -1]})
    >>> cpd.sample(2, parent_samples, random_seed=42).shape
    (2,)
    """

    def __init__(self, variable, fn, parents=None, vectorized=False):
        if not isinstance(variable, str):
            raise TypeError('`variable` must be a string.')
        if not callable(fn):
            raise ValueError('`fn` must be a callable function.')

        self.variable = variable
        self.fn = fn
        self.parents = list(parents) if parents else []
        self.variables = [variable] + self.parents
        self.vectorized = vectorized

    def _validate_parent_sample(self, n_samples, parent_sample):
        if parent_sample is None:
            return
        if not isinstance(parent_sample, pd.DataFrame):
            raise TypeError('`parent_sample` must be a pandas DataFrame.')

        missing_parents = [
            parent for parent in self.parents
            if parent not in parent_sample.columns
        ]
        if missing_parents:
            raise ValueError(
                'Missing values for parent variables: {}'.format(
                    missing_parents
                )
            )
        if len(parent_sample) != n_samples:
            raise ValueError('Length of `parent_sample` must match `n_samples`.')

    def _draw_distribution(self, distribution, draws=1, random_seed=None):
        samples = pm.draw(distribution, draws=draws, random_seed=random_seed)
        return np.asarray(samples)

    def sample(self, n_samples=100, parent_sample=None, random_seed=None):
        """Simulate values for this CPD's variable.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 100.
        parent_sample : pandas.DataFrame, optional
            Parent values, one row per sample. Columns must include every
            parent in ``self.parents``.
        random_seed : int, numpy.random.Generator, optional
            Random seed forwarded to ``pymc.draw``.

        Returns
        -------
        numpy.ndarray
            Array of sampled values for ``self.variable``.
        """
        if n_samples < 1:
            raise ValueError('`n_samples` must be at least 1.')

        self._validate_parent_sample(n_samples, parent_sample)

        if self.vectorized:
            distribution = self.fn(parent_sample)
            return self._draw_distribution(
                distribution, draws=1, random_seed=random_seed
            ).reshape(-1)

        sampled_values = []
        if isinstance(random_seed, (int, np.integer)):
            rng = np.random.default_rng(random_seed)
            seeds = rng.integers(0, 2 ** 30, size=n_samples).tolist()
        else:
            seeds = [random_seed] * n_samples

        for i in range(n_samples):
            if parent_sample is None:
                parents = None
            else:
                parents = parent_sample.iloc[i].to_dict()
            distribution = self.fn(parents)
            sample = self._draw_distribution(
                distribution, draws=1, random_seed=seeds[i]
            ).reshape(-1)[0]
            sampled_values.append(sample)

        return np.asarray(sampled_values)

    def __str__(self):
        fn_name = getattr(self.fn, '__name__', self.fn.__class__.__name__)
        if fn_name == '<lambda>':
            fn_name = 'lambda fun.'
        if self.parents:
            return 'P({} | {}) = {}'.format(
                self.variable, ', '.join(self.parents), fn_name
            )
        return 'P({}) = {}'.format(self.variable, fn_name)

    def __repr__(self):
        return '<FunctionalCPD: {}> at {}'.format(self.__str__(), hex(id(self)))
