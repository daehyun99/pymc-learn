"""Functional conditional probability distributions built with PyMC."""

# Authors: pymc-learn Developers
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as tt


class FunctionalCPD(object):
    """
    Functional conditional probability distribution built using PyMC.

    ``FunctionalCPD`` represents an arbitrary conditional distribution where a
    user-supplied callable maps parent values to a PyMC distribution. The class
    mirrors the small, user-facing style of pymc-learn estimators while keeping
    the CPD independent from a full fitted estimator.

    Parameters
    ----------
    variable : str
        Name of the variable for which this CPD is defined.

    fn : callable
        Function that accepts a dictionary of parent values and returns a PyMC
        distribution, for example ``pm.Normal.dist(mu=..., sigma=...)``.

    parents : list[str], optional
        List of parent variable names. Defaults to no parents.

    vectorized : bool, optional
        If True, ``fn`` receives all parent columns at once and must return a
        batched PyMC distribution. If False, one distribution is created per
        sample row. Defaults to False.

    Examples
    --------
    >>> import pandas as pd
    >>> import pymc as pm
    >>> from pmlearn.factors import FunctionalCPD
    >>> cpd = FunctionalCPD(
    ...     variable='x3',
    ...     fn=lambda parent_sample: pm.Normal.dist(
    ...         mu=1.0 + 0.2 * parent_sample['x1'] + 0.3 * parent_sample['x2'],
    ...         sigma=1,
    ...     ),
    ...     parents=['x1', 'x2'],
    ... )
    >>> parent_samples = pd.DataFrame({'x1': [5, 10], 'x2': [1, -1]})
    >>> cpd.sample(2, parent_samples).shape
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

    def sample(self, n_samples=100, parent_sample=None, random_seed=None):
        """
        Simulate values for the variable based on its CPD.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 100.

        parent_sample : pandas.DataFrame, optional
            DataFrame where each column represents a parent variable and each
            row contains the parent values for one generated sample.

        random_seed : int, optional
            Seed passed to ``pymc.draw`` for reproducible samples.

        Returns
        -------
        sampled_values : numpy.ndarray
            Array of sampled values for the variable.
        """
        if n_samples < 1:
            raise ValueError('`n_samples` must be a positive integer.')

        if parent_sample is not None:
            parent_sample = self._validate_parent_sample(parent_sample,
                                                         n_samples)

        if self.vectorized:
            distribution = self.fn(self._vectorized_parent_values(parent_sample))
            draws = pm.draw(distribution, draws=1, random_seed=random_seed)
            return np.asarray(draws).reshape(-1)

        sampled_values = []
        for index in range(n_samples):
            parent_values = self._row_parent_values(parent_sample, index)
            distribution = self.fn(parent_values)
            sampled_values.append(pm.draw(distribution, draws=1,
                                          random_seed=random_seed).item())

        return np.asarray(sampled_values)

    def _validate_parent_sample(self, parent_sample, n_samples):
        if not isinstance(parent_sample, pd.DataFrame):
            raise TypeError('`parent_sample` must be a pandas DataFrame.')

        missing_parents = [p for p in self.parents
                           if p not in parent_sample.columns]
        if missing_parents:
            raise ValueError('Missing values for parent variables: {}'.format(
                missing_parents))

        if len(parent_sample) != n_samples:
            raise ValueError('Length of `parent_sample` must match `n_samples`.')

        return parent_sample

    def _vectorized_parent_values(self, parent_sample):
        if parent_sample is None:
            return None

        return {
            parent: tt.as_tensor_variable(parent_sample[parent].to_numpy())
            for parent in self.parents
        }

    def _row_parent_values(self, parent_sample, index):
        if parent_sample is None:
            return None

        row = parent_sample.iloc[index]
        return {
            parent: tt.as_tensor_variable(row[parent])
            for parent in self.parents
        }

    def __str__(self):
        fn_name = getattr(self.fn, '__name__', self.fn.__class__.__name__)
        if fn_name == '<lambda>':
            fn_name = 'lambda fun.'

        if self.parents:
            return 'P({} | {}) = {}'.format(
                self.variable, ', '.join(self.parents), fn_name)
        return 'P({}) = {}'.format(self.variable, fn_name)

    def __repr__(self):
        return '<FunctionalCPD: {}> at {}'.format(self.__str__(), hex(id(self)))
