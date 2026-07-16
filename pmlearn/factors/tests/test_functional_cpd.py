import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pmlearn.factors import FunctionalCPD


def test_functional_cpd_samples_from_parent_values():
    cpd = FunctionalCPD(
        variable='x3',
        fn=lambda parent_sample: pm.Normal.dist(
            mu=1.0 + 0.2 * parent_sample['x1'] + 0.3 * parent_sample['x2'],
            sigma=1.0,
        ),
        parents=['x1', 'x2'],
    )
    parent_samples = pd.DataFrame({'x1': [5, 10], 'x2': [1, -1]})

    samples = cpd.sample(2, parent_samples, random_seed=42)

    assert samples.shape == (2,)
    assert np.issubdtype(samples.dtype, np.floating)
    assert str(cpd) == 'P(x3 | x1, x2) = lambda fun.'


def test_functional_cpd_validates_parent_sample():
    cpd = FunctionalCPD(
        variable='x',
        fn=lambda parent_sample: pm.Normal.dist(mu=parent_sample['z'], sigma=1),
        parents=['z'],
    )

    with pytest.raises(ValueError, match='Missing values'):
        cpd.sample(1, pd.DataFrame({'other': [1]}))


def test_functional_cpd_vectorized_sampling():
    cpd = FunctionalCPD(
        variable='x',
        fn=lambda parent_sample: pm.Normal.dist(
            mu=parent_sample['z'].to_numpy(), sigma=1.0
        ),
        parents=['z'],
        vectorized=True,
    )

    samples = cpd.sample(3, pd.DataFrame({'z': [0.0, 1.0, 2.0]}), random_seed=7)

    assert samples.shape == (3,)
