import numpy as np
from experiments.reproduce_wan2020 import (tracking_integral, posterior_matrix,
    exact_variance, filtering_variances, approximate_variance)


def test_single_round_posterior_closed_form():
    d,s=.2182,.0005
    exact=.25*d*d*s/(d*d+s)
    np.testing.assert_allclose(exact_variance(1,d,s),exact,rtol=1e-14)


def test_independent_matrix_and_filter():
    d,s=.2182,.0005
    values=filtering_variances(200,d,s)
    for m in [1,2,5,20,100,200]:
        np.testing.assert_allclose(exact_variance(m,d,s),values[m-1],rtol=1e-13)
    assert values[-1] < s/3


def test_neumann_variance_is_second_order_expansion():
    d,s=.2182,.0005
    m=20
    B,a,_=posterior_matrix(m,d,s)
    expanded=s*(a@a)-s*s/(d*d)*(a@B.T@B@a)
    np.testing.assert_allclose(approximate_variance(m,d,s),expanded,rtol=1e-14)


def test_tracking_zero_error_limit_and_monotonicity():
    zero,_=tracking_integral(.2182,0)
    small,_=tracking_integral(.2182,.0001)
    large,_=tracking_integral(.2182,.0005)
    assert zero == 0
    assert 0 < small < large < .001
