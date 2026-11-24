"""Independent arbitrary-precision controls of high-order reference functions."""
import math
import numpy as np
import mpmath as mp
import pytest
from quantum.measurement_benchmark import qutip_wave_basis


@pytest.mark.parametrize('n,x',[(4095,40.),(4095,60.),(4095,80.),
                               (8190,120.),(8191,80.),(8191,100.),(8191,120.),
                               (12287,100.),(12287,140.),(12287,150.),(12287,160.),
                               (16383,160.),(16382,120.)])
def test_high_order_reference_matches_arbitrary_precision(n,x):
    with mp.workdps(90):
        golden=float(mp.hermite(n,mp.mpf(x))*mp.exp(-mp.mpf(x)**2/2)/
                     mp.sqrt(mp.sqrt(mp.pi)*mp.power(2,n)*mp.factorial(n)))
    actual=qutip_wave_basis(n+1,np.array([x]))[n,0]
    assert actual==pytest.approx(golden,rel=2e-10,abs=2e-12)


@pytest.mark.parametrize('x',[-8.,-1.2,0.,1.2,8.])
def test_reference_even_odd_parity_and_low_order_precision(x):
    rows=qutip_wave_basis(40,np.array([x,-x]))
    np.testing.assert_allclose(rows[:,1],(-1.)**np.arange(40)*rows[:,0],atol=1e-14)
    for n in (0,1,7,20,39):
        with mp.workdps(70):
            golden=float(mp.hermite(n,x)*mp.exp(-mp.mpf(x)**2/2)/
                         mp.sqrt(mp.sqrt(mp.pi)*mp.power(2,n)*mp.factorial(n)))
        assert rows[n,0]==pytest.approx(golden,rel=2e-12,abs=2e-14)
