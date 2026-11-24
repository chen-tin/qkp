"""Independent finite-circuit spectral action against QuTiP ODE controls."""
import numpy as np
import pytest
import qutip as qt
from quantum.spectral_circuit import SpectralCircuit,real_left
from quantum.measurement_benchmark import ExplicitQutip


@pytest.mark.parametrize('axis',['q','p'])
@pytest.mark.parametrize('coupling',[.87,1.,1.13])
def test_spectral_two_mode_unitary_matches_independent_ode(axis,coupling):
    state=qt.coherent(8,.2+.1j).full().ravel()
    expected=ExplicitQutip(8,64,.5,np.empty(0),coupling=coupling,axes=(axis,)).evolve(state,axis)
    actual=SpectralCircuit(8,64,.5,coupling=coupling,chunk=7).evolve(state,axis)
    np.testing.assert_allclose(actual,expected,atol=3e-10,rtol=3e-10)
    assert abs(np.vdot(actual,actual).real-1)<1e-12


def test_tiled_real_transform_matches_dense_complex_multiplication():
    rng=np.random.default_rng(26112401)
    vectors=rng.normal(size=(17,13));matrix=rng.normal(size=(13,11))+1j*rng.normal(size=(13,11))
    np.testing.assert_allclose(real_left(vectors,matrix,3),vectors@matrix,atol=1e-13)
    np.testing.assert_allclose(real_left(vectors,matrix[:,0],3),vectors@matrix[:,0],atol=1e-13)
