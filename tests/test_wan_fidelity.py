import numpy as np
from scipy.integrate import quad
from scipy.special import roots_hermitenorm

from experiments.wan_fidelity_audit import L,ModularChannel,fidelity,worst_fidelity


def test_extracted_map_is_a_quantum_channel():
    model=ModularChannel()
    for variance in [0,.1]:
        channel=model.map(variance,variance)
        np.testing.assert_allclose(np.einsum('aaij->ij',channel),np.eye(2),atol=1e-12)
        np.testing.assert_allclose(channel,channel.transpose(1,0,3,2),atol=1e-13)
        choi=channel.transpose(0,2,1,3).reshape(4,4)
        assert np.linalg.eigvalsh(choi).min()>-1e-12


def test_sum_cutoffs_are_converged():
    small,large=ModularChannel(),ModularChannel(peaks=16,bins=24)
    for variance in [0,.1]:
        np.testing.assert_allclose(small.map(variance,variance),large.map(variance,variance),atol=1e-10)


def test_analytic_bin_integrals_against_wavefunction_quadrature():
    model=ModularChannel()
    nodes,weights=roots_hermitenorm(64)
    weights/=np.sqrt(2*np.pi)
    for variance in [0,.1]:
        out=model.map(variance,variance)
        # Nontrivial diagonal and coherence components, using independent
        # quadrature over both modular position and random displacement.
        for ell,other,i,j in [(0,0,0,0),(0,1,0,1),(0,1,1,0)]:
            val=0.
            for m in range(-10,11):
                def integrand(u):
                    a=model.wavefunctions((2*m+ell)*L+u-nodes*np.sqrt(variance))[:,i]
                    b=model.wavefunctions((2*m+other)*L+u-nodes*np.sqrt(variance))[:,j]
                    return float(weights@(a*b))
                val+=quad(integrand,-L/2,L/2,epsabs=2e-11)[0]
            val*=np.exp(-variance*((ell-other)*L)**2/2)
            np.testing.assert_allclose(val,out[ell,other,i,j],atol=2e-10)


def test_worst_state_includes_complex_amplitudes():
    channel=ModularChannel().map(.1,.1)
    worst=worst_fidelity(channel)['fidelity']
    assert worst<=fidelity(channel,[1,1j])+1e-10
    assert worst<fidelity(channel,[1,1])-.009
