import numpy as np
import pytest
import qutip as qt

from quantum.channels import pure_loss,dephasing


def test_identity_and_complete_loss():
    psi=qt.coherent(20,.8+.2j).full()
    same,k=pure_loss(psi,1,np.random.default_rng(1))
    np.testing.assert_array_equal(same,psi);assert k[0]==0
    vacuum,_=pure_loss(psi,0,np.random.default_rng(1))
    np.testing.assert_array_equal(vacuum[:,0],qt.basis(20,0).full()[:,0])
    np.testing.assert_array_equal(dephasing(psi,0,np.random.default_rng(1)),psi)


def test_fock_loss_distribution_and_energy():
    b=30000;psi=np.zeros((12,b),complex);psi[7]=1
    out,k=pure_loss(psi,.7,np.random.default_rng(12))
    assert abs(k.mean()-7*.3)<5*np.sqrt(7*.3*.7/b)
    assert np.all(np.argmax(np.abs(out),axis=0)==7-k)
    np.testing.assert_allclose(np.sum(np.abs(out)**2,axis=0),1,atol=1e-13)


def test_loss_against_qutip_lindblad_density_matrix():
    n,b=10,50000
    ket=(qt.basis(n,0)+.7j*qt.basis(n,2)+.4*qt.basis(n,4)).unit()
    eta=.79
    out,_=pure_loss(np.repeat(ket.full(),b,axis=1),eta,np.random.default_rng(41))
    empirical=out@out.conj().T/b
    expected=qt.mesolve(0*qt.num(n),ket,[0,-np.log(eta)],c_ops=[qt.destroy(n)]).states[-1].full()
    assert np.max(np.abs(empirical-expected))<.008


def test_dephasing_against_qutip_lindblad_density_matrix():
    n,b=8,60000
    ket=(qt.basis(n,0)+1j*qt.basis(n,1)+.6*qt.basis(n,3)).unit()
    strength=.17
    out=dephasing(np.repeat(ket.full(),b,axis=1),strength,np.random.default_rng(82))
    empirical=out@out.conj().T/b
    expected=qt.mesolve(0*qt.num(n),ket,[0,strength],c_ops=[qt.num(n)]).states[-1].full()
    assert np.max(np.abs(empirical-expected))<.006
    np.testing.assert_allclose(empirical.diagonal(),expected.diagonal(),atol=1e-9)


def test_invalid_channel_parameters_rejected():
    psi=qt.basis(8,0).full()
    for eta in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError):pure_loss(psi,eta,np.random.default_rng(0))
    with pytest.raises(ValueError):dephasing(psi,-.1,np.random.default_rng(0))
