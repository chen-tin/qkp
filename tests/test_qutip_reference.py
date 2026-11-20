import math

import numpy as np
import qutip as qt
import pytest
from scipy.integrate import trapezoid

from quantum.qutip_reference import Ancilla,Oscillator,CARDINAL,raw_code,L,hermite_functions
from experiments.qutip_reference import noise,decoder
from simulator.sim import Simulator,wrap,logical_flip


def test_code_and_logical_extraction():
    osc=Oscillator(96)
    np.testing.assert_allclose(osc.code.conj().T@osc.code,np.eye(2),atol=1e-12)
    assert osc.extraction_trace_error<1e-7
    for _,c,a,s in CARDINAL:
        psi=(osc.code@c)[:,None]
        b=osc.observables(psi)['bloch'][:,0]
        assert np.linalg.norm(b)<=1+1e-10
        # Finite-energy gauge tracing has nonunit initial X/Y fidelity.
        assert (1+s*b[a])/2>.95


def test_high_order_hermite_turning_points_do_not_underflow():
    x=np.linspace(-45,45,18001)
    phi=hermite_functions(x,780)
    for n in [0,383,767,779]:
        assert abs(trapezoid(phi[n]**2,x)-1)<1e-10


def test_displacement_against_qutip_operator():
    osc=Oscillator(64)
    psi=qt.coherent(64,.2+.1j).full()
    actual=osc.displacement(psi,np.array([.1]),np.array([.2]))
    expected=(qt.displace(64,(.1+.2j)/math.sqrt(2))*qt.Qobj(psi)).full()
    assert abs(np.vdot(expected,actual))**2>1-1e-12
    corrected=osc.correct(osc.correct(actual,np.array([.2]),'p'),np.array([.1]),'q')
    assert abs(np.vdot(psi,corrected))**2>1-1e-12


def test_finite_ancilla_mehler_against_fock_wavefunction():
    anc=Ancilla(.5,dx=.0005)
    c=raw_code(100,anc.delta).sum(axis=1);c/=np.linalg.norm(c)
    x=np.linspace(-8,8,2001)
    expected=c@hermite_functions(x,100)
    measured=anc.amplitude(x)
    phase=np.sign(np.dot(measured,expected))
    np.testing.assert_allclose(measured,phase*expected,atol=5e-4)
    assert abs(np.dot(anc.amp,anc.amp)*anc.dx-1)<1e-12


@pytest.mark.parametrize('axis',['q','p'])
def test_conditional_kraus_against_explicit_two_mode_qutip_sum(axis):
    nd,na=6,64
    a=qt.destroy(nd);q=(a+a.dag())/math.sqrt(2)
    b=qt.destroy(na);p=-1j*(b-b.dag())/math.sqrt(2)
    anc=Ancilla(.5,dx=.0001)
    c=raw_code(na,anc.delta).sum(axis=1);c/=np.linalg.norm(c)
    data=qt.coherent(nd,.15+.10j)
    if axis=='q':
        joint=(-1j*qt.tensor(q,p)).expm()*qt.tensor(data,qt.Qobj(c[:,None]))
        measured_phase=np.ones(na)
        observable=q
    else:
        pd=-1j*(a-a.dag())/math.sqrt(2);qa=(b+b.dag())/math.sqrt(2)
        ancilla_p=(1j**np.arange(na))*c
        joint=(1j*qt.tensor(pd,qa)).expm()*qt.tensor(data,qt.Qobj(ancilla_p[:,None]))
        measured_phase=(-1j)**np.arange(na)
        observable=pd
    y=.37
    phi=hermite_functions(np.array([y]),na)[:,0]
    explicit=joint.full().reshape(nd,na)@(phi*measured_phase)
    vals,vec=np.linalg.eigh(observable.full())
    reduced=vec@(anc.amplitude(y-vals)*(vec.conj().T@data.full().ravel()))
    np.testing.assert_allclose(reduced,explicit,atol=1e-4)


def test_classical_control_matches_original_simulator_each_quadrature():
    params=(.20,.05,.90);b,rounds=24,5
    dq,dp,nq,np_,_,_=noise(params,b,rounds,1741)
    for name in ['nearest','bayes_gaussian']:
        for increments,measurement in [(dq,nq),(dp,np_)]:
            control=decoder(name,params,b);x=np.zeros(b);records=[]
            for t in range(rounds):
                x+=increments[:,t]
                c=control.decode(wrap(x+measurement[:,t]));c=(c+L)%(2*L)-L
                x-=c;records.append(logical_flip(x))
            expected=Simulator(trials=b,rounds=rounds).run(decoder(name,params,b),increments,measurement).logical_errors
            np.testing.assert_array_equal(np.array(records).T,expected)
