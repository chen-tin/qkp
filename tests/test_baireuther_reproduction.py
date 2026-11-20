import numpy as np
from experiments.baireuther_circuit import author_class, simulate


class TapeRandom:
    def __init__(self, tape):
        self.values = iter(tape.ravel())
    def rand(self, *shape):
        if not shape:
            return next(self.values)
        return np.array([next(self.values) for _ in range(np.prod(shape))]).reshape(shape)


def test_batched_circuit_matches_original_on_identical_faults():
    # High rates deliberately exercise many X/Y/Z faults, measurement faults,
    # ancilla reuse, hook propagation and the final-readout ordering.
    tape = np.random.default_rng(11).random((4, 12, 350))
    observed = simulate(4, 12, 0, px=.04, py=.03, pz=.02, pm=.05, tape=tape)
    for j in range(4):
        ref = author_class()(seed=0, pqx=.04, pqy=.03, pqz=.02,
                             pax=.04, pay=.03, paz=.02, pm=.05)
        reset = ref._reinitialize
        def reset_with_tape(seed):
            reset(seed)
            ref.rng = TapeRandom(tape[j])
        ref._reinitialize = reset_with_tape
        _, raw, events, _, final, labels = ref.make_run(0, 12)
        for actual, expected in zip(observed, (events, final, labels, raw)):
            np.testing.assert_array_equal(actual[j], expected)


def test_zero_fault_control():
    assert all(not a.any() for a in simulate(64, 100, 10, px=0, py=0, pz=0, pm=0))


def test_decay_fit_matches_published_form():
    from experiments.reproduce_baireuther2018 import fit_decay, fidelity_curve
    times = np.arange(81, 101)
    values = fidelity_curve(times, .00209, 1.4)
    np.testing.assert_allclose(fit_decay(times, values), [.00209, 1.4], rtol=2e-5)


def test_paper_probability_combination_and_causality():
    import torch
    from experiments.reproduce_baireuther2018 import PaperDecoder
    torch.manual_seed(2)
    model = PaperDecoder().eval()
    a,b = torch.randn(10),torch.randn(10)
    p,q = a.sigmoid(),b.sigmoid()
    torch.testing.assert_close(model.combine(a,b).sigmoid(),p*(1-q)+q*(1-p))
    x = torch.randn(3,20,8)
    f = torch.randn(3,4)
    lengths = torch.tensor([11,14,20])
    with torch.inference_mode():
        before = model(x,f,lengths)
        for i,t in enumerate(lengths):
            x[i,t:] = torch.randn_like(x[i,t:])
        torch.testing.assert_close(model(x,f,lengths),before)
