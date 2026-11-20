# Original user research notes — received 2026-11-19

The text below preserves the wording supplied by the user, including spelling, unfinished sections and reference numbering, with whitespace normalized. It is a historical source, not a statement that every interpretation or citation has been validated. See [RESEARCH_NOTES.md](../RESEARCH_NOTES.md) for organized notes, measured results and explicit corrections.

---

# Prior work (RL):
T. Fösel, P. Tighineanu, T. Weiss, and F. Marquardt,
Reinforcement learning with neural networks for quantum
feedback, Physical Review X 8, 031084 (2018).
M. Bukov, A. G. Day, D. Sels, P. Weinberg,
A. Polkovnikov, and P. Mehta, Reinforcement learning
in different phases of quantum control, Physical Review
X 8, 031086 (2018).
V. Sivak, A. Eickbusch, H. Liu, B. Royer, I. Tsiout-
sios, and M. Devoret, Model-free quantum control with
reinforcement learning, Physical Review X 12, 011059
(2022).
F. Metz and M. Bukov, Self-correcting quantum many-
body control using reinforcement learning with tensor net-
works, Nature Machine Intelligence 5, 780 (2023).
M. Bukov and F. Marquardt, Reinforcement learning for
quantum technology, arXiv:2601.18953 (2026).

# Prior work (Experimental application of RL agent):
L. Ding, M. Hays, Y. Sung, B. Kannan, J. An,
A. Di Paolo, A. H. Karamlou, T. M. Hazard, K. Azar,
D. K. Kim, et al., High-fidelity, frequency-flexible two-
qubit fluxonium gates with a transmon coupler, Physical
Review X 13, 031035 (2023).
[31] Y. Baum, M. Amico, S. Howell, M. Hush, M. Liuzzi,
P. Mundada, T. Merkh, A. R. Carvalho, and M. J.
Biercuk, Experimental deep reinforcement learning for
error-robust gate-set design on a superconducting quan-
tum computer, PRX Quantum 2, 040324 (2021).
[32] K. Reuer, J. Landgraf, T. Fösel, J. O’Sullivan, L. Bel-
trán, A. Akin, G. J. Norris, A. Remm, M. Kerschbaum,
J.-C. Besse, et al., Realizing a deep reinforcement learning
agent for real-time quantum feedback, Nature Communi-
cations 14, 7138 (2023)


# QC Notes
Knill-LaFlamme Conditions (https://people.eecs.berkeley.edu/~jswright/quantumcodingtheory24/scribe%20notes/lecture04.pdf)
Specifies what type of error can be corrected


## Noise (Noise.py)
Noise is sampled from a normal distribution 2D gaussian
shifts are independend from round to round --> no temporal correlation (can toggle)
Returns Measurment noise and mirror displacement noise sampled from normal distribution
NOTE DOWN DISTRIBUTION PARAMS USED IN FINAL EXPERIMENTS

Variance used in (https://arxiv.org/pdf/quant-ph/0008040)

# Gradual Noise Drift

# Unexpected noise changes

# Unseen Noise conditions



## Simulation (sim.py)
Based on https://spiral.imperial.ac.uk/server/api/core/bitstreams/848a611d-955e-4b4f-89ae-1a7539925da2/content
Depends on Noise and Decoderwhy
1. adds a displacement error sampled from Noise.py
2. Measure the syndrome (from -[L/2, L/2])
3. call decoder for the correction
4. apply x = x_error - correction
Error is logged when there is a odd multiple of L = sqrt(pi)
    if |0> at q = 2n/sqrt(pi) and |1> at q = (2n + 1) / sqrt(pi) then
   if shift = 2sqrt(pi) nothing is shown since it shifts one period
   if shfit = sqrt(pi) flips peaks
   if shift = r < L/2 corrected by syndrome "snaps back"

### Validating Simulator
One-round flip probability Fukui et al. Eq. 22 (https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054)




## GKP Encoding
Stores Qubit in a wave
|0> for even k sqrt(pi)
|1> for odd k sqrt(pi)

## 1 nearest - lattice decoder
Simple push back to the nearest lattive given a displacement
works for errors under sqrt(pi) / 2


# 2 Baysian Stats



## 3 LSTM


## 4
# RL Notes
Reinforcement learning control of Quantum Error Correction
https://arxiv.org/pdf/2511.08493v3
Experimenting is done in a Q-memeory algortihm where logical state is preserved over repetitive applciation of QEC cycles
They inject drift



# Stack
Harbor: https://www.harborframework.com/




# MultiAgentic RL Notes
Mutli-agentic enviornments (https://huggingface.co/learn/deep-rl-course/en/unit7/introduction-to-marl):

Two types of enviornments:
cooparative: maximize a common benifit (warehouse robots)

Competitive / Adversarial envs: maximize own agent's benifit and minimize the opponents




# Multiagentic apporach for QEC
https://www.alphaxiv.org/pdf/2509.03974
They propose real time adaptive QEC using multi-agent RL

Stages:
1. Code discovery
   - 3 agents complete encoding --> syndrome extraction --> recovery using a stationary noise model
   - each agent places elementary gates in a circuit construction enviornment
each agent recieves task specfic rewards
   - This paper does not fix the initial circuit, but lets 3 agents with different policies construct it
   -

# Policies
