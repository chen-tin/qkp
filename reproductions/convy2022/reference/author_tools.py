import math

import torch
import numpy as np

# This module holds the classes and functions used to generate and process
# the quantum data.


class Real_Simulated():
    
    # This class simulates a quantum memory system using features found
    # in experimental data. Specifically, it incorporates autocorrelation
    # within the Gaussian noise, ring-up and ring-down patterns after a state
    # transition, and different variances across the two resonators. Each of
    # these features can be removed from the simulated data so that it more closely
    # resembles a set of idealized trajectories.

    def __init__(self, error_rate_sec, time_step_sec=32e-9, gamma_m_sec=4.7e6, eta=0.5,
                 init_state=0, only_t1=False, autocorr=False, ring=False, reson_eql=True,
                 scale_by_dt=False, drift_func=None, num_trajs=1, log=True):

        # The class is initialized with a time step, an error rate, a measurement
        # strength and an eta value, along with an initial state used for active correction.
        # The different non-ideal features are turned on by default and can be individually
        # toggled when an instance is created. The "autocorr" option determines whether
        # sequential noise values are correlated, the "ring" option controls whether the
        # mean trajectory contains a gradual "ringing" pattern after bit flips, and the
        # "reson_eql" option determines whether the two resonators will have different
        # variances. If the "only_t1" option is True then only decay transitions are allowed.
        # The "drift_func" is a function [f(traj) = shift] which takes in a trajectory
        # index and outputs how much the syndrome means should be shifted. The "scale_by_dt"
        # option determines whether the simulator should return the raw signal (True) or the
        # averaged signal (False).

        self.init_state_ = init_state
        self.gamma = error_rate_sec*1e-6
        self.step_size = time_step_sec*1e6
        self.measure_size = 2
        if drift_func is None:
            self.drift_func = lambda traj: 0*traj
        else:
            self.drift_func = drift_func
        self.autocorr = autocorr
        self.ring = ring
        self.reson_eql = reson_eql
        self.t1 = only_t1
        self.scale = time_step_sec if scale_by_dt else 1
        self.variance = 1 / (2 * gamma_m_sec * eta * time_step_sec)
        self.covs = torch.tensor(self.get_covs(self.variance))
        self.max_cond = self.covs.shape[1] - 1
        self.state_means = - \
            torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1],
                         [-1, 1], [-1, -1], [1, -1], [1, 1]])
        self.flip_means = self.get_flip_means()
        self.active_means = self.get_flip_means(active=True)
        self.flip_index = torch.tensor([-1, 2, 1, 0, 0, 1, 2, -1])
        self.allowed_t1 = torch.tensor(np.unpackbits(np.arange(8, dtype="uint8")[
                                       :, None], 1, 8).astype("int")[:, 5:]).bool()
        self.log = log
        self.num_trajs = num_trajs
        self.reset()
        self.scale_by_dt = scale_by_dt

    def get_covs(self, variance):

        # This function retrieves the correlation coefficents that were observed
        # in the data and scales them to the desired variance. If "autocorr" is
        # set to False then the covariance matrices are simply equal to the identity.

        # Observed ratio between resonators
        ratio_2_1 = 0.574 if not self.reson_eql else 1
        if self.autocorr:
            corr = np.load("corr.npy")
            corr[1] = corr[0]  # just use resonator0's autocorrelations
            cov = corr * variance
            cov[1] *= ratio_2_1
        else:
            cov = np.array([[[1.0]], [[1.0]]]) * variance
            cov[1] *= ratio_2_1
        return cov

    def get_flip_means(self, active=False):

        # This function retreives the ring-up/down pattern observed in the
        # data. The curves have been scaled and shifted so that they lie between
        # +1 and -1, though their relative proportions are left unchanged. If the
        # "ring" option is set to False then these means are not used and the signal
        # instead jumps instantaneously between +1 and -1. If "active" is set to
        # True then the returned tensor will still contain the means at the first
        # step of the ring pattern even if "ring" was set to False. This is needed
        # for the active correction code.

        mean_dict = np.load("src/data/flip_means.npz")
        if self.ring:
            flip_means = torch.zeros(8, 3, 94, 2)
        elif active:
            flip_means = torch.zeros(8, 3, 1, 2)
        else:
            flip_means = torch.zeros(8, 3, 0, 2)
        for state in range(8):
            for flip in range(3):
                if self.ring:
                    mean_array = np.stack(
                        [mean_dict[f"{state}_{flip + 1}_1"], mean_dict[f"{state}_{flip + 1}_2"]], axis=-1)
                    flip_means[state, flip] = torch.tensor(mean_array)
                else:
                    if active:
                        flip_means[state,
                                   flip] = self.state_means[state:state + 1]
                    else:
                        flip_means[state, flip] = torch.zeros([0, 2])
        return flip_means

    def reset(self, num_steps=None):

        # This function resets the system and clears its memory. Additionally,
        # the number of trajs used in the active correction must be set here. The
        # "num_steps" argumement allows you to specify in advance how long the runs
        # will be, so that the log_ tensors can be preallocated for faster run time.

        self.num_steps = num_steps if num_steps is not None else 0
        self.step = 0
        self.states = torch.full(
            [self.num_trajs], self.init_state_, dtype=torch.long)
        self.prev_states = self.states.clone()
        self.mean_pos = torch.zeros(self.num_trajs, dtype=torch.long)
        if self.log:
            self.log_ = {}
            self.log_['states'] = torch.zeros(
                self.num_trajs, self.num_steps).long()
            self.log_['errors'] = torch.zeros(
                self.num_trajs, self.num_steps).long()
            self.log_['measurements'] = torch.zeros(
                self.num_trajs, self.num_steps, self.measure_size)
        self.prev_noise = torch.zeros(self.num_trajs, 0, 2)
        self.flip_size = 94 if self.ring else 0  # Ring pattern has 94 steps

    def update_logs(self, errors, syndromes):

        # This function updates the system logs with the appropriate data.

        if self.num_steps:
            self.log_['states'][:, self.step] = self.states
            self.log_['errors'][:, self.step] = errors
            self.log_['measurements'][:, self.step] = syndromes
        else:
            self.log_['states'] = torch.cat(
                (self.log_['states'], self.states[:, None]), dim=1)
            self.log_['errors'] = torch.cat(
                (self.log_['errors'], errors[:, None]), dim=1)
            self.log_['measurements'] = torch.cat(
                (self.log_['measurements'], syndromes[:, None]), dim=1)

    def measure(self):

        # This function advances the state of the system by one time step and returns
        # the values of the syndromes. Specifically, a new state is selected which may
        # or may not match the previous state depending on whether an error occured during
        # the time step, and then this new state is used to generate the syndromes after
        # being combined with (possibly correlated) Gaussian noise.

        new_states = self.get_next_states(self.step_size, self.states)
        errors = self.flip_index[self.states ^ new_states]
        self.prev_states[errors != -1] = self.states[errors != -1]
        self.prev_states[self.mean_pos == 0] = self.states[self.mean_pos == 0]
        self.mean_pos[(errors != -1)] = self.flip_size
        self.states = new_states

        prev_error = self.flip_index[self.states ^ self.prev_states]
        means = self.active_means[self.prev_states, prev_error, -self.mean_pos]
        noise = self.get_gaussian_noise(
            self.num_trajs, 1, self.prev_noise)[:, 0]
        syndromes = noise + means + \
            self.drift_func(torch.arange(self.num_trajs))[:, None]
        self.mean_pos = torch.maximum(torch.tensor([0]), self.mean_pos - 1)

        if self.log:
            self.update_logs(errors, syndromes)
        self.prev_noise = torch.cat([self.prev_noise, noise[:, None]], dim=1)[
            :, -self.max_cond:][:, :self.max_cond]
        self.step += 1
        return syndromes * self.scale

    def apply(self, flips):

        # This function applies a set of bit flips specified by the binary form of an
        # integer from 0 to 7. While all possible combinations of three-qubit flips are
        # allowed, the experimental data for the flips will be taken from the single-qubit
        # flip that matches the syndromes.

        new_states = self.states ^ flips
        errors = self.flip_index[flips]
        self.prev_states[errors != -1] = self.states[errors != -1]
        self.prev_states[(self.mean_pos == 0)
                         ] = self.states[(self.mean_pos == 0)]
        self.mean_pos[(errors != -1)] = self.flip_size
        self.states = new_states

    def get_next_states(self, time, curr_states):

        # This function is used to determine whether an error has occured
        # within a given time increment. Using the error rate, a sample is
        # taken from an exponential distribution and compared to the step size.
        # If an error occurs, the affected qubit iss chosen at random. If "only_t1"
        # is set to True, then any proposed excitation is ignored.

        if self.gamma > 0:
            flip_time = torch.zeros(
                [self.num_trajs]).exponential_(3*self.gamma)
        else:
            return curr_states
        flipped = flip_time < time
        num_flipped = torch.count_nonzero(flipped)
        flip_index = torch.randint(3, [num_flipped])
        flips = torch.tensor([4, 2, 1])[flip_index]
        if self.t1:
            valid_flips = self.allowed_t1[curr_states[flipped], flip_index]
            flips[~valid_flips] = 0
        new_states = curr_states.clone()
        new_states[flipped] = flips ^ curr_states[flipped]
        return new_states

    def get_flip_times(self, num_traj, T):

        # This function generates a set of bit-flip times by sampling
        # from an exponential distribution. The "gamma" argument sets
        # the average number of errors per microsecond, while the "T"
        # argument gives the period during which errors should be
        # generated. Since there will be a different number of errors
        # per trajectory, the rows the array are padded with values of
        # -1 to ensure uniform length. The returned time array has a
        # shape of [num_traj, max_error_num], where "max_error_num" is
        # the largest number of errors generated in a single trajectory.

        if self.gamma > 0:
            average_flip_number = math.ceil(3*self.gamma * T)
            flip_intervals = torch.zeros(
                [num_traj, 3*average_flip_number]).exponential_(3*self.gamma)
            flip_times = torch.cumsum(flip_intervals, dim=1)
            flip_times[flip_times > T] = -1
            trunc_mask = ~torch.all(flip_times == -1, dim=0)
            trunc_flip_times = flip_times[:, trunc_mask]
        else:
            trunc_flip_times = torch.zeros([num_traj, 0])
        return trunc_flip_times

    def get_gaussian_noise(self, num_traj, num_steps, init_cond=None):

        # This function generates Gaussian noise for the system. After specifying
        # the number of trajectories and steps, the "init_cond" option can be
        # used to specify a set of prior noise values (of mean zero) to condition
        # on. If no values are given, or fewer than the total correlation length
        # are provided, then the marginal distributions will be sampled first until
        # a sufficient number of samples are avaiable to use the full conditional
        # distribution. The array passed to "init_cond" must have shape
        # num_traj x d x 2, where the depth "d" can be any number.

        noise = torch.zeros([num_traj, num_steps, 2], dtype=torch.float64)
        if init_cond is None:
            prev_noise = torch.zeros([num_traj, 0, 2])
        else:
            prev_noise = init_cond[:, -self.max_cond:][:, :self.max_cond]
        for step in range(num_steps):
            num_cond = prev_noise.shape[1]
            marg_covs = self.covs[:, :num_cond + 1, :num_cond + 1]
            prec_matrices = torch.inverse(marg_covs)
            prec = prec_matrices[:, -1, -1]
            cond_corr_coeff = prec_matrices[:, -1, :-1].T
            cond_means = (-1/prec)*(cond_corr_coeff[None]*prev_noise).sum(1)
            samples = torch.normal(cond_means, 1/prec**0.5)
            noise[:, step] = samples
            prev_noise = torch.cat([prev_noise, samples[:, None]], dim=1)
            prev_noise = prev_noise[:, -self.max_cond:][:, :self.max_cond]
        return noise

    def generate_data(self, num_traj, T, random=True, init_state=None):

        # This function generates a bulk set of trajectories based on the
        # specified paramters. If "random" is set to False then the errors
        # and initial state (but not the noise) will be generated identically
        # across all trajectories. This allows for the mean of the data to be
        # properly visualized.

        flip_times = self.get_flip_times(num_traj, T)
        num_steps = int(T / self.step_size)
        means = torch.zeros([num_traj, num_steps, 2])
        states = torch.zeros([num_traj, num_steps]).long()
        permute = [4, 2, 1]  # Bit flips for the first, second, and third qubit
        for traj in range(num_traj):
            print(f"{traj+1}/{num_traj}", end="\r")
            if init_state is None:
                # Starting state chosen at random or set to 0
                curr_state = torch.randint(8, [1]).item() if random else 0
            else:
                curr_state = init_state
            start = 0
            error_index = traj if random else 0
            error_steps = [int(time / self.step_size)
                           for time in flip_times[error_index][flip_times[error_index] != -1]]
            for end in error_steps + [num_steps]:
                if start > end:  # Handles the case when two error sequences overlap
                    start = end
                means[traj, start:end] = self.state_means[curr_state]
                states[traj, start:end] = curr_state
                start = end
                # Idenity of error is chosen at random or set to 1
                flip = torch.randint(low=0, high=3, size=[
                                     1]).item() if random else 1
                # Excitations are simply ignored if only_t1 = True
                if not self.t1 or self.allowed_t1[curr_state, flip]:
                    flip_synd = self.flip_means[curr_state, flip]
                    # State is updated based on flip
                    curr_state ^= permute[flip]
                    flip_end_step = min(end + flip_synd.shape[0], num_steps)
                    means[traj, end:flip_end_step] = flip_synd[:flip_end_step - end]
                    states[traj, end:flip_end_step] = curr_state
                    start = flip_end_step
        print("")
        noise = self.get_gaussian_noise(num_traj, num_steps)
        measurements = means + noise + \
            self.drift_func(torch.arange(num_traj))[:, None, None]
        return (measurements * self.scale, states)


class Perfect_Simulated():

    # This class simulates a quantum memory system using artifical data without any imperfections.

    def __init__(self, error_rate_sec, time_step_sec, gamma_m_sec, eta, init_state=0):
        # We use time_step_sec = 32e-9, gamma_m_sec = 2.5e6, and eta = 0.5

        # This class is initialized using parameters in units of seconds, but works
        # internally with time in microseconds.

        self.init_state_ = init_state
        self.step = time_step_sec*1e6
        self.gamma = error_rate_sec*1e-6
        self.variance = time_step_sec / (2 * gamma_m_sec * eta)
        self.state_means = time_step_sec * \
            torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1],
                         [-1, 1], [-1, -1], [1, -1], [1, 1]])

    def generate_exact_data(self, num_traj, T):

        # This function generates "num_traj" simulated trajectories, with a length in
        # microseconds specified by the "T" argument and an error rate specified by
        # "gamma". The generation pipeline starts with a set of error times, which
        # then get converted into discrete index values using the time step value.
        # A single-qubit errors is randomly sampled for each index, and these are
        # then inserted into a blank array with shape [num_traj, num_steps]. The
        # states are then generated by performing a cumulative bitwise xor operation
        # on this array across the steps. The states are then used to generate mean
        # syndrome values each step, which are then converted into sampled measurement
        # values by adding Gaussian noise with variance dictated by the gamma_m, eta,
        # and the time step.

        num_steps = int(T / self.step)
        if self.gamma:
            step_errors = torch.distributions.poisson.Poisson(
                self.step*self.gamma).sample([num_traj, num_steps, 3]).int()
        else:
            step_errors = torch.zeros([num_traj, num_steps, 3]).int()

        max_errors = step_errors.reshape([-1, 3]).max(0)[0]
        fracs = torch.zeros([num_traj, num_steps, 2])
        for i in range(max_errors[0] + 1):
            mask_1 = (step_errors[:, :, 0] == i)
            for j in range(max_errors[1] + 1):
                mask_2 = (step_errors[:, :, 1] == j)
                for k in range(max_errors[2] + 1):
                    mask_3 = (step_errors[:, :, 2] == k)
                    mask = mask_1 & mask_2 & mask_3
                    num = torch.count_nonzero(mask)
                    if num:
                        mean_fracs = get_partial_means(num, [i, j, k])
                        fracs[mask] = mean_fracs.float()

        single_errors = step_errors % 2
        net_errors = 4*single_errors[:, :, 0] ^ 2 * \
            single_errors[:, :, 1] ^ 1*single_errors[:, :, 2]
        states = torch.zeros([num_traj, num_steps + 1]).long()
        for step in range(num_steps):
            states[:, step + 1] = states[:, step] ^ net_errors[:, step]

        traj_synds = self.state_means[states]
        traj_means = traj_synds[:, :-1] * fracs
        traj_means = traj_means[:, :]
        initial = states[:, 0]
        states = states[:, 1:]
        noise = torch.normal(mean=0, std=self.variance **
                             0.5, size=traj_means.shape)
        trajs = traj_means + noise
        return (trajs, initial, states)

    def set_gamma(self, gamma):
        self.gamma = gamma


def get_partial_means(num_samples, error_triplet):

    # This function samples a possible mean value for the
    # step based on the number of errors that occur on each
    # qubit. The position of the errors is first randomly
    # sampled, and then assigned to the different syndromes.
    # The differences between the error positions and then be
    # added/subtraced to compute the mean syndrome value.

    (num_1, num_2, num_3) = error_triplet
    positions = np.random.uniform(size=[num_samples, num_1 + num_2 + num_3])
    synd_1 = np.concatenate(
        [positions[:, :num_1], positions[:, num_1:num_1 + num_2]], axis=1)
    synd_2 = np.concatenate(
        [positions[:, num_1:num_1 + num_2], positions[:, num_1 + num_2:]], axis=1)
    synd_1_sorted = np.sort(synd_1, axis=1)
    synd_2_sorted = np.sort(synd_2, axis=1)
    points_1 = np.concatenate(
        [np.zeros([num_samples, 1]), synd_1_sorted, np.ones([num_samples, 1])], axis=1)
    points_2 = np.concatenate(
        [np.zeros([num_samples, 1]), synd_2_sorted, np.ones([num_samples, 1])], axis=1)
    segments_1 = points_1[:, 1:] - points_1[:, :-1]
    segments_2 = points_2[:, 1:] - points_2[:, :-1]
    segments_1[:, 1::2] = -segments_1[:, 1::2]
    segments_2[:, 1::2] = -segments_2[:, 1::2]
    partial_means_1 = segments_1.sum(1)
    partial_means_2 = segments_2.sum(1)
    partial_means = np.stack([partial_means_1, partial_means_2], axis=1)
    return torch.tensor(partial_means)


def torch_cov(X):

    # Computes the covariance from a data tensor X.

    num_samples = X.shape[-2]
    mean = torch.mean(X, dim=-2, keepdim=True)
    centered_X = X - mean
    cov = (centered_X.transpose(-1, -2)  @ centered_X) / (num_samples - 1)
    return cov


def count_flips(labels):

    # This function counts the number of times label i flips to label j
    # when i != j.

    counts = {(i, j): 0 for i in range(8) for j in range(8)}
    for i in range(8):
        start_check = (labels[:, :-1] == i)
        for j in range(8):
            if i != j:
                end_check = (labels[:, 1:] == j)
                combined = start_check & end_check
                counts[(i, j)] = torch.sum(combined)
    return counts


def get_flip_probs(labels):

    # This function calculates the probability for each of the 64 different
    # transitions by simply counting the number of said transitions.

    total_counts = torch.unique(labels.flatten(), return_counts=True)[1]
    flip_counts = count_flips(labels)
    label_prob_matrix = []
    for i in range(8):
        label_counts = torch.tensor([flip_counts[(i, j)] for j in range(8)])
        label_counts[i] = total_counts[i] - \
            torch.sum(label_counts) - (1/8)*labels.shape[0]
        label_probs = label_counts / torch.sum(label_counts)
        label_prob_matrix.append(label_probs)
    return torch.stack(label_prob_matrix)


def get_gaussian(m_1, m_2, labels, depth):

    # This function calculates the mean and covariance of the data for
    # each label, which can then be used to parameterize the Gaussian
    # likelihood functions.

    num_steps = m_1.shape[1]
    means = []
    covs = []
    m_1_offset = [m_1[:, i:num_steps - (depth - i)] for i in range(depth + 1)]
    m_2_offset = [m_2[:, i:num_steps - (depth - i)] for i in range(depth + 1)]
    offset = torch.stack(m_1_offset + m_2_offset, dim=2)
    flat_offset = offset.reshape([-1, 2*(depth + 1)])
    for label in range(8):
        mask = (labels[:, depth:] == label).flatten()
        selected = flat_offset[mask]
        mean = torch.mean(selected, dim=0)
        cov = torch_cov(selected)
        means.append(mean)
        covs.append(cov)
    return (torch.stack(means), torch.stack(covs))
