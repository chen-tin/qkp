import numpy as np
import torch

from src import tools

# This module runs numerical experiments using data generated off
# of an imperfect measurement record from a real quantum device.


class Bayesian():

    # This class represenrs the Bayesian classifier designed to handle
    # the autocorrelations present in imperfect measurement data.

    def __init__(self, depth):

        # The depth attribute determines how many prior syndrome measurements to
        # condition on when calculating the Gaussian likelihood of a new syndrome
        # measurement. For depth > 0, this allows the model to leverage any
        # autocorrelation that may be present in the data.

        self.depth = depth

        # This is simply a list of the other attributes that will be used.

        self.transition_matrix = None
        self.gauss_means = None
        self.gauss_precs = None
        self.prior = None
        self.prev_synd = None

    def perfect_train(self, gamma_m, eta, error_rate, time_step):

        # This method calculates ideal transition matrices and Gaussian
        # precisions for the simulated data using the parameters provided.
        # The depth must be set to 0 when using this training scheme.

        assert self.depth == 0, "Depth must be 0 when using perfect_train."
        expon = torch.tensor(-2*error_rate*time_step)
        no_flip_prob = 0.5*(1 + torch.exp(expon))
        flip_prob = 0.5*(1 - torch.exp(expon))
        self.transition_matrix = torch.zeros([8, 8], dtype=torch.float32)
        for i in range(8):
            for j in range(8):
                diff = i ^ j
                if diff == 0:
                    prob = no_flip_prob**3
                elif ((diff == 4) | (diff == 2) | (diff == 1)):
                    prob = flip_prob*(no_flip_prob**2)
                elif diff == 7:
                    prob = flip_prob**3
                else:
                    prob = (flip_prob**2)*no_flip_prob
                self.transition_matrix[i, j] = prob
        gauss_prec = (2 * gamma_m * eta) / time_step
        self.joint_means = time_step * \
            torch.tensor([[1, 1], [1, -1], [-1, -1], [-1, 1],
                         [-1, 1], [-1, -1], [1, -1], [1, 1]])
        self.joint_precs = gauss_prec * \
            torch.eye(2).unsqueeze(0).expand(8, -1, -1)

    def train(self, train_data, labels):

        # This method constructs and stores the empirical transition matrix
        # and the mean/precision of the Gaussians using the passed training data.
        # The marg_precs and marg_means arrays are the parameters for the marginal
        # Gaussian distribution of the prior syndrome sequence that will be
        # conditioned on when depth > 0. They are construced by simply deleting
        # the most recent pair of syndrome measurements from the mean and covariance
        # arrays of the joint Gaussian.

        print("Training...")
        m_1 = train_data[:, :, 0]
        m_2 = train_data[:, :, 1]
        self.transition_matrix = tools.get_flip_probs(labels)
        (self.joint_means, joint_covs) = tools.get_gaussian(
            m_1, m_2, labels, self.depth)
        mask = torch.full_like(self.joint_means[0], True, dtype=torch.bool)
        mask[[self.depth, -1]] = False
        marg_covs = joint_covs[:, mask][:, :, mask]
        self.joint_precs = torch.inverse(joint_covs)
        self.marg_precs = torch.inverse(marg_covs)
        self.marg_means = self.joint_means[:, mask]

    def set_initial_state(self, prior, prev_syndromes):

        # This method sets the initial state of the classifier for when the
        # "classify" method is used. Subsequent calls to "classify" will update
        # the prior in response to new measurments. Note that the prior must have
        # shape (num_runs, 8), and the previous measurments must have the shape
        # (num_runs, depth, 2).

        self.prior = prior
        self.prev_synd = prev_syndromes

    def classify(self, measurements):

        # This method returns the posterior probability, with shape (num_runs, 8),
        # of each label given a batch of measurments with shape (num_runs, 2). The
        # internal state of the classifier is also updated appropriately, with the
        # prior being updated to the posterior and the previous syndromes being
        # updated to include the new measurements.

        m_1_batch = torch.cat(
            [self.prev_synd[:, 0:1], measurements[None, 0:1]], dim=1)
        m_2_batch = torch.cat(
            [self.prev_synd[:, 1:2], measurements[None, 1:2]], dim=1)
        likelihood = self.get_gaussian_outputs(m_1_batch, m_2_batch)
        unnorm_probs = torch.einsum(
            "al,lr->ar", self.prior.float(), self.transition_matrix.float()) * likelihood
        posteriors = unnorm_probs / \
            torch.sum(unnorm_probs, dim=1, keepdim=True)
        self.prior = posteriors
        if self.depth:
            self.prev_synd = torch.cat(
                [self.prev_synd[1:, :], measurements[None, :]])
        return posteriors[0]

    def correction_callback(self, correction):

        # This function is called after a correction is made to update
        # the prior distribution.

        self.prior = self.prior[:, torch.arange(8) ^ correction]

    def tracking_test(self, test_data, labels):

        # This method runs an error-tracking test of the Bayesian model on the passed
        # test data, printing the progress and eventual accuracy while also returning a
        # tensor of prior probabilities at each step. All runs are processed in parallel
        # For each run the prior probabililty is determineed by looking at the first label
        # of the trajectory.

        print("Testing...")
        m_1 = test_data[:, :, 0]
        m_2 = test_data[:, :, 1]
        (num_runs, num_steps) = m_1.shape
        outputs = torch.zeros([num_runs, 0])
        self.prior = torch.zeros([num_runs, 8])
        self.prior[torch.arange(num_runs), labels[:, max(
            self.depth - 1, 0)].type(torch.long)] = 1
        num_correct = torch.zeros([num_runs])
        for step in range(self.depth, num_steps):

            # For each step, the new prior is calculated by contracting the previous
            # prior with the transition matrix, then scaling by the measurement
            # probabilities.

            m_1_slice = m_1[:, step - self.depth:step + 1]
            m_2_slice = m_2[:, step - self.depth:step + 1]
            sample_probs = self.get_gaussian_outputs(m_1_slice, m_2_slice)
            unnorm_probs = torch.einsum(
                "al,lr->ar", self.prior.float(), self.transition_matrix.float()) * sample_probs
            self.prior = unnorm_probs / \
                torch.sum(unnorm_probs, dim=1, keepdim=True)
            predictions = torch.argmax(self.prior, dim=1)
            true_states = labels[:, step]
            fidelity = (predictions == true_states)
            num_correct = num_correct + fidelity
            outputs = torch.cat([outputs, predictions[:, None]], dim=1)
            print("\rStep {}/{}".format(step, num_steps), end="")

        acc = num_correct * 100 / num_steps
        print("\nAcc: {:.2f}%".format(torch.mean(acc)))
        print(f"Fidelity: {torch.mean(fidelity.float() * 100):.2f}%")
        return outputs

    def get_gaussian_outputs(self, m_1_batch, m_2_batch):

        # This method calculates the Gaussian likelihoods for the syndrome pair in
        # the most recent step, conditioned on a number of prior measurements determined
        # by the depth attribute. This means that the m_1_batch and m_2_batch must both
        # have shapes of (num_runs, depth + 1).

        (num_runs, num_steps) = m_1_batch.shape
        joint_batch = torch.cat([m_1_batch, m_2_batch], dim=1)
        marg_batch = torch.cat([m_1_batch[:, :-1], m_2_batch[:, :-1]], dim=1)

        joint_centered = joint_batch.reshape(
            [num_runs, 1, 2*(self.depth + 1)]) - self.joint_means.reshape([1, 8, 2*(self.depth + 1)])
        joint_expon = -0.5*torch.einsum("ali,lij,alj->al", joint_centered.float(
        ), self.joint_precs.float(), joint_centered.float())
        joint_output = joint_expon + 0.5*torch.slogdet(self.joint_precs)[1]

        if self.depth > 0:  # The marginal distribution only exists if depth > 0
            marg_batch = torch.cat(
                [m_1_batch[:, :-1], m_2_batch[:, :-1]], dim=1)
            marg_centered = marg_batch.reshape(
                [num_runs, 1, 2*self.depth]) - self.marg_means.reshape([1, 8, 2*self.depth])
            marg_expon = -0.5*torch.einsum("ali,lij,alj->al", marg_centered.float(
            ), self.marg_precs.float(), marg_centered.float())
            marg_output = marg_expon + 0.5*torch.slogdet(self.marg_precs)[1]
            log_output = joint_output - marg_output
        else:
            log_output = joint_output
        outputs = torch.exp(log_output)  # Only exponentiate at the end
        return outputs

    def load(self):

        # This function loads a previously trained model.

        self.transition_matrix = torch.tensor(np.load("transition_matrix.npy"))
        self.joint_means = torch.tensor(np.load("means.npy"))
        joint_covs = torch.tensor(np.load("cov.npy"))
        mask = torch.full_like(self.joint_means[0], True, dtype=torch.bool)
        mask[[self.depth, -1]] = False
        marg_covs = joint_covs[:, mask][:, :, mask]
        self.joint_precs = torch.inverse(joint_covs)
        self.marg_precs = torch.inverse(marg_covs)
        self.marg_means = self.joint_means[:, mask]

    def save(self):

        # This function saves the model.

        np.save("transition_matrix.npy", self.transition_matrix.numpy())
        np.save("means.npy", self.joint_means.numpy())
        np.save("cov.npy", torch.inverse(self.joint_precs).numpy())

    def reset(self):
        self.log_ = {}

    def log(self):
        return self.log_


class Bayesian_Corrector():

    # This class represents the error correction algorithm
    # that operates based on the predicitions on the Bayesian
    # model.

    def __init__(self, sys, bayes, num_trajs=1, time_step_sec=32e-9):

        # The sys and bayes attributes hold the specific system and
        # modle instances that will be used by the corrector.

        self.sys = sys
        self.init_state = sys.init_state_
        self.model = bayes
        self.time_step_sec = time_step_sec
        self.num_trajs = num_trajs
        prior = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0]]).repeat(num_trajs, 1)
        prior[:, self.init_state] = 1
        self.model.set_initial_state(prior)

    def simulate_spliced(self, num_steps, ignore_period=0, streak=None, inactive=False):

        # This function simulates a quantum device with the same imperfections
        # that were observed in the real experimental data.

        ignore = torch.zeros(self.num_trajs, dtype=torch.long)
        self.sys.reset(num_steps)
        self.net_corrections = np.zeros(
            [self.num_trajs, num_steps], dtype="int32")
        self.signals = np.zeros([self.num_trajs, num_steps, 2])
        self.sys_state = np.zeros([self.num_trajs, num_steps], dtype="int32")
        self.pred_state = np.zeros([self.num_trajs, num_steps], dtype="int32")

        for step in range(num_steps):
            current_signals = self.sys.measure()
            self.signals[:, step] = current_signals

            # Feed in RNN the re-calibrated signals
            active = ignore <= 0
            # pred, h_out, c_out, self.label_logits = self.evaluate(input, h_out, c_out)
            self.posteriors = self.model.classify(
                torch.Tensor(current_signals.float()), active)
            pred = np.argmax(self.posteriors.numpy(), axis=1)
            self.pred_state[:, step] = pred

            # After determining an error with confidence, applied the correction, update the metric state
            # to be the state without correction, and record the correction into the bitfip counter
			
            if step >= 1:
                error = pred ^ init_state
                do_corr = self.deter_corr(step, streak)
                ignore[do_corr & error != 0] = ignore_period
                error[~do_corr | ~active.numpy()] = 0
                if not inactive:
                    self.sys.apply(torch.tensor(error))
                    self.net_corrections[:,
                                         step] = error ^ self.net_corrections[:, step - 1]
                self.model.correction_callback(torch.tensor(error))

            ignore = ignore - 1
            self.sys_state[:, step] = self.sys.states # Update the current system state
        self.baseline_state = self.sys_state ^ self.net_corrections

    def deter_corr(self, step, streak):
		
        # This function determines whether a correction should
        # be applied based on the output of the model and the
        # passed values of the hyperparameters.

        start = max(0, step - streak + 1)
        stop = step + 1
        pred_window = self.pred_state[:, start:stop]
        do_corr = (pred_window == pred_window[:, 0:1]).all(-1)
        return do_corr


if __name__ == "__main__":

    # The following code runs numerical experiments using the Bayesian
    # model and corrector.

    active = True # Determines if the model should correct errors (True) or track them (False) 
    t1_list = [True, False] # Determines if excitations and decays (False) or only decay (True) should occur
    T_list = [120]  # Time of runs
    gamma_list = [0.04]  # Error rates
    system_list = ["D"]  # Which group; of imperfections should be used\
    system_types = {"A": (False, False, True, 0, 1, None), "B": (True, False, True, 0, 1, None),
                    "C": (True, True, True, 0, 35, None),  "D": (True, True, True, 0, 35, lambda traj: 0.4*traj / 10000)}

    for sys_type in system_list:
        (corr, ring, reson, ignore, streak,
         drift_func) = system_types[sys_type]
        for gamma in gamma_list:
            for t1 in t1_list:
                init_state = 7
                train_system = tools.Real_Simulated(gamma*1e6, init_state=init_state, autocorr=corr,
                                                    ring=ring, reson_eql=reson, only_t1=False, drift_func=None, num_trajs=10000)
                test_system = tools.Real_Simulated(gamma*1e6, init_state=init_state, autocorr=corr,
                                                   ring=ring, reson_eql=reson, only_t1=t1, drift_func=drift_func, num_trajs=10000)
                depth = 3 if corr else 0
                bayes = Bayesian(depth)
                (train_data, train_labels) = train_system.generate_data(30000, 120)
                bayes.train(train_data, train_labels)
                for T in T_list:
                    print(f"System: {sys_type} | Time: {T} | Error: {gamma}")
                    num_steps = int(T / 0.032)
                    if active:
                        states = np.zeros([0, num_steps])
                        for traj_set in range(20):
                            print(f"Trajectory {traj_set + 1}/{20}", end="\r")
                            np.random.seed(traj_set)
                            torch.manual_seed(traj_set)
                            corrector = Bayesian_Corrector(
                                test_system, bayes, 10000)
                            corrector.simulate_spliced(
                                num_steps, ignore_period=ignore, streak=streak)
                        print("")

                    else:
                        measurements = torch.zeros([0, num_steps, 2])
                        states = torch.zeros([0, num_steps], dtype=torch.long)
                        for traj_set in range(1):
                            print(traj_set, end="\r")
                            np.random.seed(traj_set)
                            torch.manual_seed(traj_set)
                            (data, labels) = test_system.generate_data(30000, T)
                            measurements = torch.cat(
                                [measurements, data], axis=0)
                            states = torch.cat([states, labels], axis=0)
                        print("")
                        outputs = bayes.tracking_test(measurements, states)
