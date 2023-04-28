# https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e



# a look at the notation
# T - length of the observation sequence.
# N - number of latent (hidden) states.
# M - number of observables.
# Q = {q₀, q₁, …} - hidden states.
# V = {0, 1, …, M — 1} - set of possible observations.
# A - state transition matrix.
# B - emission probability matrix.
# π- initial state probability distribution.
# O - observation sequence.
# X = (x₀, x₁, …), x_t ∈ Q - hidden state sequence.

# Having that set defined, we can calculate the probability of 
# any state and observation using the matrices:
# A = {a_ij} — begin an transition matrix.
# B = {b_j(k)} — being an emission matrix.

# The probabilities associated with transition and observation 
# (emission) are:
# see Resources/Pics/np_pd_img1.jpg

# The model is therefore defined as a collection:
# see Resources/Pics/np_pd_img2.jpg


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# essentially creating a dictionary that ensures the values behave 
# correctly and importantly enforces certain rules
class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs  = probabilities.values()

        # The probabilities must match the states.
        assert len(states) == len(probs)
        # The states must be unique."
        assert len(states) == len(set(states))
        # Probabilities must sum up to 1.
        assert abs(sum(probs) - 1.0) < 1e-12
        # Probabilities must be numbers from [0, 1] interval.
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs)
        
        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: 
            probabilities[x], self.states))).reshape(1, -1)
        
    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    # comparison (__eq__) - to know if any two PV's are equal
    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    # __getitem__ to enable selecting value by the key
    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    # element-wise multiplication of two PV’s or multiplication 
    # with a scalar (__mul__ and __rmul__)
    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    # dot product (__matmul__) - to perform vector-matrix multiplication
    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    # division by number (__truediv__)
    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    # argmax to find for which state the probability is the highest
    def argmax(self):
        index = self.values.argmax()
        return self.states[index]



# ex1
a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
a2 = ProbabilityVector({'sun': 0.1, 'rain': 0.9})
print(a1.df)
print(a2.df)

print("Comparison:", a1 == a2)
print("Element-wise multiplication:", a1 * a2)
print("Argmax:", a1.argmax())
print("Getitem:", a1['rain'])
# OUTPUT
# >>>              rain  sun
#     probability   0.7  0.3
#                  rain  sun
#     probability   0.9  0.1
# >>> Comparison: False
# >>> Element-wise multiplication: [[0.63 0.03]]
# >>> Argmax: rain
# >>> Getitem: 0.7


# Probability Matrix
# Formally, the A and B matrices must be row-stochastic, meaning 
# that the values of every row must sum up to 1. We can, therefore, 
# define our PM by stacking several PV's, which we have constructed 
# in a way to guarantee this constraint
class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):

        # The numebr of input probability vector must be greater than one.
        assert len(prob_vec_dict) > 1
        # All internal states of all the vectors must be indentical.
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1
        # All observables must be unique.
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys()))

        self.states      = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values      = np.stack([prob_vec_dict[x].values \
                            for x in self.states]).squeeze() 

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables))/ (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array: 
                    np.ndarray, 
                    states: list, 
                    observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) \
                    for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values, 
                            columns=self.observables, 
                            index=self.states
                            )

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)


# ex2
a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
a2 = ProbabilityVector({'rain': 0.6, 'sun': 0.4})
A  = ProbabilityMatrix({'hot': a1, 'cold': a2})

print(A)
print(A.df)
# >>> PM (2, 2) states: ['cold', 'hot'] -> obs: ['rain', 'sun'].
# >>>      rain  sun
#    cold   0.6  0.4
#    hot    0.7  0.3

b1 = ProbabilityVector({'0S': 0.1, '1M': 0.4, '2L': 0.5})
b2 = ProbabilityVector({'0S': 0.7, '1M': 0.2, '2L': 0.1})
B =  ProbabilityMatrix({'0H': b1, '1C': b2})

print(B)
print(B.df)
# >>> PM (2, 3) states: ['0H', '1C'] -> obs: ['0S', '1M', '2L'].
# >>>       0S   1M   2L
#      0H  0.1  0.4  0.5
#      1C  0.7  0.2  0.1

P = ProbabilityMatrix.initialize(list('abcd'), list('xyz'))
print('Dot product:', a1 @ A)
print('Initialization:', P)
print(P.df)
# >>> Dot product: [[0.63 0.37]]
# >>> Initialization: PM (4, 3) 
#     states: ['a', 'b', 'c', 'd'] -> obs: ['x', 'y', 'z'].
# >>>          x         y         z
#    a  0.323803  0.327106  0.349091
#    b  0.318166  0.326356  0.355478
#    c  0.311833  0.347983  0.340185
#    d  0.337223  0.316850  0.345927


from itertools import product
from functools import reduce


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
    
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)
    
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))
    
    def score(self, observations: list) -> float:
        def mul(x, y): return x * y
        
        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))
            
            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]
            
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


# ex3
a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})

A = ProbabilityMatrix({'1H': a1, '2C': a2})
B = ProbabilityMatrix({'1H': b1, '2C': b2})
pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

hmc = HiddenMarkovChain(A, B, pi)
observations = ['1S', '2M', '3L', '2M', '1S']

print("Score for {} is {:f}.".format(observations, hmc.score(observations)))
# >>> Score for ['1S', '2M', '3L', '2M', '1S'] is 0.003482.


all_possible_observations = {'1S', '2M', '3L'}
chain_length = 3  # any int > 0
all_observation_chains = list(product(*(all_possible_observations,) * chain_length))
all_possible_scores = list(map(lambda obs: hmc.score(obs), all_observation_chains))
print("All possible scores added: {}.".format(sum(all_possible_scores)))
# >>> All possible scores added: 1.0.


class HiddenMarkovChain_FP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) 
                         @ self.T.values) * self.E[observations[t]].T
        return alphas
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())
    

# ex4
hmc_fp = HiddenMarkovChain_FP(A, B, pi)

observations = ['1S', '2M', '3L', '2M', '1S']
print("Score for {} is {:f}.".format(observations, hmc_fp.score(observations)))
# >>> All possible scores added: 1.0.


class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        # The chain needs to be a non-negative number.
        assert length >= 0
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)
        
        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())
        
        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
        
        return o_history, s_history


# ex5
# chart1
# An example of a Markov process. The states and the observable 
# sequences are shown
hmc_s = HiddenMarkovChain_Simulation(A, B, pi)
observation_hist, states_hist = hmc_s.run(100)  # length = 100
stats = pd.DataFrame({
    'observations': observation_hist,
    'states': states_hist}).applymap(lambda x: int(x[0])).plot()
plt.show()

# ex6
# chart2
# Convergence of the probabilities against the length of the chain
hmc_s = HiddenMarkovChain_Simulation(A, B, pi)

stats = {}
for length in np.logspace(1, 5, 40).astype(int):
    observation_hist, states_hist = hmc_s.run(length)
    stats[length] = pd.DataFrame({
        'observations': observation_hist,
        'states': states_hist}).applymap(lambda x: int(x[0]))

S = np.array(list(map(lambda x: 
        x['states'].value_counts().to_numpy() / len(x), stats.values())))

plt.semilogx(np.logspace(1, 5, 40).astype(int), S)
plt.xlabel('Chain length T')
plt.ylabel('Probability')
plt.title('Converging probabilities.')
plt.legend(['1H', '2C'])
plt.show()


class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                         * self.E[observations[t]].T
        return alphas
    
    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas
    
    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))


# ex7
np.random.seed(42)

a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})
b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5}) 
b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})
A  = ProbabilityMatrix({'1H': a1, '2C': a2})
B  = ProbabilityMatrix({'1H': b1, '2C': b2})
pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

hmc = HiddenMarkovChain_Uncover(A, B, pi)

observed_sequence, latent_sequence = hmc.run(5)
uncovered_sequence = hmc.uncover(observed_sequence)
# |                    | 0   | 1   | 2   | 3   | 4   | 5   |
# |:------------------:|:----|:----|:----|:----|:----|:----|
# | observed sequence  | 3L  | 3M  | 1S  | 3L  | 3L  | 3L  |
# | latent sequence    | 1H  | 2C  | 1H  | 1H  | 2C  | 1H  |
# | uncovered sequence | 1H  | 1H  | 2C  | 1H  | 1H  | 1H  |


# evaluates the likelihood of different latent sequences resulting 
# in our observation sequence
all_possible_states = {'1H', '2C'}
chain_length = 6  # any int > 0
all_states_chains = list(product(*(all_possible_states,) * chain_length))

df = pd.DataFrame(all_states_chains)
dfp = pd.DataFrame()

for i in range(chain_length):
    dfp['p' + str(i)] = df.apply(lambda x: 
        hmc.E.df.loc[x[i], observed_sequence[i]], axis=1)

scores = dfp.sum(axis=1).sort_values(ascending=False)
df = df.iloc[scores.index]
df['score'] = scores
df.head(10).reset_index()
# |    index | 0   | 1   | 2   | 3   | 4   | 5   |   score |
# |:--------:|:----|:----|:----|:----|:----|:----|--------:|
# |        8 | 1H  | 1H  | 2C  | 1H  | 1H  | 1H  |     3.1 |
# |       24 | 1H  | 2C  | 2C  | 1H  | 1H  | 1H  |     2.9 |
# |       40 | 2C  | 1H  | 2C  | 1H  | 1H  | 1H  |     2.7 |
# |       12 | 1H  | 1H  | 2C  | 2C  | 1H  | 1H  |     2.7 |
# |       10 | 1H  | 1H  | 2C  | 1H  | 2C  | 1H  |     2.7 |
# |        9 | 1H  | 1H  | 2C  | 1H  | 1H  | 2C  |     2.7 |
# |       25 | 1H  | 2C  | 2C  | 1H  | 1H  | 2C  |     2.5 |
# |        0 | 1H  | 1H  | 1H  | 1H  | 1H  | 1H  |     2.5 |
# |       26 | 1H  | 2C  | 2C  | 1H  | 2C  | 1H  |     2.5 |
# |       28 | 1H  | 2C  | 2C  | 2C  | 1H  | 1H  |     2.5 |


dfc = df.copy().reset_index()
for i in range(chain_length):
    dfc = dfc[dfc[i] == latent_sequence[i]]
    
dfc
# |   index | 0   | 1   | 2   | 3   | 4   | 5   |   score |
# |:-------:|:----|:----|:----|:----|:----|:----|--------:|
# |      18 | 1H  | 2C  | 1H  | 1H  | 2C  | 1H  |     1.9 |


# Training the model
# Expanding the class
# Here, our starting point will be the HiddenMarkovModel_Uncover 
# that we have defined earlier. We will add new methods to train 
# it.
# The model training can be summarized as follows:

# Initialize A, B and π.
# Calculate γ(i, j).
# Update the model’s A, B and π.
# We repeat the 2. and 3. until the score p(O|λ) no longer increases.

class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas


class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)

    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self.layer._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score 

        L = len(alpha)
        obs_idx = [self.layer.observables.index(x) for x in observations]
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0

        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)
            
        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score


# ex8
np.random.seed(42)

observations = ['3L', '2M', '1S', '3L', '3L', '3L']

states = ['1H', '2C']
observables = ['1S', '2M', '3L']

hml = HiddenMarkovLayer.initialize(states, observables)
hmm = HiddenMarkovModel(hml)

# 25 epochs
hmm.train(observations, 25)


# |    | 0   | 1   | 2   | 3   | 4   | 5   |
# |---:|:----|:----|:----|:----|:----|:----|
# |  0 | 3L  | 2M  | 1S  | 3L  | 3L  | 3L  |
RUNS = 100000
T = 5

chains = RUNS * [0]
for i in range(len(chains)):
    chain = hmm.layer.run(T)[0]
    chains[i] = '-'.join(chain)


# If we have truly trained the model, we should see a strong 
# tendency for it to generate us sequences that resemble the 
# one we require
# df = pd.DataFrame(pd.Series(chains).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
# df = pd.merge(df, df['chain'].str.split('-', expand=True), left_index=True, right_index=True)

# s = []
# for i in range(T + 1):
#     s.append(df.apply(lambda x: x[i] == observations[i], axis=1))

# df['matched'] = pd.concat(s, axis=1).sum(axis=1)
# df['counts'] = df['counts'] / RUNS * 100
# df = df.drop(columns=['chain'])
# df.head(30)
# ---
# |---:|---------:|:----|:----|:----|:----|:----|:----|----------:|
# |  0 |    8.907 | 3L  | 3L  | 3L  | 3L  | 3L  | 3L  |         4 |
# |  1 |    4.422 | 3L  | 2M  | 3L  | 3L  | 3L  | 3L  |         5 |
# |  2 |    4.286 | 1S  | 3L  | 3L  | 3L  | 3L  | 3L  |         3 |
# |  3 |    4.284 | 3L  | 3L  | 3L  | 3L  | 3L  | 2M  |         3 |
# |  4 |    4.278 | 3L  | 3L  | 3L  | 2M  | 3L  | 3L  |         3 |
# |  5 |    4.227 | 3L  | 3L  | 1S  | 3L  | 3L  | 3L  |         5 |
# |  6 |    4.179 | 3L  | 3L  | 3L  | 3L  | 1S  | 3L  |         3 |
# |  7 |    2.179 | 3L  | 2M  | 3L  | 2M  | 3L  | 3L  |         4 |
# |  8 |    2.173 | 3L  | 2M  | 3L  | 3L  | 1S  | 3L  |         4 |
# |  9 |    2.165 | 1S  | 3L  | 1S  | 3L  | 3L  | 3L  |         4 |
# | 10 |    2.147 | 3L  | 2M  | 3L  | 3L  | 3L  | 2M  |         4 |
# | 11 |    2.136 | 3L  | 3L  | 3L  | 2M  | 3L  | 2M  |         2 |
# | 12 |    2.121 | 3L  | 2M  | 1S  | 3L  | 3L  | 3L  |         6 |
# | 13 |    2.111 | 1S  | 3L  | 3L  | 2M  | 3L  | 3L  |         2 |
# | 14 |    2.1   | 1S  | 2M  | 3L  | 3L  | 3L  | 3L  |         4 |
# | 15 |    2.075 | 3L  | 3L  | 3L  | 2M  | 1S  | 3L  |         2 |
# | 16 |    2.05  | 1S  | 3L  | 3L  | 3L  | 3L  | 2M  |         2 |
# | 17 |    2.04  | 3L  | 3L  | 1S  | 3L  | 3L  | 2M  |         4 |
# | 18 |    2.038 | 3L  | 3L  | 1S  | 2M  | 3L  | 3L  |         4 |
# | 19 |    2.022 | 3L  | 3L  | 1S  | 3L  | 1S  | 3L  |         4 |
# | 20 |    2.008 | 1S  | 3L  | 3L  | 3L  | 1S  | 3L  |         2 |
# | 21 |    1.955 | 3L  | 3L  | 3L  | 3L  | 1S  | 2M  |         2 |
# | 22 |    1.079 | 1S  | 2M  | 3L  | 2M  | 3L  | 3L  |         3 |
# | 23 |    1.077 | 1S  | 2M  | 3L  | 3L  | 3L  | 2M  |         3 |
# | 24 |    1.075 | 3L  | 2M  | 1S  | 2M  | 3L  | 3L  |         5 |
# | 25 |    1.064 | 1S  | 2M  | 1S  | 3L  | 3L  | 3L  |         5 |
# | 26 |    1.052 | 1S  | 2M  | 3L  | 3L  | 1S  | 3L  |         3 |
# | 27 |    1.048 | 3L  | 2M  | 3L  | 2M  | 1S  | 3L  |         3 |
# | 28 |    1.032 | 1S  | 3L  | 1S  | 2M  | 3L  | 3L  |         3 |
# | 29 |    1.024 | 1S  | 3L  | 1S  | 3L  | 1S  | 3L  |         3 |


# verify the quality of our model, let’s plot the outcomes 
# together with the frequency of occurrence and compare it 
# against a freshly initialized model, which is supposed to 
# give us completely random sequences

# chart3
# Result after training of the model. The dotted lines represent 
# the matched sequences. The lines represent the frequency of 
# occurrence for a particular sequence: trained model (red) and 
# freshly initialized (black). The initialized results in almost 
# perfect uniform distribution of sequences, while the trained 
# model gives a strong preference towards the observable sequence
hml_rand = HiddenMarkovLayer.initialize(states, observables)
hmm_rand = HiddenMarkovModel(hml_rand)

RUNS = 100000
T = 5

chains_rand = RUNS * [0]
for i in range(len(chains_rand)):
    chain_rand = hmm_rand.layer.run(T)[0]
    chains_rand[i] = '-'.join(chain_rand)

df2 = pd.DataFrame(pd.Series(chains_rand).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
df2 = pd.merge(df2, df2['chain'].str.split('-', expand=True), left_index=True, right_index=True)

s = []
for i in range(T + 1):
    s.append(df2.apply(lambda x: x[i] == observations[i], axis=1))

df2['matched'] = pd.concat(s, axis=1).sum(axis=1)
df2['counts'] = df2['counts'] / RUNS * 100
df2 = df2.drop(columns=['chain'])

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.plot(df['matched'], 'g:')
ax.plot(df2['matched'], 'k:')

ax.set_xlabel('Ordered index')
ax.set_ylabel('Matching observations')
ax.set_title('Verification on a 6-observation chain.')

ax2 = ax.twinx()
ax2.plot(df['counts'], 'r', lw=3)
ax2.plot(df2['counts'], 'k', lw=3)
ax2.set_ylabel('Frequency of occurrence [%]')

ax.legend(['trained', 'initialized'])
ax2.legend(['trained', 'initialized'])

plt.grid()
plt.show()
































