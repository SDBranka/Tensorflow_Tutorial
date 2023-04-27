
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

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]



# ex
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












































































