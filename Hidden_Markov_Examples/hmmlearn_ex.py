# Use hmmlearn library to model actual historical gold prices 
# using 3 different hidden states corresponding to 3 possible 
# market volatility levels.



# The hmmlearn Library
# hmmlearn is a Python library which implements Hidden Markov 
# Models in Python! hmmlearn provides three models out of the 
# box — a multinomial emissions model, a Gaussian emissions model
# and a Gaussian mixture emissions model, although the framework 
# does allow for the implementation of custom emissions models.

# The Multinomial Emissions Model
# The multinomial emissions model assumes that the observed
# processes X consists of discrete values, such as for the mood 
# case study above.

# We will next take a look at 2 models used to model continuous 
# values of X.

# The Gaussian Emissions Model
# The Gaussian emissions model assumes that the values in X are 
# generated from multivariate Gaussian distributions (i.e. 
# N-dimensional Gaussians), one for each hidden state. Each 
# multivariate Gaussian distribution is defined by a multivariate 
# mean and covariance matrix.

# hmmlearn allows us to place certain constraints on the 
# covariance matrices of the multivariate Gaussian distributions.

# covariance_type = “diag” — the covariance matrix for each 
# hidden state is a diagonal matrix, and these matrices may be 
# different.
# covariance_type = “spherical” —the covariance matrix for each 
# hidden state is proportional to the identity matrix, and these 
# matrices may be different.
# covariance_type = “full” —no restrictions placed on the 
# covariance matrices for any of the hidden states.
# covariance_type = “tied” —all hidden states share the same 
# full covariance matrix.
# Gaussian Mixture Emissions Model
# This is the most complex model available out of the box. The 
# Gaussian mixture emissions model assumes that the values in X 
# are generated from a mixture of multivariate Gaussian 
# distributions, one mixture for each hidden state. Each 
# multivariate Gaussian distribution in the mixture is defined 
# by a multivariate mean and covariance matrix.

# As with the Gaussian emissions model above, we can place 
# certain constraints on the covariance matrices for the Gaussian
# mixture emissions model as well.

# covariance_type = “diag” —for each hidden state, the covariance 
# matrix for each multivariate Gaussian distribution in the mixture 
# is diagonal, and these matrices may be different.
# covariance_type = “spherical” — for each hidden state, the 
# covariance matrix for each multivariate Gaussian distribution in
# the mixture is proportional to the identity matrix, and these 
# matrices may be different.
# covariance_type = “full” — no restrictions placed on the 
# covariance matrices for any of the hidden states or mixtures.
# covariance_type = “tied” — for each hidden state, all mixture 
# components use the same full covariance matrix. These matrices 
# may be different between the hidden states, unlike in the 
# Gaussian emission model.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

base_dir = "https://github.com/natsunoyuki/Data_Science/blob/master/gold/gold/gold_price_usd.csv?raw=True"

data = pd.read_csv(base_dir)

# Convert the datetime from str to datetime object.
data["datetime"] = pd.to_datetime(data["datetime"])

# Determine the daily change in gold price.
data["gold_price_change"] = data["gold_price_usd"].diff()

# Restrict the data to later than 2008 Jan 01.
data = data[data["datetime"] >= pd.to_datetime("2008-01-01")]

# chart1
# Historical gold prices in USD, 
# as well as the change in the corresponding daily change. 
# Plot the daily gold prices as well as the daily change.
plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
plt.plot(data["datetime"], data["gold_price_usd"])
plt.xlabel("datetime")
plt.ylabel("gold price (usd)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(data["datetime"], data["gold_price_change"])
plt.xlabel("datetime")
plt.ylabel("gold price change (usd)")
plt.grid(True)
plt.show()


# Instead of modeling the gold price directly, we model the daily
# change in the gold price — this allows us to better capture the
# state of the market. We fit the daily change in gold prices to 
# a Gaussian emissions model with 3 hidden states. The reason for
# using 3 hidden states is that we expect at the very least 3 
# different regimes in the daily changes — low, medium and high 
# votality.

# Use the daily change in gold price as the observed measurements X.
X = data[["gold_price_change"]].values
# Build the HMM model and fit to the gold price change data.
model = hmm.GaussianHMM(n_components = 3, 
                        covariance_type = "diag", 
                        n_iter = 50, 
                        random_state = 42
                        )
model.fit(X)
# Predict the hidden states corresponding to observed X.
Z = model.predict(X)
states = pd.unique(Z)

print(f"Unique states: {states}")
# Unique states: [0 1 2]
# We find that the model does indeed return 3 unique hidden states. 
# These numbers do not have any intrinsic meaning. Which state 
# corresponds to which volatility regime must be confirmed by 
# looking at the model parameters

# We find that for this particular data set, the model will almost 
# always start in state 0.

print(f"\nStart probabilities: {model.startprob_}")
# Start probabilities: [1.00000000e+00 4.28952054e-24 1.06227453e-46]

# The transition matrix for the 3 hidden states show that the 
# diagonal elements are large compared to the off diagonal 
# elements. This means that the model tends to want to remain in 
# that particular state it is in — the probability of 
# transitioning up or down is not high.

print("\nTransition matrix:")
print(model.transmat_)
# Transition matrix:
# [[8.56499275e-01 1.42858023e-01 6.42701428e-04]
#  [2.43257082e-01 7.02528333e-01 5.42145847e-02]
#  [1.33435298e-03 1.67318160e-01 8.31347487e-01]]


# Finally, we take a look at the Gaussian emission parameters. 
# Remember that each observable is drawn from a multivariate 
# Gaussian distribution. For state 0, the Gaussian mean is 0.28, 
# for state 1 it is 0.22 and for state 2 it is 0.27. The fact that 
# states 0 and 2 have very similar means is problematic — our 
# current model might not be too good at actually representing 
# the data.

print("\nGaussian distribution means:")
print(model.means_)
# Gaussian distribution means:
# [[0.27988823]
#  [0.2153654 ]
#  [0.26501033]]

# We also have the Gaussian covariances. Note that because our data 
# is 1 dimensional, the covariance matrices are reduced to scalar 
# values, one for each state. For state 0, the covariance is 33.9, 
# for state 1 it is 142.6 and for state 2 it is 518.7. This seems 
# to agree with our initial assumption about the 3 volatility 
# regimes — for low volatility the covariance should be small, 
# while for high volatility the covariance should be very large.


print("\nGaussian distribution covariances:")
print(model.covars_)
# Gaussian distribution covariances:
# [[[ 33.89296208]]

#  [[142.59176749]]

#  [[518.65294332]]]


# Plotting the model’s state predictions with the data, we find 
# that the states 0, 1 and 2 appear to correspond to low volatility, 
# medium volatility and high volatility.

# chart2
# Market volatility as modeled using a Gaussian emissions Hidden 
# Markov Model. Blue/state 0 — low volatility, orange/state 
# 1— medium volatility, green/state 2 — high volatility
plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_usd"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price (usd)", fontsize=16)
plt.subplot(2,1,2)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_change"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price change (usd)", fontsize=16)
plt.show()

# From the graphs above, we find that periods of high volatility 
# correspond to difficult economic times such as the Lehmann shock 
# from 2008 to 2009, the recession of 2011–2012 and the covid 
# pandemic induced recession in 2020. Furthermore, we see that 
# the price of gold tends to rise during times of uncertainty as 
# investors increase their purchases of gold which is seen as a 
# stable and safe asset


