[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pailabteam/ml_finance/master)
[![frontmark](https://img.shields.io/badge/powered%20by-frontmark-lightgrey.svg)](https://www.frontmark.de/)
[![RIVACON](https://img.shields.io/badge/powered%20by-RIVACON-lightgrey.svg)](https://www.rivacon.com/en/)

# Machine Learning in Finance

This is a collection of jupyter notebooks containing basic use cases of Machine Learning methods in the financial sector. The notebooks use only standard ML and Data Science toolkits, most notably sklearn and tensorflow. It is intended as a first introduction and reference for certain use cases.

# Start
Open the overview.ipynb notebook to get started with the use cases. Note that all use cases with tensorflow are designed to work with tensorflow 2.2 or higher versions. If you do not have tohe ability to checkout the repository and run the notebooks on your platform, you can just click the binder-badge to use the binder hosted jupyter to play around with the examples.

# Notebook overview
An overview of all existing notebooks is contained in the [overview](overview.ipynb) notebook.

## Data Visualisation with ML

### Volatility Surface Visualisation

The idea of [this notebook](Volatilities_MDS.ipynb) is to visualize distances and the development of volatility surfaces. For this purpose the Multi Dimensional Scaling (MDS) algorithm is used.

## Scoring

### Peer Group Scoring using Siamese networks

[This notebook](peer_scoring_siamese/siamese.ipynb) illustrates the application of Siamese networks to construct a peer group scoring used to create peer groups for structured products. 


## Deep Hedging

### Black-Scholes Call with Bid-Ask Spreads
[Here](deep_hedging/deep_hedging.ipynb) we show how to implement a neural network to compute strategies for hedging a call option (in a model fre sense) on underlyings that are traded with bid-ask spreads.

# Tribute

Thanks to our sponsors [frontmark](https://www.frontmark.de/) and [RIVACON](https://www.rivacon.com/).

[<img src="images/favicon_2.png" width='70px'>](https://www.frontmark.de/)

[<img src="images/logo.png" width='100px'>](https://www.rivacon.com/)
