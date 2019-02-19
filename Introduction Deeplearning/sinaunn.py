import numpy as np
from scipy.stats import truncnorm

def trunc_normal(mean = 0, sd = 1, low = 0, up = 10):
  return truncnorm( (low - mean)/sd, (up - mean)/sd, scale = sd, loc = mean )

rad = 1 / np.sqrt(3)
X = trunc_normal(mean = 2, sd = 1, low = -rad, up = rad)
s = X.rvs((3, 4))

import matplotlib.pyplot as plt
plt.hist(s)
plt.show()