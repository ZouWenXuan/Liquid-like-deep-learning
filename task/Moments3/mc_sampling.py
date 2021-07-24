# =============================================================================
# Monte Carlo for mm3 sampling
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
import time
from model.Tools.MonteCarlo import MonteCarlo   
print("Modules imported!")

#%% load data and parameter
J = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/up/J_FULL.txt")
h = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/up/h_FULL.txt")
h = h.reshape(-1,1)

#%% main
# mc parameter 
relaxation = 10000
interval = 10
M = 10000
beta = 1

# run
time1 = time.time()
monte_carlo_sampling = MonteCarlo(beta, relaxation, interval, M)
sigma = monte_carlo_sampling.Sampling(J, h)
time2 = time.time()
print("Time cost: {:.2f}".format(time2-time1))

# save 
np.save("mc_sigma.npy", sigma)


#%% extract save
index_up = np.loadtxt("index_up.txt")
sigma_index = sigma[:, index_up.astype(int)]
np.save("mc_sigma_index.npy", sigma_index)

