# =============================================================================
# Compute Ising energy for all the sigma
# =============================================================================

import numpy as np

#%% load data
case = 'up'
J = np.load('./pseudo/J_{}.npy'.format(case))
h = np.load('./pseudo/h_{}.npy'.format(case))
data = np.load("./pseudo/data_{}_10w.npy".format(case))

#%% compute energy
sigma = data[:,1:]
energies = np.zeros([data.shape[0],1])
ea_sigma = np.concatenate((energies, data), axis=1)
for u in range(0, sigma.shape[0]):
    if ((u+1)%1000 == 0):
        print('Compute: {}'.format(u+1))
    energy = -0.5*np.sum(J*np.dot(sigma[u:u+1].T,sigma[u:u+1]))-np.dot(sigma[u],h)
    ea_sigma[u][0] = energy

np.save('./pseudo/data_energy_{}_10w.npy'.format(case), ea_sigma)
print('100w_energy saved.')


e_a = ea_sigma[:,[0,1]]
np.savetxt('./pseudo/energy_accuracy_pseudo_{}.txt'.format(case), e_a)
print('energy_accuracy saved.')

