# =============================================================================
# energy vs accuracy
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

#%% functions
# compute energy vs accuracy
def energy_accuracy(X):
    X = X[np.argsort(X[:,1])]
    mean = []
    flag = 0.8
    e = []
    for i in range(0, X.shape[0]):
        print("\rSteps: {}".format(i), end='')
        a = X[i][1]
        if a == flag:
            e.append(X[i][0])
        else:
           e = np.array(e)
           e_mean = np.mean(e)
           e_var = np.std(e)
           count = np.size(e)
           e_a = []
           e_a.append(e_mean)
           e_a.append(e_var)
           e_a.append(flag)
           e_a.append(count)
           mean.append(e_a)
           flag = a
           e=[]
           e.append(X[i][0])
        if i == (X.shape[0]-1):
           e.append(X[i][0])
           e = np.array(e)
           e_mean = np.mean(e)
           e_var = np.std(e)
           count = np.size(e)
           e_a = []
           e_a.append(e_mean)
           e_a.append(e_var)
           e_a.append(flag)
           e_a.append(count)
           mean.append(e_a)
    mean = np.array(mean)
    return mean
    
#%% main
case = 'up'
e_a = np.loadtxt('./{}/energy_accuracy_{}.txt'.format(case, case))
e_a = e_a[np.argsort(e_a[:,1])]
#%% compute
mean = energy_accuracy(e_a)
# np.save("./{}/mean_ea.npy".format(case), mean)

#%% plot: e vs a
plt.figure(dpi=256, figsize=(10,6))
accuracy = mean[:, 2]
energy_mean = mean[:, 0]/675
mean_e = (e_a[:,0]).mean(0)/675
plt.plot(accuracy, energy_mean, color='b', label='{}'.format(case), linewidth=2)
plt.plot(accuracy, mean_e*np.ones(531), color='b', ls='--', label="mean energy", linewidth=2)
energy_std = mean[:, 1]/675
plt.fill_between(accuracy, energy_mean-energy_std, energy_mean+energy_std, color='b', alpha=0.3)
plt.xlabel('test accuracy',size=20)
plt.ylabel('r$\epsilon_{Ising}$',size=20)
plt.legend(loc=1,fontsize=15)


#%% plot: energy distribution
# draw energy 
plt.figure(figsize=(10,6), dpi=200)
energy = e_a[:,0]
plt.hist(energy/675, 50, density=True, label="under-parameteried")
plt.plot(-0.115*np.ones(100), np.linspace(0, 32, 100), c='k', lw=2, ls='--')
plt.plot(-0.075*np.ones(100), np.linspace(0, 32, 100), c='k', lw=2, ls='--')
plt.ylim(0, 32)
plt.legend(loc=1, fontsize=25)
plt.xlabel('$\epsilon_{Ising}$',size=25)
plt.ylabel('histogram',size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
plt.savefig("./{}/Ehist_new_{}.png".format(case,case), dpi=200, bbox_inched='tight')




