# =============================================================================
# Model test on l1 log_reg
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
import matplotlib.pyplot as plt
from model.InverseIsing.InverseIsing import Bethe
from model.Tools.MonteCarlo import MonteCarlo   

#%% load data
mi_data = np.load("mi_data_10w.npy")
mi_data = np.array(mi_data, ndmin=2).T
Ca_data = np.load("Ca_data_10w.npy")
Ca_data[np.diag_indices_from(Ca_data)] = 0


#%% parameter
beta = 1
N = 675
l1 = 0.5
path = "D:/PMI/Projects/LLDL/task/Pseudo/PseudoL1/SparseTest/{}".format(l1)

# create object
bethe_estimator = Bethe(beta)

#%% define function
def read_model(path, index_r, beta, l1):
    fm = open(path + "/model/model_{}_{}".format(l1, index_r), "r")
    lines = fm.readlines()
    h_r = lines[9]
    J_r = lines[10:]
    h_r = float(h_r.rstrip("\n"))
    for i in range(len(J_r)):
        J_r[i] = float(J_r[i].rstrip("\n"))
    J_r = np.array(J_r).reshape(-1)
    J_r = np.insert(J_r, index_r, 0)
    J_r = J_r/(2*beta)
    h_r = h_r/(2*beta)
    return J_r, h_r

#%% main
J_esti = np.zeros([N,N])
h_esti = np.zeros([N,1])
for index_r in range(N):
    print("\rLoading data of row {}..".format(index_r), end='')
    J_r, h_r = read_model(path, index_r, beta, l1)
    J_esti[index_r] = 1*J_r
    h_esti[index_r] = 1*h_r
J_esti = 1/2*(J_esti+J_esti.T)   


#%% save data
# np.save(path + "/J_esti.npy", J_esti)
# np.save(path + "/h_esti.npy", h_esti)

#%% load data
l1 = 0.5
path = "D:/PMI/Projects/LLDL/task/Pseudo/PseudoL1/SparseTest/{}".format(l1)

J_esti = np.load(path + "/J_esti.npy")
h_esti = np.load(path + "/h_esti.npy")

print("L1: {}".format(l1))

#%%% Bethe
mi_model, Ca_model, _ = bethe_estimator.model_aver(J_esti, h_esti)
mse, mse_mi, mse_Ca = bethe_estimator.mse(mi_data, mi_model, Ca_data, Ca_model)


#%%% model parameter J,h distribution
def SparseOfJ(J):
    N = J.shape[0]
    return ((J_esti==0).sum()-N)/(N**2-N)
Sparse = SparseOfJ(J_esti)

plt.figure(figsize=(16,6), dpi=80)
ax1 = plt.subplot(1,2,1)
J_show = J_esti*1
J_show[np.diag_indices_from(J_show)] = None
plt.hist(J_esti.reshape(-1), bins=50, density=True, edgecolor = 'black', color='skyblue')
plt.ylabel('Histogram',size=20)
plt.xlabel('Coupling',size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.text(0.1, 0.8, r'$J_{ij}$', transform=ax1.transAxes, fontsize=40)
plt.text(0.55, 0.5, "Sparsity: {:.2f}%".format(Sparse*100), transform=ax1.transAxes, fontsize=20)
plt.grid(False)

ax2 = plt.subplot(1,2,2)
plt.hist(h_esti, bins=50, density=True, edgecolor = 'black', color = 'steelblue')
plt.ylabel('Histogram',size=20)
plt.xlabel('External field',size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.text(0.1, 0.8, r'$h_i$', transform=ax2.transAxes, fontsize=40)
plt.grid(False)
plt.savefig(path + "/modelPara_dist.png", dpi=200, bbox_inches='tight')
plt.show()


#%%% MonteCarlo
def MonteCarlo_Modelavg(J, h):
    relaxation = 10000
    interval = 20
    M = 5000
    monte_carlo_sampling = MonteCarlo(beta, relaxation, interval, M)
    sigma = monte_carlo_sampling.Sampling(J, h)
    mi_model = np.array(sigma.mean(0), ndmin=2).T
    Ca_temp = 0
    for i in range(M):
        Ca_temp += np.dot( (sigma[i:(i+1)]).T, sigma[i:(i+1)])
    Ca_model = Ca_temp/M
    Ca_model[np.diag_indices_from(Ca_model)] = 0
    return mi_model, Ca_model
    
mi_model, Ca_model = MonteCarlo_Modelavg(J_esti, h_esti)
mse, mse_mi, mse_Ca = bethe_estimator.mse(mi_data, mi_model, Ca_data, Ca_model)
print("MSE:{}, MSE_mi:{}, MSE_Ca:{};".format(mse, mse_mi, mse_Ca))


#%% save data
l1 = 0.01
path = "D:/PMI/Projects/LLDL/task/Pseudo/PseudoL1/SparseTest/{}".format(l1)
np.save(path + "/mi_model.npy", mi_model)
np.save(path + "/Ca_model.npy", Ca_model)

#%%% Inverse Ising error
plt.figure(figsize=(16,6), dpi=80)
ax1 = plt.subplot(1,2,1)
ub = np.maximum(mi_data.max(), mi_data.max())
lb = np.minimum(mi_data.min(), mi_model.min())    
x = np.arange(lb-0.2, ub+0.2, 0.1)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(mi_data, mi_model, c ='green', s=2, label='mi')   
plt.xlim(lb,ub)
plt.ylim(lb,ub)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(loc=2, fontsize=20)
plt.text(0.5, 0.2, 'MSE={:.4f}\nMSE_mi={:.4f}'.format(mse, mse_mi),\
         transform=ax1.transAxes, fontsize=20)

ax2 = plt.subplot(1,2,2)
ub = np.maximum(Ca_data.max(), Ca_data.max())
lb = np.minimum(Ca_data.min(), Ca_model.min())    
x = np.arange(lb-0.2, ub+0.2, 0.1)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(Ca_data, Ca_model, c ='blue', s=2, label='Ca')
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(loc=2, fontsize=20)
plt.text(0.5, 0.2, 'MSE_Ca={:.4f}'.format(mse_Ca), transform=ax2.transAxes,\
         fontsize=20)
#plt.savefig(path + "/InverseIsing.png", dpi=200, bbox_inches='tight')
plt.show()
























