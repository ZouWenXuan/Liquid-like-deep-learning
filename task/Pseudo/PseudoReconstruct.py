# =============================================================================
# Pseudo-likelihood estimate the J and h
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
import time
from model.InverseIsing.InverseIsing import Pseudo, Bethe
from model.Tools.Optimizers import SGD
from model.Tools.MonteCarlo import MonteCarlo   
print("Modules imported!")


#%% load data
data = np.load("data_10w.npy", allow_pickle=True)
mi_data = np.load("mi_data_10w.npy")
mi_data = np.array(mi_data, ndmin=2).T
Ca_data = np.load("Ca_data_10w.npy", allow_pickle=True)
Ca_data[np.diag_indices_from(Ca_data)] = 0


#%% hyper-parameters
beta = 1
lr_J, lr_h = 0.04, 0.02
trials = 200
sigma = data[:,1:]

# create object
pseudo_estimator = Pseudo(beta)
bethe_estimator = Bethe(beta)
Optimizer_J = SGD()
Optimizer_h = SGD()

#%% learning function
def pseudo_learning(pseudo_estimator, Optimizer_J, Optimizer_h, sigma, \
                    trials, lr_J, lr_h):
    # initialize the J,h and optimizer
    N = sigma.shape[1]
    J,h = pseudo_estimator.initialized(N)
    pseudo_estimator.optimizer(Optimizer_J, Optimizer_h)

    # estimating
    start = time.time()
    print("Begin estimating...")
    grad = np.zeros((trials,2))
    loss = np.zeros(trials)
    J_temp, h_temp = J*1, h*1
    for i in range(0, trials):
        J, h, gJ, gh, f = pseudo_estimator.estimate(sigma.T, J_temp,h_temp, lr_J, lr_h)
        grad[i][0], grad[i][1], loss[i] = np.abs(gJ).mean(), np.abs(gh).mean(), f
        print("Trial {}, loss:{:.4f}, grad_J={:.4f}, grad_h={:.4f};".\
              format(i, f, grad[i][0], grad[i][1]))
        J_temp, h_temp = J*1, h*1
    end = time.time()
    print("Total time cost: {:.2f}s".format(end-start))    
    return J, h, grad, loss


J, h, grad, loss = pseudo_learning(pseudo_estimator, Optimizer_J, Optimizer_h,\
                                    sigma, trials, lr_J, lr_h)   
mi_model, Ca_model, _ = bethe_estimator.model_aver(J, h)
# mse, mse_mi, mse_Ca = bethe_estimator.mse(mi_data, mi_model, Ca_data, Ca_model)
# print("MSE:{}, MSE_mi:{}, MSE_Ca:{};".format(mse, mse_mi, mse_Ca))

#%% MonteCarlo
def MonteCarlo_Modelavg(J, h):
    relaxation = 100
    interval = 100
    M = 10000
    monte_carlo_sampling = MonteCarlo(beta, relaxation, interval, M)
    sigma = monte_carlo_sampling.Sampling(J, h)
    mi_model = np.array(sigma.mean(0), ndmin=2).T
    Ca_temp = 0
    for i in range(M):
        Ca_temp += np.dot( (sigma[i:(i+1)]).T, sigma[i:(i+1)])
    Ca_model = Ca_temp/M
    Ca_model[np.diag_indices_from(Ca_model)] = 0
    return mi_model, Ca_model

def Modelavg(sigma):
    # sigma: [M,N]
    mi_model = np.array(sigma.mean(0), ndmin=2).T
    Ca_temp = 0
    for i in range(sigma.shape[0]):
        print("\r{}".format(i),end='')
        Ca_temp += np.dot( (sigma[i:(i+1)]).T, sigma[i:(i+1)])
    Ca_model = Ca_temp/sigma.shape[0]
    Ca_model[np.diag_indices_from(Ca_model)] = 0
    return mi_model, Ca_model

    
# mi_model, Ca_model = MonteCarlo_Modelavg(J, h)
# mse, mse_mi, mse_Ca = bethe_estimator.mse(mi_data, mi_model, Ca_data, Ca_model)
# print("MSE:{}, MSE_mi:{}, MSE_Ca:{};".format(mse, mse_mi, mse_Ca))

# MC_dict = np.load("D:/PMI/Projects/LLDL/task/Generate/newTest/MC_dict.npy",\
#                   allow_pickle=True).item()
# sigma = MC_dict['sigma']
# mi_model, Ca_model = Modelavg(sigma)
# mse, mse_mi, mse_Ca = bethe_estimator.mse(mi_data, mi_model, Ca_data, Ca_model)

#%% save J, h, etc.
# np.savetxt("J_test.txt", J)
# np.savetxt("h_test.txt", h)
# np.savetxt("loss_test.txt", loss)
# np.savetxt("grad_test.txt", grad)
# np.savetxt("mi_model_test.txt", mi_model)
# np.savetxt("Ca_model_test.txt", Ca_model)

#%% plot
import matplotlib.pyplot as plt
data_path = "D:/PMI/Projects/LLDL/task/Pseudo/PseudoOrigin/TestResults/data_origin"
J = np.loadtxt(data_path + "/J_test.txt")
h = np.loadtxt(data_path + "/h_test.txt")
loss = np.loadtxt(data_path + "/loss_test.txt")
grad = np.loadtxt(data_path + "/grad_test.txt")
mi_model = np.loadtxt(data_path + "/mi_model_test.txt")
Ca_model = np.loadtxt(data_path + "/Ca_model_test.txt")


#%%% Inverse Ising error
steps = np.arange(0, trials, 1)
step_optimal = np.argmin(loss)

plt.figure(figsize=(16,10), dpi=80)
ax1 = plt.subplot(2,2,1)
plt.plot(steps, loss, c='g', label = 'loss')
plt.legend(loc=1, fontsize=20)
plt.text(0.5, 0.4, 'optimal step: {}\nloss = {:.4f}'.format(step_optimal,\
                    loss[step_optimal]), transform=ax1.transAxes, fontsize=20)

ax2 = plt.subplot(2,2,2)
plt.plot(steps, grad[:,0], label='grad_J')
plt.plot(steps, grad[:,1], label='grad_h')
plt.legend(loc=1, fontsize=20)
plt.text(0.2, 0.4,'grad_J = {:.4f}\ngrad_h = {:.4f}'.format(grad[step_optimal][0],\
                grad[step_optimal][1]),transform=ax2.transAxes, fontsize=20)

ax3 = plt.subplot(2,2,3)
ub = np.maximum(mi_data.max(), mi_data.max())
lb = np.minimum(mi_data.min(), mi_model.min())    
x = np.arange(lb-0.02, ub+0.02, 0.1)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(mi_data, mi_model, c ='green', s=0.5, label='mi')   
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.legend(loc=2, fontsize=20)
plt.text(0.5, 0.2, 'MSE_mi={:.4f}'.format(mse_mi), transform=ax3.transAxes, fontsize=20)


ax4 = plt.subplot(2,2,4)
ub = np.maximum(Ca_data.max(), Ca_data.max())
lb = np.minimum(Ca_data.min(), Ca_model.min())    
x = np.arange(lb-0.02, ub+0.02, 0.1)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(Ca_data, Ca_model, c ='blue', s=0.5, label='Ca')
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.legend(loc=2, fontsize=20)
plt.text(0.5, 0.2, 'MSE_Ca={:.4f}'.format(mse_Ca), transform=ax4.transAxes, fontsize=20)
# plt.savefig('Pseudo_InverseIsing.png', dpi=200, bbox_inches='tight')
plt.show()


#%%% model parameter J,h distribution
plt.figure(figsize=(16,6), dpi=80)
ax1 = plt.subplot(1,2,1)
J[np.diag_indices_from(J)] = None
plt.hist(J.reshape(-1), bins=50, density=True, edgecolor = 'black', color='skyblue')
plt.ylabel('Histogram',size=20)
plt.xlabel('Coupling',size=20)
plt.text(0.6, 0.5, r'$J_{ij}$', transform=ax1.transAxes, fontsize=40)
plt.grid(False)

ax2 = plt.subplot(1,2,2)
plt.hist(h, bins=50, density=True, edgecolor = 'black', color = 'steelblue')
plt.ylabel('Histogram',size=20)
plt.xlabel('External field',size=20)
plt.text(0.7, 0.5, r'$h_i$', transform=ax2.transAxes, fontsize=40)
plt.grid(False)
# plt.savefig('Pseudo_Hist.png', dpi=200, bbox_inches='tight')
plt.show()