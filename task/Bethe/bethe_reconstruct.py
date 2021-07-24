# =============================================================================
# Cavity method estimate the J and h
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
from model.InverseIsing.InverseIsing import Bethe
from model.Tools.Optimizers import SGD
print("Modules imported!")


#%% load data
# parameter
case = 'fa'
boundary = 'FULL'


data_path = "D:/PMI/Projects/LLDL/data/weights_space"
mi_data = np.loadtxt(data_path+"/new/{}/moments/mi_data.txt".format(case))
Ca_data = np.loadtxt(data_path+"/new/{}/moments/Ca_data.txt".format(case))
mi_data = np.array(mi_data, ndmin=2).T
Ca_data[np.diag_indices_from(Ca_data)] = 0

N = mi_data.size
var_data = Ca_data - np.dot(mi_data.reshape(N,1), mi_data.reshape(N,1).T)
var_data[np.diag_indices_from(var_data)] = 0

#%% select boundary
segment = {}
segment['LB'] = {}
segment['LB']['start'], segment['LB']['end'] = 0, 300
segment['ITER'] = {}
segment['ITER']['start'], segment['ITER']['end'] = 300, 525
segment['RB'] = {}
segment['RB']['start'], segment['RB']['end'] = 525, 675
segment['FULL'] = {}
segment['FULL']['start'], segment['FULL']['end'] = 0, 675


mi_data = mi_data[segment[boundary]['start']:segment[boundary]['end']]
Ca_data = Ca_data[segment[boundary]['start']:segment[boundary]['end'],\
                  segment[boundary]['start']:segment[boundary]['end']]
var_data = var_data[segment[boundary]['start']:segment[boundary]['end'],\
                    segment[boundary]['start']:segment[boundary]['end']]
N = mi_data.size

#%% hyper-parameters
beta = 1
lr_J, lr_h = 0.01, 0.3
trials = 100
trials_cutoff = 50
gamma = 0.0001
reg = 'l2'

# create object
bethe_estimator = Bethe(beta)
Optimizer_J = SGD()
Optimizer_h = SGD()
print("Learning rate, J:{}, h:{}.".format(lr_J, lr_h))

#%% learning
def initialized(N):
    # J diagonal without diagonal elements
    J = 1/N*np.random.randn(N,N)
    J = J * J.T
    J[np.diag_indices_from(J)] = 0
    h = np.zeros([N,1])
    return J, h
    

def learning(bethe_estimator, mi_data, Ca_data, Optimizer_J, Optimizer_h, \
                                     trials, trials_cutoff, lr_J, lr_h, gamma, reg):
    N = mi_data.size
    mi_data = mi_data.reshape(N,1)
    Ca_data[np.diag_indices_from(Ca_data)] = 0

    # record the learning process
    mse, mse_mi, mse_Ca = np.zeros(trials), np.zeros(trials), np.zeros(trials)
    
    # initialize the J,h
    J, h = initialized(N)
    bethe_estimator.optimizer(Optimizer_J, Optimizer_h)
    
    for i in range(0, trials): 
        mi_model, Ca_model, _ = bethe_estimator.model_aver(J, h)
        mse_trial, mse_mi_trial, mse_Ca_trial = bethe_estimator.mse\
                                    (mi_data, mi_model, Ca_data, Ca_model)
        mse[i], mse_mi[i], mse_Ca[i] = mse_trial, mse_mi_trial, mse_Ca_trial
        print("Trial: {}, error: {:.4f};".format(i, mse[i]))
        
        # record the optimal
        if i==0:
            mse_optimal = mse_trial
            J_optimal, h_optimal = J*1, h*1
        else:
            if ((mse_trial < mse_optimal) & (i>trials_cutoff)):
                mse_optimal = mse_trial
                J_optimal, h_optimal = J*1, h*1

        gJ = Ca_model - Ca_data
        gh = mi_model - mi_data
        if reg == 'None':
            pass
        elif reg == 'l1':
            gJ += gamma*np.sign(J)
        elif reg == 'l2':
            gJ += gamma*J
        else:
            raise ValueError("Reg can only be l1,l2 or None!")            
        J = bethe_estimator.opti_J.optimize(lr_J, J, gJ)
        h = bethe_estimator.opti_h.optimize(lr_h, h, gh)   
    return mse, mse_mi, mse_Ca, J_optimal, h_optimal

mse, mse_mi, mse_Ca, J_optimal, h_optimal = learning(bethe_estimator, mi_data,\
         Ca_data, Optimizer_J, Optimizer_h, trials, trials_cutoff, lr_J, lr_h, gamma, reg)
mi_model, Ca_model, _ = bethe_estimator.model_aver(J_optimal, h_optimal)
var_model = Ca_model - np.dot(mi_model.reshape(N,1), mi_model.reshape(N,1).T)
var_model[np.diag_indices_from(var_model)] = 0    


#%% Test plot
import matplotlib.pyplot as plt
plt.figure(dpi=80,figsize=(20,8))
plt.subplot(2,3,1)
x = [i for i in range(0, mse.size)]
plt.plot(x, mse, 'red', linestyle='-',linewidth = 1, label='MSE')
plt.legend()
plt.xlabel('Learning step', size=15)
plt.ylabel('MSE', size=15)

plt.subplot(2,3,2)
x = [i for i in range(0, mse.size)]
plt.plot(x, mse_mi, 'green', linestyle='-',linewidth = 1, label='MSE_mi')
plt.legend()
plt.xlabel('Learning step', size=15)
plt.ylabel('MSE_mi', size=15)

plt.subplot(2,3,3)
x = [i for i in range(0, mse.size)]
plt.plot(x, mse_Ca, 'blue', linestyle='-',linewidth = 1, label='MSE_Ca')
plt.legend()
plt.xlabel('Learning step', size=15)
plt.ylabel('MSE_Ca', size=15)


step = np.argmin(mse[trials_cutoff:])+trials_cutoff
plt.subplot(2,3,4)
ub = np.maximum(var_data.max(), var_model.max())
lb = np.minimum(var_data.min(), var_model.min())    
x = np.arange(lb-0.002, ub+0.002, 0.01)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(var_data, var_model, c ='red', s=0.5, label='Step: {}, MSE: {}'\
            .format(step, mse[step]))   
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.legend()

plt.subplot(2,3,5)
ub = np.maximum(mi_data.max(), mi_model.max())
lb = np.minimum(mi_data.min(), mi_model.min())    
x = np.arange(lb-0.0002, ub+0.0002, 0.001)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(mi_data, mi_model, c ='green', s=0.5, label='Step: {}, MSE: {}'\
            .format(step, mse_mi[step]))   
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.legend()

plt.subplot(2,3,6)
ub = np.maximum(Ca_data.max(), Ca_model.max())
lb = np.minimum(Ca_data.min(), Ca_model.min())    
x = np.arange(lb-0.002, ub+0.002, 0.01)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(Ca_data, Ca_model, c ='blue', s=0.5, label='Step: {}, MSE: {}'\
            .format(step, mse_Ca[step]))   
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.legend()
plt.show()


#%% save
np.savetxt("D:/PMI/Projects/LLDL/task/Bethe/results/{}/J_{}.txt".format(case, boundary), J_optimal)
np.savetxt("D:/PMI/Projects/LLDL/task/Bethe/results/{}/h_{}.txt".format(case, boundary), h_optimal)

#%% Results plot 
modelpara_path = "D:/PMI/Projects/LLDL/task/Bethe/results/{}".format(case)
J = np.loadtxt(modelpara_path + '/J_{}.txt'.format(boundary))
h = np.loadtxt(modelpara_path + '/h_{}.txt'.format(boundary))
mi_model, Ca_model, _ = bethe_estimator.model_aver(J, h)
mse, mse_mi, mse_Ca = bethe_estimator.mse(mi_data, mi_model, Ca_data, Ca_model)

#%%% plot
# Inverse Ising Fit and J,h hist
import matplotlib.pyplot as plt
plt.figure(dpi=200, figsize=(16,10))
ax1 = plt.subplot(2,2,1)
ub = np.maximum(mi_data.max(), mi_model.max())
lb = np.minimum(mi_data.min(), mi_model.min())    
x = np.arange(lb-0.0002, ub+0.0004, 0.0002)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(mi_data, mi_model, c ='green', s=2)   
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.xlabel('mi (data)',size=20)
plt.ylabel('mi (model)',size=20)
plt.text(0.6, 0.3, "mse={:.4f}".format(mse), transform=ax1.transAxes, fontsize=20)


plt.subplot(2,2,2)
ub = np.maximum(Ca_data.max(), Ca_model.max())
lb = np.minimum(Ca_data.min(), Ca_model.min())    
x = np.arange(lb-0.0002, ub+0.0002, 0.0002)
plt.plot(x, x, c='gray', linewidth=2, ls='--')
plt.scatter(Ca_data, Ca_model, c ='blue', s=2)   
plt.xlim(lb,ub)
plt.ylim(lb,ub) 
plt.xlabel('Ca (data)',size=20)
plt.ylabel('Ca (model)',size=20)

# model parameter J,h distribution
ax3 = plt.subplot(2,2,3)
J[np.diag_indices_from(J)] = None
plt.hist(J.reshape(-1), bins=50, density=True, edgecolor = 'black', color='skyblue')
plt.ylabel('Histogram',size=20)
plt.xlabel('Coupling',size=20)
plt.text(0.3, 0.5, r'$J_{ij}$', transform=ax3.transAxes, fontsize=40)
plt.grid(False)


ax4 = plt.subplot(2,2,4)
plt.hist(h, bins=50, density=True, edgecolor = 'black', color = 'steelblue')
plt.ylabel('Histogram',size=20)
plt.xlabel('External field',size=20)
plt.text(0.7, 0.5, r'$h_i$', transform=ax4.transAxes, fontsize=40)
plt.grid(False)
plt.savefig("D:/PMI/Projects/LLDL/task/Bethe/results/{}/FitHist_{}.png"\
            .format(case,boundary), dpi=200, bbox_inches='tight')

    
#%% suppliment task: Iterior test mC hist
plt.figure(figsize=(12,6),dpi=200)
ax1 = plt.subplot(1,2,1)
plt.hist(mi_model, bins=50, density=True, edgecolor = 'black',\
         color='skyblue', alpha=0.7, label='mi_model')
plt.hist(mi_data.reshape(-1), bins=50, density=True, edgecolor = 'black',\
         color='green', alpha=0.7, label='mi_data')
plt.ylabel('Histogram',size=20)
plt.legend(fontsize=20, loc=1)
plt.title('mi', size=30)
plt.xticks([-0.071, -0.07, -0.069, -0.068, -0.067])
plt.grid(False)

    
ax2 = plt.subplot(1,2,2)
Ca_model[np.diag_indices_from(Ca_model)] = None
Ca_data[np.diag_indices_from(Ca_data)] = None

plt.hist(Ca_model.reshape(-1), bins=50, density=True, edgecolor = 'black',\
         color='skyblue', alpha=0.7, label='Ca_model')
plt.hist(Ca_data.reshape(-1), bins=50, density=True, edgecolor = 'black',\
         color='green', alpha=0.7, label='Ca_data')
plt.ylabel('Histogram',size=20)
plt.title('Ca', size=30)
plt.legend(fontsize=20, loc=1)
plt.grid(False)

plt.savefig("D:/PMI/Projects/LLDL/task/Bethe/TestResults/Momonts_{}_new.png"\
            .format(boundary), dpi=200, bbox_inches='tight')











