# =============================================================================
# TAP free energy
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
from model.InverseIsing.InverseIsing import TAP
print("Modules imported!")

#%% load data
# data_path = "D:/PMI/Projects/LLDL/task/Bethe/TestResults"
# networks = ['under_para']
# boundaries = ['LB','ITER','RB','FULL']
# parameters = ['J','h']
# model_para = {}
# print("loading data...")
# for network in networks:
#     model_para[network] = {}
#     if network == 'shallow':
#         for parameter in parameters:
#             path = data_path + "/{}/{}.txt".format(network, parameter)    
#             model_para[network][parameter] = np.loadtxt(path)
#     else:
#         for boundary in boundaries:
#             for parameter in parameters:
#                 path = data_path + "/{}/{}_{}.txt".format(network, parameter, boundary)
#                 model_para[network]["{}_{}".format(parameter, boundary)] = np.loadtxt(path)
# print("data loaded!")



#%% TAP free energy
beta = 1
tap = TAP(beta)
M = 2000

def TAPFreeEnergy_distribution(tap, J, h, M):
    TAPFs = []
    for i in range(M):
        tap_m, _ = tap.stationary_points(J, h, 1e-5)
        tap_F = tap.TAP_free_energy(J, h, tap_m)
        TAPFs.append(tap_F)
        print("\rComputing {}/{}".format(i+1,M), end='')
    print("\n")
    return np.array(TAPFs).reshape(-1)

#%%% model
# for network in networks:
#     if network == 'shallow':
#         print("Compute TAP: {};".format(network))
#         J = model_para[network]['J']
#         h = (model_para[network]['h']).reshape(-1,1)
#         TAP_Fs = TAPFreeEnergy_distribution(tap, J, h, M) 
#         np.savetxt("TAPF_{}.txt".format(network), TAP_Fs)
#     else:
#         for boundary in boundaries:
#             J = model_para[network]['J' + '_' + boundary]
#             h = (model_para[network]['h' + '_' + boundary]).reshape(-1,1)            
#             print("Compute TAP: {}, {};".format(network, boundary))
#             TAP_Fs = TAPFreeEnergy_distribution(tap, J, h, M) 
#             np.savetxt("TAPF_{}_{}.txt".format(network, boundary), TAP_Fs)


#%%% new test
# over-para
J = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/op/J_FULL.txt")
h = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/op/h_FULL.txt").reshape(-1,1)
TAP_Fs = TAPFreeEnergy_distribution(tap, J, h, M) 
np.savetxt("TAPF_op.txt", TAP_Fs)


#%% plot
import matplotlib.pyplot as plt

#%%% under para
plt.figure(figsize=(10,8),dpi=80)
ax1 = plt.subplot(2,2,1)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_under_para_FULL.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='FULL')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax1.transAxes, fontsize=10)
plt.grid(False)

ax2 = plt.subplot(2,2,2)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_under_para_LB.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='LB')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax2.transAxes, fontsize=10)
plt.grid(False)

ax3 = plt.subplot(2,2,3)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_under_para_ITER.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='ITR')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax3.transAxes, fontsize=10)
plt.grid(False)

ax4 = plt.subplot(2,2,4)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_under_para_RB.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='RB')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax4.transAxes, fontsize=10)
plt.grid(False)
plt.savefig('TAP_up.png', dpi=200, bbox_inches='tight')
plt.show()


#%%% over para
plt.figure(figsize=(10,8),dpi=80)
ax1 = plt.subplot(2,2,1)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_over_para_FULL.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='FULL')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax1.transAxes, fontsize=10)
plt.grid(False)

ax2 = plt.subplot(2,2,2)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_over_para_LB.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='LB')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax2.transAxes, fontsize=10)
plt.grid(False)

ax3 = plt.subplot(2,2,3)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_over_para_ITR.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='ITR')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax3.transAxes, fontsize=10)
plt.grid(False)

ax4 = plt.subplot(2,2,4)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_over_para_RB.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='RB')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nSingle peak"\
         .format(f.max()-f.min(), f.mean()),transform=ax4.transAxes, fontsize=10)
plt.grid(False)
plt.savefig('TAP_op.png', dpi=200, bbox_inches='tight')
plt.show()


#%%% shallow
plt.figure(figsize=(5,4),dpi=80)
ax1 = plt.subplot(111)
f = np.loadtxt("D:/PMI/Projects/LLDL/task/TAP/TAPF_shallow_J.txt")
plt.hist(f, bins=40, range=(f.min()-0.02,f.max()+0.02), density=True, \
         edgecolor = 'black', color='skyblue', label='shallow')
plt.ylabel('Histogram',size=15)
plt.xlabel('TAP free energy',size=15)
plt.legend(fontsize=15)
plt.text(0.54, 0.5, "Fmax-Fmin = {:.2e}\nFmean = {:.2f}\nTwo peaks"\
         .format(f.max()-f.min(), f.mean()), transform=ax1.transAxes, fontsize=10)
plt.grid(False)
plt.savefig('TAP_sl.png', dpi=200, bbox_inches='tight')
plt.show()
