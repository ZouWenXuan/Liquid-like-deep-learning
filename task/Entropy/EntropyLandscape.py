# =============================================================================
# Entropy analysis
# =============================================================================

#%% import module
import sys
sys.path.append("D:/PMI/Projects/LLDL/model")
import numpy as np
from model.InverseIsing.EntropyAnalysis import EntropyLandscape
print("Modules imported!")


#%% set parameter
segment = {}
segment['LB'] = {}
segment['LB']['start'], segment['LB']['end'] = 0, 300
segment['ITER'] = {}
segment['ITER']['start'], segment['ITER']['end'] = 300, 525
segment['RB'] = {}
segment['RB']['start'], segment['RB']['end'] = 525, 675
segment['FULL'] = {}
segment['FULL']['start'], segment['FULL']['end'] = 0, 675

ModelPara_path = "D:/PMI/Projects/LLDL/task/Bethe/results/up"
boundaries = ["LB", "ITER", "RB", "FULL"]
J_dict, h_dict = {}, {}
for boundary in boundaries:
    J_dict[boundary] = np.loadtxt(ModelPara_path+"/J_{}.txt".format(boundary))
    h_dict[boundary] = np.loadtxt(ModelPara_path+"/h_{}.txt".format(boundary))

ea_sigma = np.loadtxt("D:/PMI/Projects/LLDL/task/Entropy/ref/up/El_up.txt")
sigma = ea_sigma[:,2:]

#%%% Entropy Analysis Manual test
beta = 1
entropy_landscape = EntropyLandscape(beta)
start = -3
end = 3
interval = 0.1
direct = -1
i = 1

EntropyLandscape_dict = {}
for boundary in boundaries:
    sigma_test = sigma[:,segment[boundary]['start'] : segment[boundary]['end']]
    J, h = J_dict[boundary], h_dict[boundary]
    Nd = int((end-start)/interval) + 1
    M = 1
    Ds = np.zeros([M, Nd])
    Ss = np.zeros([M, Nd])
    sigma_star = sigma_test[i]
    D,S,E = entropy_landscape.Landscape(beta, J, h, sigma_star, start, end, interval, direct)
    EntropyLandscape_dict[boundary] = {}
    EntropyLandscape_dict[boundary]['D'] = D*1
    EntropyLandscape_dict[boundary]['S'] = S*1
    print(boundary)
   

#%% plot
import matplotlib.pyplot as plt
ELd = EntropyLandscape_dict
labels = ['Layer: LB','Layer: ITR','Layer: RB', 'Layer: Full network']
color = ['lightcoral','coral','red','darkred']
boundaries = ["LB", "ITER1", "RB", "FULL"]
ls = ['--','-.',':','-']

plt.figure(figsize=(10,8))  
d = np.arange(0.01,1,0.01)
s = -d*np.log(d)-(1-d)*np.log(1-d)
plt.plot(d,s,c='gray',label='Upper bound',linewidth =2)
for i, boundary in enumerate(boundaries):
    D = ELd[boundary]['D']
    S = ELd[boundary]['S']
    plt.plot(D, S, c=color[i], linestyle=ls[i], label=labels[i], linewidth=2)
plt.legend(loc=1,fontsize=18,ncol=1)
plt.xlabel('d',size=30)
plt.ylabel('s(d)',size=30)  
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

