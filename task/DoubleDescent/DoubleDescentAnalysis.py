# =============================================================================
# Double Descent Analysis
# =============================================================================

#%% import module
import numpy as np
import matplotlib.pyplot as plt

#%% load data
network = 'FNN10'
task = '20(new)'
DD_dict = np.load('./TestResults/{}/DD_dict_{}_{}.npy'.format(network,network,\
                                              task), allow_pickle=True).item()
hiddens = 30
train_error = np.zeros((hiddens, 10000))
test_error = np.zeros(hiddens)
test_accuracy = np.zeros(hiddens)
for i in np.arange(1,hiddens+1,1):
    train_error[i-1] = DD_dict[i]['train_error']
    test_error[i-1] = DD_dict[i]['test_error']
    test_accuracy[i-1] = DD_dict[i]['test_accuracy']
p = 2000
h = np.arange(1,hiddens+1,1)
if network == 'FNN2' or network == 'Lrnet2': 
    alpha = (h**2 + (1+int(task))*h)/p
elif task == '20(new)':
    alpha = (h**2 + (10+20)*h)/p
else:
    alpha = (h**2 + (10+int(task))*h)/p
    

#%% delete nan
# delete_list = [3,6,7,8]
# test_accuracy = np.delete(test_accuracy, delete_list)
# test_error = np.delete(test_error, delete_list)
# alpha = np.delete(alpha, delete_list)
# train_error = np.delete(train_error, delete_list, axis=0)

#%% plot: Double Descent: FNN
plt.figure(figsize=(22,6))
ax1 = plt.subplot(121)
ax2 = ax1.twinx() 
label1, = ax1.plot(alpha, test_error, c='darkred', linewidth=2,\
                    marker='o', ms=5, mew=5)  
label2, = ax2.plot(alpha, train_error[:,-1], c='darkblue', linewidth=2,\
                    marker='o', ms=5, mew=5)
plt.legend([label1,label2], ['test error','train error'],
            bbox_to_anchor=(1.0, 0.6), fontsize=20, ncol=1) 
ax1.set_xlabel(r'$\alpha=\frac{M}{P}$',size=30) 
ax1.set_ylabel('test error',size=30)
ax2.set_ylabel('train error', size=30)
ax1.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)
ax1.set_ylim(0, 11)
ax2.set_ylim(0, 2.2)

# Interpolation threshold
start = 0
end = 1000
cm = plt.get_cmap("Blues_r")
cmap = [cm((i)/5) for i in range(5)]
peak = np.argmax(test_error)
IT_start = peak-2
IT_end = peak+2 + 1
alpha_IT = alpha[IT_start:IT_end]
train_error_IT = train_error[IT_start:IT_end,start:end]
step = range(end-start)
plt.subplot(122)
for i in range(5):
    plt.plot(step, train_error_IT[i], color=cmap[i], lw=3,
              label=r'$\alpha={}$'.format(alpha_IT[i]))
plt.xlabel('training epoch', size=30)
plt.ylabel('train error', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(loc=1, fontsize=18, ncol=2)
plt.annotate(r'Test error peak locates at $\alpha={}$'.format(alpha[peak]),\
              xy=(20, 0.01), xytext=(150, 0.3), ha='left', size = 20,\
              arrowprops=dict(width = 5, facecolor='black', shrink=0.1))
plt.subplots_adjust(wspace=0.5)
plt.savefig("./TestResults/{}/DD_{}_{}.png".format(network,network,task),\
            dpi=200, bbox_inches='tight')

#%%% test accuracy
# delete the first few alpha to make it clear
start = 3
plt.figure(figsize=(22,6))
ax1 = plt.subplot(121)
ax2 = ax1.twinx() 
label1, = ax1.plot(alpha[start:], test_error[start:], c='darkred', linewidth=2,\
                    marker='o', ms=5, mew=5)  
label2, = ax2.plot(alpha[start:], test_accuracy[start:], c='k', linewidth=2, ls='--',\
                    marker='o', ms=5, mew=5)
plt.legend([label1,label2], ['test error','test accuracy'],
            bbox_to_anchor=(1.0, 0.6), fontsize=20, ncol=1) 
ax1.set_xlabel(r'$\alpha=\frac{M}{P}$',size=30) 
ax1.set_ylabel('test error',size=30)
ax2.set_ylabel('test accuracy', size=30)
ax1.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)
# ax1.set_ylim(-1, 60)
# ax2.set_ylim(0.85, 0.92)
plt.savefig("./TestResults/{}/TestAccu_{}_{}.png".format(network,network,task),\
            dpi=200, bbox_inches='tight')

#%% plot: Double Descent: Lrnet
plt.figure(figsize=(22,6))
ax1 = plt.subplot(121)
ax2 = ax1.twinx() 
label1, = ax1.plot(alpha, test_error, c='darkred', linewidth=2,\
                    marker='o', ms=5, mew=5)  
label2, = ax1.plot(alpha, train_error[:,-1], c='darkblue', linewidth=2,\
                    marker='o', ms=5, mew=5)
label3, = ax2.plot(alpha, test_accuracy, c='k', linewidth=2,\
                    marker='o', ms=5, mew=5)
plt.legend([label1,label2,label3], ['test error','train error','test accuracy'],\
            bbox_to_anchor=(1.0, 0.6), fontsize=15, ncol=1) 
ax1.set_xlabel(r'$\alpha=\frac{M}{P}$',size=30) 
ax1.set_ylabel('test/train error',size=30)
ax2.set_ylabel('test accuracy', size=30)
ax1.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)
# ax1.set_ylim(0, 0.5)
# ax2.set_ylim(0.75, 0.9)

# plot trajactories
start = 0
end = 1000
cm = plt.get_cmap("Blues_r")
cmap = [cm((i)/6) for i in range(6)]
select = range(9,59,10)
alpha_s = alpha[select]
train_error_s = train_error[select, start:end]
step = range(end-start)
plt.subplot(122)
for i in range(5):
    plt.plot(step, train_error_s[i], color=cmap[i], lw=3,
              label=r'$\alpha={}$'.format(alpha_s[i]))
plt.plot(step, np.zeros(end-start), lw=3, ls='--', color='k')
plt.xlabel('training epoch', size=30)
plt.ylabel('train error', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(loc=1, fontsize=18, ncol=2)
plt.subplots_adjust(wspace=0.5)
plt.savefig("./TestResults/{}/DD_{}_{}.png".format(network,network,task),\
            dpi=200, bbox_inches='tight')



#%% train error: alpha
# train error: 2 class
network_task = ['FNN2_10', 'Lrnet2_10', 'Lrnet2_20', 'Lrnet2_30', 'Lrnet2_40', 'Lrnet2_50']
network = ['FNN2', 'Lrnet2']

train_error_final = {}
for nt in network_task:
    if nt == 'FNN2_10':
        hiddens = 30
        DD_dict = np.load('./TestResults/FNN2/DD_dict_{}.npy'.format(nt), allow_pickle=True).item()
    else:
        hiddens = 50
        DD_dict = np.load('./TestResults/Lrnet2/DD_dict_{}.npy'.format(nt), allow_pickle=True).item()
    train_error = np.zeros(hiddens)    
    for i in range(hiddens):
        train_error[i] = DD_dict[i+1]['train_error'][-1]
    train_error_final[nt] = train_error*1
    
cm = plt.get_cmap("Blues_r")
cmap = [cm((i)/8) for i in range(8)]
plt.figure(figsize=(22,6))
plt.subplot(121)
for i,nt in enumerate(network_task):
    if nt == 'FNN2_10':
        hiddens = 30
    else:
        hiddens = 50
    plt.plot(range(1,hiddens+1,1), train_error_final[nt], color=cmap[i], lw=3, label='{}'.format(nt))
plt.xlabel('hidden layer nodes', size=30)
plt.ylabel('train error of last step', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(loc=1, fontsize=18, ncol=2)

# train error: 10 class
network_task = ['FNN10_20', 'Lrnet10_10', 'Lrnet10_20', 'Lrnet10_30', 'Lrnet10_40', 'Lrnet10_50']
network = ['FNN10', 'Lrnet10']

train_error_final = {}
for nt in network_task:
    if nt == 'FNN10_20':
        hiddens = 30
        DD_dict = np.load('./TestResults/FNN10/DD_dict_FNN10_20(new).npy', allow_pickle=True).item()
    else:
        hiddens = 50
        DD_dict = np.load('./TestResults/Lrnet10/DD_dict_{}.npy'.format(nt), allow_pickle=True).item()
    train_error = np.zeros(hiddens)    
    for i in range(hiddens):
        train_error[i] = DD_dict[i+1]['train_error'][-1]
    train_error_final[nt] = train_error*1
    
cm = plt.get_cmap("Blues_r")
cmap = [cm((i)/8) for i in range(8)]
plt.subplot(122)
for i,nt in enumerate(network_task):
    if nt == 'FNN10_20':
        hiddens = 30
    else:
        hiddens = 50
    plt.plot(range(1,hiddens+1,1), train_error_final[nt], color=cmap[i], lw=3, label='{}'.format(nt))
plt.xlabel('hidden layer nodes', size=30)
plt.ylabel('train error of last step', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(loc=1, fontsize=18, ncol=2)
plt.subplots_adjust(wspace=0.3)
# plt.savefig("./TestResults/TranError.png", dpi=200, bbox_inches='tight')

