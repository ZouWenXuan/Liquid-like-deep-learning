# =============================================================================
# t-SNE code
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import manifold

#%% data load
# for i in range(0, 5):
#     if i == 0:
#         data80 = np.loadtxt("data_{}.txt".format(i))
#         data10 = np.loadtxt("data0_{}.txt".format(i))
#     else:        
#         data1 = np.loadtxt("data_{}.txt".format(i))
#         data2 = np.loadtxt("data0_{}.txt".format(i))
#         data80 = np.concatenate((data80, data1), axis=0)
#         data10 = np.concatenate((data10, data2), axis=0)

# np.save("data_2w_80.npy", data80)
# np.save("data_2w_10.npy", data10)

#%% main
data1 = np.load('data_2w_80.npy')
data2 = np.load('data_2w_10.npy')

X1 = data1[:,1:]
X2 = data2[:,1:]
X = np.concatenate((X1,X2), axis=0)

Y1 = data1[:,0]
Y2 = data2[:,0]
Y = np.concatenate((Y1,Y2))

time_start = time.time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, verbose=2)
X_tsne = tsne.fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min) 
time_end = time.time()

print('The total time cost is {}'.format(time_end-time_start))
np.save("X_norm.npy", X_norm)

#%% plot
X_norm = np.load("X_norm.npy")
X1 = X_norm[0:20000]
X2 = X_norm[20000:40000]

# scatter
plt.scatter(X2[:,0], X2[:,1], s=1, c='g', marker='o', alpha=0.5)
plt.scatter(X1[:,0], X1[:,1], s=1, c=Y1, cmap=plt.cm.Blues, marker='o')

# label
plt.scatter(2, 2, s=40, c='g', marker='o',label='initial states')
plt.scatter(2, 2, s=40, c='b', marker='o',label='solution states')

plt.xlim(-0.03,1.03)
plt.ylim(-0.03,1.03)
plt.xlabel('Component-1',size=30)
plt.ylabel('Component-2',size=30)
plt.legend(loc=1,fontsize=(18))
plt.subplots_adjust(wspace = 0.4, hspace =0)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)



