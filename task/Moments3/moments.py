# =============================================================================
# Moments compute
# =============================================================================
import numpy as np

def moments(sigmas):
    M = sigmas.shape[0]
    N = sigmas.shape[1]
    moment3 = 0
    moment2 = 0
    moment1 = sigmas.mean(0)
    print("Calculating the moments: first moment finished")  
    for i in range(M):
        sigma = sigmas[i].reshape(1,N)
        m2 = np.dot(sigma.T, sigma) 
        m3 = np.tensordot(m2.reshape(N,N,1), sigma, [2,0])
        moment2 += m2
        moment3 += m3
        if (i+1)%10==0:
            print("\rCalculating the second and third moments: {:.2f}%".format((i+1)/M*100), end='')
    moment3 = moment3/M
    moment2 = moment2/M
    return moment1, moment2, moment3

def moment3_unique(moment3):
    moment3u = []
    N = moment3.shape[0]
    for i in range(N):
        for j in range(i):
            for k in range(j):
                moment3u.append(moment3[i][j][k])
    return np.array(moment3u)


# data_sigma = np.load('data_sigma_index.npy')
mc_sigma = np.load('mc_sigma_index.npy')
print("Data is loaded!")
# print("Data shape: {}".format(data_sigma.shape))
print("MC shape: {}".format(mc_sigma.shape))




# compute
# mm moments, mm3u moments only i<j<k
mc_mm1, mc_mm2, mc_mm3 = moments(mc_sigma)
# data_mm1, data_mm2, data_mm3 = moments(data_sigma)

mc_mm3u = moment3_unique(mc_mm3) 
# data_mm3u = moment3_unique(data_mm3)


# save
# data moments
# data_mm = {}
# data_mm[1] = data_mm1*1
# data_mm[2] = data_mm2*1
# data_mm[3] = data_mm3*1
# data_mm[4] = data_mm3u*1

# mc moments
mc_mm = {}
mc_mm[1] = mc_mm1*1
mc_mm[2] = mc_mm2*1
mc_mm[3] = mc_mm3*1
mc_mm[4] = mc_mm3u*1

# np.save("data_mm.npy", data_mm)
np.save("mc_mm.npy", mc_mm)
























