# =============================================================================
# Secant method for iso-energy landscape
# =============================================================================

import numpy as np

#%% landscape functions
def cavity_method(J,h,mi_a):
    N = J.shape[0]
    mi_a[np.diag_indices_from(mi_a)] = 0
    for t in range(0,5000):
        mi_a_temp = mi_a * 1
        mb_i = np.tanh(J)* mi_a
        mi_a = np.tanh( h.T + np.sum(np.arctanh(mb_i),axis=0).reshape(1,N)-np.arctanh(mb_i)).T
        mi_a[np.diag_indices_from(mi_a)] = 0
        delta = np.max(np.abs(mi_a - mi_a_temp))
        # F and E
        if delta <= 1e-4:
            # f
            Hp = np.exp(h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1+mb_i)),axis=0))
            Hn = np.exp(-h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1-mb_i)),axis=0))
            beta_Fi = -np.log(Hp+Hn)
            beta_Fa = -np.log(np.cosh(J)*(1+np.tanh(J)*mi_a*mi_a.T))
            beta_f = (np.sum(beta_Fi) - 0.5*np.sum(beta_Fa))/N
            # e
            Gp = np.sum(np.exp(h.T)*(J*np.sinh(J)*(1+mb_i)
                 +J*np.cosh(J)*(1-np.tanh(J)*np.tanh(J))*mi_a)
                 *np.exp(np.sum(np.log(np.cosh(J)*(1+mb_i)),axis=0)).reshape(1,N)/(np.cosh(J)*(1+mb_i)),axis=0)
            Gn = np.sum(np.exp(-h.T)*(J*np.sinh(J)*(1-mb_i)
                 -J*np.cosh(J)*(1-np.tanh(J)*np.tanh(J))*mi_a)
                 *np.exp(np.sum(np.log(np.cosh(J)*(1-mb_i)),axis=0)).reshape(1,N)/(np.cosh(J)*(1-mb_i)),axis=0)
            beta_Ei = (h.T*(Hp-Hn)+Gp+Gn)/(Hp+Hn)
            beta_Ea = J*(np.tanh(J)+mi_a*mi_a.T)/(1+np.tanh(J)*mi_a*mi_a.T)
            beta_e = (-np.sum(beta_Ei)+0.5*np.sum(beta_Ea))/N
            mi = np.tanh(h.T + np.sum(np.arctanh(mb_i),axis=0).reshape(1,N)).reshape(N,1)
            return beta_f,beta_e,mi,mi_a
        if t==4999:
            print('failed to converge.')
            return 0,0,0,0
        
def S_d(beta,J0,h0,direct,sigma_star):
    N = J0.shape[0]
    mi_a_temp = np.random.normal(0,0.01,[N,N])
    J = beta*J0
    D = np.zeros(121)
    S = np.zeros(121)
    E = np.zeros(121)
    for j, x_n in enumerate(np.arange(-3,3.05,0.05)):
        x = direct * x_n
        h = (beta*h0).reshape(N,1) +(x*sigma_star).reshape(N,1)
        beta_f,beta_e,mi,mi_a = cavity_method(J,h,mi_a_temp)
        q = np.sum(sigma_star.reshape(N,1)*mi)/N
        d = (1-q)/2
        D[j] = d
        S[j] = -beta_f + beta_e
        E[j] = beta_e+x*q
        mi_a_temp = mi_a  
    return D,S,E


#%% secant method
def secantF(J0, h0, beta, x, sigma_star, mi_a_temp):
    N = J0.shape[0]
    J = beta*J0
    h = (beta*h0).reshape(N,1) +(x*sigma_star).reshape(N,1)
    beta_f,beta_e,mi,mi_a = cavity_method(J,h,mi_a_temp)
    q = np.sum(sigma_star.reshape(N,1)*mi)/N
    e = (beta_e + x*q)/beta 
    return e, mi_a
    

def secant_method(beta_0, beta_1, sigma_star, x ,J0, h0, e00, mi_a_temp):
    itermax = 1000 
    x_0 = beta_0
    x_1 = beta_1
    e_0, _ = secantF(J0, h0, x_0, x, sigma_star, mi_a_temp)
    e_1, mi_a1 = secantF(J0, h0, x_1, x, sigma_star, mi_a_temp)
    f_0 = e_0 - e00
    f_1 = e_1 - e00
    
    if np.sign(f_0)==np.sign(f_1):
        print('no solutions.')
        return 0
    for i in range(itermax):
        x_new = (x_0*f_1-x_1*f_0)/(f_1-f_0)
        e_new, _ = secantF(J0, h0, x_new, x, sigma_star, mi_a_temp)
        f_new = e_new - e00
        if np.sign(f_0)==np.sign(f_new):
            x_0 = x_new
            f_0 = f_new
        else:
            x_1 = x_new
            f_1 = f_new        
        if (np.fabs(f_new)<1e-5 or np.fabs((x_1-x_0)<1e-3)):
            break
        if i == (itermax-1):
            print('Max iteration, no solution.')
    beta = x_new
    return beta    

def sd_isoE(J0, h0, e00, sigma_star, direct):
    N = J0.shape[0]
    mi_a_temp = np.random.normal(0,0.01,[N,N])
    D = np.zeros(121)
    S = np.zeros(121)
    E = np.zeros(121)
    B = np.zeros(121)
    for j, x_n in enumerate(np.arange(-3,3.05,0.05)):
        x = direct * x_n
        print("x:{:.2f} is trained.".format(x))   
    
        if np.abs(x)>=2:
            beta_0, beta_1 = 0.1,5
        else:
            beta_0, beta_1 = 0.1,3
  
        beta = secant_method(beta_0,beta_1,sigma_star,x,J0,h0,e00,mi_a_temp)
        J = beta*J0
        h = (beta*h0).reshape(N,1) +(x*sigma_star).reshape(N,1)
        beta_f,beta_e,mi,mi_a = cavity_method(J,h,mi_a_temp)
        q = np.sum(sigma_star.reshape(N,1)*mi)/N
        d = (1-q)/2
        s = -beta_f + beta_e
        e = (beta_e+x*q)/beta
        D[j] = d
        S[j] = s
        E[j] = e
        B[j] = beta
        mi_a_temp = mi_a*1          
    return D,S,E,B


#%% manual test for one ref sigma
J0 = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/up/J_FULL.txt")
h0 = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/up/h_FULL.txt")

sigma_index = 0
sigma = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/data/up/El_up.txt")
sigma_star = sigma[sigma_index,2:]
e00 = sigma[sigma_index,0]/J0.shape[0]

direct = -1
D,S,E,B = sd_isoE(J0,h0,e00,sigma_star,direct)
D1,S1,E1 = S_d(1, J0, h0, -1, sigma_star)
direct = 1
Dp,Sp,Ep,Bp = sd_isoE(J0,h0,e00,sigma_star,direct)
Dp1,Sp1,Ep1 = S_d(1, J0, h0, -1, sigma_star)


#%%% manual test plot
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5),dpi=200)
x = np.arange(-3,3.05,0.05)
plt.subplot(1,3,1)
plt.plot(D, E, linewidth=1, c='b', label=r'$\epsilon$')
plt.plot(x, B, linewidth=1, c='g', label =r'$\beta$')
plt.plot(D, S, linewidth=1, c='orange', label=r'$s(d)$')
plt.legend()

plt.subplot(1,3,2)
a = np.arange(0.01,1.01,0.01)
sub = -a*np.log(a)-(1-a)*np.log(1-a)
plt.plot(D, S, lw=2, c='b', label='N')
plt.plot(a,sub,linewidth=1,c='gray',label='S-ub')
plt.legend()
plt.title(r'$s(d)-d$')

#%% batch test
J0 = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/up/J_FULL.txt")
h0 = np.loadtxt("D:/PMI/Projects/LLDL/task/Bethe/results/up/h_FULL.txt")
sigma = np.loadtxt("D:/PMI/Projects/LLDL/task/Entropy/ref/up/El_up.txt")
M = 10
Ds = np.zeros([M,2,121])
Ss = np.zeros([M,2,121])
Es = np.zeros([M,2,121])
Bs = np.zeros([M,2,121])
for sigma_index in range(M):
    sigma_star = sigma[sigma_index,2:]
    e00 = sigma[sigma_index,0]/J0.shape[0]
    for j,direct in enumerate([-1,1]):
        print('\nProcess step {}/20'.format((sigma_index)*2+j+1))
        D,S,E,B = sd_isoE(J0,h0,e00,sigma_star,direct)
        Ds[sigma_index][j] = D*1
        Ss[sigma_index][j] = S*1
        Es[sigma_index][j] = E*1
        Bs[sigma_index][j] = B*1

#%% save data
S_D_E_iso = {}
S_D_E_iso['D'] = Ds*1
S_D_E_iso['S'] = Ss*1
S_D_E_iso['E'] = Es*1
S_D_E_iso['B'] = Bs*1
np.save("S_D_E_iso.npy", S_D_E_iso)