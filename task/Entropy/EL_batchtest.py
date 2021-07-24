# =============================================================================
# Entropy analysis
# =============================================================================

import numpy as np

# functions        
def cavity_method(J,h,mi_a):
    N = J.shape[0]
    mi_a[np.diag_indices_from(mi_a)] = 0
    for t in range(0,3000):
        mi_a_temp = mi_a * 1
        mb_i = np.tanh(J)* mi_a
        mi_a = np.tanh( h.T + np.sum(np.arctanh(mb_i),axis=0).reshape(1,N)-np.arctanh(mb_i)).T
        mi_a[np.diag_indices_from(mi_a)] = 0
        delta = np.max(np.abs(mi_a - mi_a_temp))
        #计算自由能和能量
        if delta <= 1e-6:
            #自由能密度
            Hp = np.exp(h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1+mb_i)),axis=0))
            Hn = np.exp(-h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1-mb_i)),axis=0))
            beta_Fi = -np.log(Hp+Hn)
            beta_Fa = -np.log(np.cosh(J)*(1+np.tanh(J)*mi_a*mi_a.T))
            beta_f = (np.sum(beta_Fi) - 0.5*np.sum(beta_Fa))/N
            #能量密度
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
        if t==2999:
            log = open("log.txt", "a")
            print('failed...',file=log)
            log.close() 
            return beta_f,beta_e,mi,mi_a

def S_d(N,beta,J0,h0,D,S,E,layer,ref,direct,sigma_star):
    mi_a_temp = np.random.normal(0,0.01,[N,N])
    J = beta*J0
    for j, x_n in enumerate(np.arange(-3,3.05,0.05)):
        x = direct * x_n
        h = (beta*h0).reshape(N,1) +(x*sigma_star).reshape(N,1)
        beta_f,beta_e,mi,mi_a = cavity_method(J,h,mi_a_temp)
        q = np.sum(sigma_star.reshape(N,1)*mi)/N
        d = (1-q)/2
        D[layer][ref][j] = d
        S[layer][ref][j] = -beta_f + beta_e
        E[layer][ref][j] = beta_e+x*q
        mi_a_temp = mi_a

# main test
Sn = np.zeros([4,60,121])
Dn = np.zeros([4,60,121])
En = np.zeros([4,60,121])
Sp = np.zeros([4,60,121])
Dp = np.zeros([4,60,121])
Ep = np.zeros([4,60,121])
beta = 1
case = 'fa'
for j, name in enumerate(['./ref/{}/Eh_{}.txt'.format(case, case),\
                          './ref/{}/Em_{}.txt'.format(case, case),\
                          './ref/{}/El_{}.txt'.format(case, case)]):
    # load data
    data = np.loadtxt(name)
    for r in range(0,20):
        patterns = data[:,2:][r]
        for direct in [-1,1]:
            for i,k in enumerate(['LB','ITER','RB','FULL']):
                if i == 0:
                    start = 0
                    N = 300
                elif i == 1:
                    start = 300
                    N = 225
                elif i == 2:
                    start = 525
                    N = 150
                else:
                    start = 0
                    N = 675
                J = np.loadtxt('D:/PMI/Projects/LLDL/task/Bethe/results/{}/J_{}.txt'.format(case,k)).reshape(N,N)
                h = np.loadtxt('D:/PMI/Projects/LLDL/task/Bethe/results/{}/h_{}.txt'.format(case,k)).reshape(1,N)
                sigma_star = patterns[start:(start+N)]
                if direct == -1:
                    S_d(N,beta,J,h,Dn,Sn,En,i,r+j*20,direct,sigma_star)
                elif direct == 1:
                    S_d(N,beta,J,h,Dp,Sp,Ep,i,r+j*20,direct,sigma_star)
                print('Finish Name:{} refs:{} direct:{} layer:{}.'.format(name,r,direct,k))
S_D_E = np.concatenate((Sn,Dn,En,Sp,Dp,Ep),axis=0)
np.savetxt('S_D_E_{}.txt'.format(case), S_D_E.reshape(-1))
