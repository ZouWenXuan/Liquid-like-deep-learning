# =============================================================================
# Entropy Landscape Analysis
# =============================================================================

import numpy as np

class EntropyLandscape:
    def __init__(self, beta):
        self.beta = beta
    
    def cavity(self, J, h, mi_a, trials_BP=5000):
        N = h.size
        mi_a[np.diag_indices_from(mi_a)] = 0
        for t in range(0, trials_BP):
            mi_a_temp = mi_a * 1
            mb_i = np.tanh(J)* mi_a
            mi_a = np.tanh( h.T + np.sum(np.arctanh(mb_i),axis=0).reshape(1,N)\
                                                       -np.arctanh(mb_i)).T
            mi_a[np.diag_indices_from(mi_a)] = 0
            delta = np.max(np.abs(mi_a - mi_a_temp))
            #F and E
            if delta <= 1e-6:
                #Free energy
                Hp = np.exp(h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1+mb_i)),axis=0))
                Hn = np.exp(-h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1-mb_i)),axis=0))
                beta_Fi = -np.log(Hp+Hn)
                beta_Fa = -np.log(np.cosh(J)*(1+np.tanh(J)*mi_a*mi_a.T))
                beta_f = (np.sum(beta_Fi) - 0.5*np.sum(beta_Fa))/N
                #Energy
                Gp = np.sum(np.exp(h.T)*(J*np.sinh(J)*(1+mb_i)\
                     +J*np.cosh(J)*(1-np.tanh(J)*np.tanh(J))*mi_a)\
                     *np.exp(np.sum(np.log(np.cosh(J)*(1+mb_i)),axis=0))\
                         .reshape(1,N)/(np.cosh(J)*(1+mb_i)),axis=0)
                Gn = np.sum(np.exp(-h.T)*(J*np.sinh(J)*(1-mb_i)\
                     -J*np.cosh(J)*(1-np.tanh(J)*np.tanh(J))*mi_a)\
                     *np.exp(np.sum(np.log(np.cosh(J)*(1-mb_i)),axis=0))\
                         .reshape(1,N)/(np.cosh(J)*(1-mb_i)),axis=0)
                beta_Ei = (h.T*(Hp-Hn)+Gp+Gn)/(Hp+Hn)
                beta_Ea = J*(np.tanh(J)+mi_a*mi_a.T)/(1+np.tanh(J)*mi_a*mi_a.T)
                beta_e = (-np.sum(beta_Ei)+0.5*np.sum(beta_Ea))/N
                mi = np.tanh(h.T + np.sum(np.arctanh(mb_i),axis=0).reshape(1,N)).reshape(N,1)
                return beta_f, beta_e, mi, mi_a
            if t == (trials_BP-1):
                print("Fail to converge after {} trials!".format(trials_BP))
                return beta_f, beta_e, mi, mi_a

    def entropyofd(self, J0, h0, x, sigma_star, mi_a):
        N = h0.size
        J = self.beta*J0
        h = (self.beta*h0).reshape(N,1) +(x*sigma_star).reshape(N,1)
        beta_f, beta_e, mi, mi_a = self.cavity(J, h, mi_a)
        q = np.sum(sigma_star.reshape(N,1)*mi)/N
        d = (1-q)/2        
        entropy_density = -beta_f + beta_e
        energy_density = beta_e + x*q
        return d, entropy_density, energy_density, mi_a
    
    def Landscape(self, beta, J0, h0, sigma_star, start, end, interval, direct):
        # initialized
        if ((direct!=1) and (direct!=-1)):
            raise ValueError("direct can only be 1 or -1!")
        N = h0.size        
        mi_a_init = np.random.normal(0,0.01,[N,N])
        Nd = int((end-start)/interval)+1
        D, S, E = np.zeros(Nd), np.zeros(Nd), np.zeros(Nd)
        # landscape
        mi_a_temp = 1*mi_a_init
        for j,x in enumerate(np.arange(start, end+interval, interval)):
            x = x*direct
            d, entropy_density, energy_density, mi_a = self.entropyofd\
                                            (J0, h0, x, sigma_star, mi_a_temp)
            D[j] = d
            S[j] = entropy_density
            E[j] = energy_density
            mi_a_temp = mi_a*1
        return D,S,E