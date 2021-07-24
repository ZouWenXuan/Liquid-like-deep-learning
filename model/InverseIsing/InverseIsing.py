# =============================================================================
# Methods for Inverse Ising problem
# =============================================================================

import numpy as np

class Pseudo:
    def __init__(self, beta):
        self.beta = beta
    
    def initialized(self, N):
        # J diagonal without diagonal elements
        J = 1/N * np.random.randn(N,N)
        J = (J + J.T)/2
        J[np.diag_indices_from(J)] = 0
        h = np.zeros([N,1])
        return J, h
            
    def gradient(self, sigma, J, h):
        # sigma: [N,M]  
        N = sigma.shape[0]
        M = sigma.shape[1]
        hr = 1*h
        Jir = 1*J
        Hr = hr + np.dot(Jir, sigma)     
        exp = np.exp(-2*self.beta*sigma*Hr)
        frac = exp/(1 + exp)
        df_dJ = -2*self.beta*np.dot(sigma,(frac*sigma).T)   
        df_dh = -2*self.beta*frac*sigma
        
        # record f
        fr = np.log(1+np.exp(-2*self.beta*sigma*Hr))
        f = np.mean(fr)
        
        # If mini-batch considered (or not)
        df_dJ = df_dJ/M
        df_dh = (df_dh.mean(1)).reshape(N,1)
        
        return df_dJ, df_dh, f  
    
    def optimizer(self, Optimizer_J, Optimizer_h):
        self.opti_J = Optimizer_J
        self.opti_h = Optimizer_h
    
    def estimate(self, sigma, J, h, lr_J, lr_h):
        # sigma:[N,M]
        gJ, gh, f = self.gradient(sigma, J, h) 
        J_esti = self.opti_J.optimize(lr_J, J, gJ)
        h_esti = self.opti_h.optimize(lr_h, h, gh)
        # Symmetrize
        J_esti = (J_esti + J_esti.T)/2
        J_esti[np.diag_indices_from(J_esti)] = 0
        return J_esti, h_esti, gJ, gh, f


    def decimation(self, p, J, set_initial=False):   
        N = J.shape[0]
        # set initial
        if set_initial:
            # diagonal elements are zero (considered as decimated)
            diagonal_tuple = np.diag_indices_from(J)
            diagonal_index = N*diagonal_tuple[0]+diagonal_tuple[1]
            self.J_decimation = diagonal_index*1
            
        # decimation procedure
        N_remain = J.size - self.J_decimation.size
        N_deci = int(p*N_remain) 
        J_line = J.reshape(-1)
        
        # set decimation (last step) to 0 (for argsort)
        J_line[self.J_decimation] = 0
        N_deci_totol = N_deci + self.J_decimation.size
        J_argsort = np.argsort(np.abs(J_line))
        J_decimition = J_argsort[0:N_deci_totol]
        J_line[J_decimition] = 0
        self.J_decimation = J_decimition*1
        J = J_line.reshape(N, N)
        return J
    
    def tPLF(self, f, J, set_initial=False):
        N = J.shape[0]
        # set initial 
        if set_initial:
            x = 1
            self.PL_max = -N*f
        else:
            # exclude the diagonal elements
            x = 1-(self.J_decimation.size-N)/(J.size-N)    
        # compute tPLF
        PL = -N*f
        tPLF = PL - x*self.PL_max + (1-x)*N*np.log(2)
        return tPLF, x
    
    
    def threshold(self, delta, J, set_index):
        if set_index:
            # np.where, return a tuple([N_where], [N_where])
            self.threshold_index = np.where(J<=delta)
            N_where = self.threshold_index[0].size
        J[self.threshold_index] = 0
        return J, N_where

class Bethe:
    def __init__(self, beta):
        self.beta = beta

    def model_aver(self, J, h, trials_BP = 1000, r = 0.0, return_fix_point=False):
        N = h.size
        mi_a = np.random.normal(0,0.01,[N,N])
        mi_a[np.diag_indices_from(mi_a)] = 0
        mi_a_temp = 1*mi_a
        for t in range(0, trials_BP):
            print("\rTrials BP: {}".format(t), end='')
            mi_a_temp = r*mi_a_temp + (1-r)*mi_a
            mb_i = np.tanh(J)* mi_a
            mi_a = np.tanh( h.T + np.sum(np.arctanh(mb_i),axis=0)\
                           .reshape(1,N)-np.arctanh(mb_i)).T
            mi_a[np.diag_indices_from(mi_a)] = 0
            delta = np.mean(np.abs(mi_a - mi_a_temp))
            if delta <= 1e-4:
                mi = np.tanh(h.T + np.sum(np.arctanh(mb_i),axis=0)\
                             .reshape(1,N)).reshape(N,1)
                Ca = (np.tanh(J) + mi_a*mi_a.T)/(1+np.tanh(J) * mi_a*mi_a.T)
                if return_fix_point:
                    return mi_a, mb_i
                return mi, Ca, True
        print("")
        print("Fail to converge after {} trials!".format(trials_BP))
        mi = np.tanh(h.T + np.sum(np.arctanh(mb_i),axis=0)\
                     .reshape(1,N)).reshape(N,1)
        Ca = (np.tanh(J) + mi_a*mi_a.T)/(1+np.tanh(J) * mi_a*mi_a.T)            
        return mi, Ca, False
    
    def initialized(self, N):
        # J diagonal without diagonal elements
        J = 1/N * np.random.randn(N,N)
        J = J * J.T
        J[np.diag_indices_from(J)] = 0
        h = np.zeros([N,1])
        return J, h
    
    def optimizer(self, Optimizer_J, Optimizer_h):
        self.opti_J = Optimizer_J
        self.opti_h = Optimizer_h
    
    def mse(self, mi_data, mi_model, Ca_data, Ca_model):
        N = mi_data.size
        error_mi = 1/N * np.sum((mi_data-mi_model)**2)
        # Ca diagonal elements are set to 0
        error_Ca = 1/(N*(N-1)) * np.sum((Ca_data-Ca_model)**2)
        mse = np.sqrt(error_mi + error_Ca)
        return mse, np.sqrt(error_mi), np.sqrt(error_Ca)
    
    def free_energy(self, J, h, mi_a, mb_i):
        #Free energy
        Hp = np.exp(h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1+mb_i)),axis=0))
        Hn = np.exp(-h.T) *  np.exp(np.sum(np.log(np.cosh(J)*(1-mb_i)),axis=0))
        beta_Fi = -np.log(Hp+Hn)
        beta_Fa = -np.log(np.cosh(J)*(1+np.tanh(J)*mi_a*mi_a.T))
        beta_F = np.sum(beta_Fi) - 0.5*np.sum(beta_Fa)
        return beta_F
    
    def free_energy_decimation(self, J, h, p, set_initial=False):
        mi_a, mb_i = self.model_aver(J, h, return_fix_point=True)
        N = J.shape[0]
        if set_initial:
            delta_F = np.zeros(J.shape)
            F_origin = self.free_energy(J, h, mi_a, mb_i)
            for i in range(N):
                for j in np.arange(0,i,1):
                    J_deci = 1*J
                    J_deci[i][j] = J_deci[j][i] = 0
                    F_deci = self.free_energy(J_deci, h, mi_a, mb_i)
                    delta_F[i][j] = delta_F[j][i] = np.abs(F_deci - F_origin)
            # handle N diagonal elements
            J[np.diag_indices_from(J)] = 0
            N_deci = int(p*(N**2-N)) + N
            # decimation procedure
            J_line = J.reshape(-1)
            self.index_deci = np.argsort(J_line)[0:N_deci]
            J_line[self.index_deci] = 0
            J = J_line.reshape(N,N)
        else:
            J_line = J.reshape(-1)
            J_line[self.index_deci] = 0
            J = J_line.reshape(N, N)
        return J
    
    def learning(self, mi_data, Ca_data, Optimizer_J, Optimizer_h, trials,\
                 lr_J, lr_h, gamma, reg):
        N = mi_data.size
        mi_data = mi_data.reshape(N,1)
        Ca_data[np.diag_indices_from(Ca_data)] = 0
    
        # initialize the J,h
        J,h = self.initialized(N)
        self.optimizer(Optimizer_J, Optimizer_h)
        
        # record the learning process
        mse, error_mi, error_Ca = np.zeros(trials), np.zeros(trials), np.zeros(trials)
        
        for i in range(0, trials): 
            mi_model, Ca_model = self.model_aver(J, h)
            mse_trial, error_trial_mi, error_trial_Ca = self.mse\
                                        (mi_data, mi_model, Ca_data, Ca_model)
            mse[i], error_mi[i], error_Ca[i] = mse_trial, error_trial_mi, error_trial_Ca
            print("Learning Trial:{}, error:{}".format(i, mse_trial))
            
            # record the optimal
            if i==0:
                mse_optimal = mse_trial
                J_optimal, h_optimal = J*1, h*1
            else:
                if mse_trial < mse_optimal:
                    mse_optimal = mse_trial
                    J_optimal, h_optimal = J*1, h*1
         
            gJ = Ca_model - Ca_data
            gh = mi_model - mi_data
            if reg == 'None':
                pass
            elif reg == 'l2':
                gJ += gamma*J
            else:
                raise ValueError("Reg can only be l2 or None!")            
            J = self.opti_J.optimize(lr_J, J, gJ)
            h = self.opti_h.optimize(lr_h, h, gh)   
        return mse, error_mi, error_Ca, J_optimal, h_optimal

class TAP:
    def __init__(self, beta):
        self.beta = beta
    
    def stationary_points(self, J, h, tol, trials_TAP=2000):
        '''
        
        Parameters
        ----------
        J : [N,N]
            coupling.
        h : [N,1]
            external field.
        trials_TAP : int, optional
            max iteration. The default is 2000.

        Returns
        -------
        mi_t2 : [N,1]
            stationary points.
        converge : bool
            mark if the TAP equation converges.
            
        '''
        N = J.shape[0]
        mi_t0 = np.random.normal(0, 1, [N,1])
        mi_t1 = np.random.normal(0, 1, [N,1])        
        for i in range(trials_TAP):
            Hi = h + np.dot(J, mi_t1)
            Onsager_rt = mi_t0 * np.dot(J**2, (1-mi_t1)**2)
            mi_t2 = np.tanh(self.beta*Hi - self.beta**2*Onsager_rt)
            if np.max(np.abs(mi_t2-mi_t1)) < tol:
                return mi_t2, True
            mi_t0 = mi_t1*1
            mi_t1 = mi_t2*1
        print("Failed to find to stationary point!")
        return mi_t2, False
        
    def TAP_free_energy(self, J, h, m):
        '''
        
        Parameters
        ----------
        J : [N,N]
            coupling.
        h : [N,1]
            external field.
        m : [N,1]
            stationary point.

        Returns
        -------
        minus_beta_f : float
            TAP free energy.

        '''
        minus_beta_f = 1/2*self.beta*np.dot(m.T,np.dot(J,m)) + self.beta**2/4\
            *np.dot((1-m**2).T, np.dot(J**2, (1-m**2)))\
            -np.sum((1+m)/2*np.log((1+m)/2) + (1-m)/2*np.log((1-m)/2) )
        return minus_beta_f
