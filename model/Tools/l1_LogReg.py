# =============================================================================
# Interior-point method for l1-logreg
# =============================================================================
import numpy as np

class l1_logreg:
    '''
    Parameters
    ----------
    tol_eta : salar,
        Interior point tolerance.
    tol_nt : salar,
        Newton method tolerance.
    lambda_l1 : scalar,
        regularization parameter.
    alpha : scalar,
        backtracking line search parameter.
    beta : scalar,
        backtracking line search parameter.
    max_ml_iter: scalar,
        Max iteration of main loop.
    max_bls_iter : scalar,
        Max iteration of backtracking line search.
    max_nt_iter : scalar,
        Max iteration of Newton method.
    mu : scalar,
        parameter for t update.
    s_min : scaler,
        parameter for t update.
    '''
    def __init__(self, tol_eta, tol_nt, lambda_l1, alpha, beta, max_ml_iter,
                 max_bls_iter, max_nt_iter, mu, s_min):
        self.tol_eta = tol_eta
        self.tol_nt = tol_nt
        self.lambda_l1 = lambda_l1
        self.beta = beta
        self.alpha = alpha
        self.max_ml_iter = max_ml_iter
        self.max_bls_iter = max_bls_iter
        self.max_nt_iter = max_nt_iter
        self.mu = mu
        self.s_min = s_min
    
    
    def set_initv(self, x, b):
        '''
        Parameters
        ----------
        x : [M,N] array,
            data feature.
        b : [M,1] array,
            data label.
            
        Returns
        -------
        Set initial values.

        '''
        self.M = x.shape[0]
        self.N = x.shape[1]
        self.w = np.zeros([self.N,1])
        self.u = np.ones([self.N,1])   
        m_p = np.maximum((b[b>0]).size, 1)
        m_n = np.maximum((b[b<0]).size, 1)
        self.v = np.log(m_p/m_n)
        self.t = 1/self.lambda_l1
    
    def lambda_max(self, x, b):
        '''
        Parameters
        ----------
        x : [M,N] array,
            data feature.
        b : [M,1] array,
            data label.
            
        Returns
        -------
        lambda max

        '''        
        m_p = (b[b>0]).size
        m_n = (b[b<0]).size       
        b_tilde = np.where(b==1, m_n/self.M, -m_p/self.M)
        lambda_max = 1/self.M*np.linalg.norm(np.dot(x.T,b_tilde), ord=np.inf)
        return lambda_max
    

    def compute_search_direc(self, x, b, lambda_l1):
        '''
        Parameters
        ----------
        x : [M,N] array.
        b : [M,1] array.


        Variables
        ----------
        w : [N,1] array.
        u : [N,1] array.
        a : [M,N] array.
        v : scalar.
        z : [M,1] array.
        
        p_log : [M,1] array.
        df : [M,1] array, first derivative of f.
        d2f : [M,1] array, second derivative of f.
        
        g1 : [1,1] array, gradient of Phi_t over v. 
        g2 : [N,1] array, gradient of Phi_t over w.
        g3 : [N,1] array, gradient of Phi_t over u.
        
        D0 : [M,M] diagonal array.
        D1 : [N,N] diagonal array.
        D2 : [N,N] diagonal array.
        D3 : [N,N] diagonal array.
        
        Hred_1 : [1,1] array.
        Hred_2 : [1,N] array.
        Hred_3 : [N,1] array.
        Hred_4 : [N,N] array.
        Hred_up : [1,N+1] array.
        Hred_down : [N,N+1] array.
        Hred : [N+1, N+1] array.
        
        gred_up : [1,1] array.
        gred_down : [N,1] array.
        gred : [N+1,1] array.
        
        dv_dw : [N+1,1] array.
        dv_dw_du : [2N+1] array.
        
        
        Returns
        ----------
        dv_dw_du : [2N+1] array.
            search direction which are concatenated.
        
        gv_gw_gu : [2N+1] array.
            gradient of the main loss function phi_t.  
        
        '''
        # set intermediate variables
        a = b*x 
        z = np.dot(a, self.w) + self.v*b
        p_log = np.exp(z)/(1+np.exp(z))
        df = p_log-1
        d2f = -df*p_log
        
        # compute gradient g
        g1 = -(self.t/self.M)*np.dot(b.T,(1-p_log)) 
        g2 = -(self.t/self.M)*np.dot(a.T,(1-p_log)) + 2*self.w/(self.u**2-self.w**2)
        g3 = self.t*self.lambda_l1 - 2*self.u/(self.u**2-self.w**2)
        
        # compute hessian
        D0 = 1/self.M*np.diag(d2f.reshape(-1))
        D1 = np.diag( (2*(self.u**2+self.w**2)/(self.u**2-self.w**2)**2).reshape(-1) )
        D2 = np.diag( (-4*self.u*self.w/(self.u**2-self.w**2)**2).reshape(-1) )
        invD1 = np.linalg.inv(D1)
        D3 = D1 - np.dot( D2, np.dot(invD1, D2) )
        
        # reduced Newton system
        Hred_1 = self.t*np.dot(b.T, np.dot(D0,b))
        Hred_2 = self.t*np.dot(b.T, np.dot(D0,a))
        Hred_3 = self.t*np.dot(a.T, np.dot(D0,b))
        Hred_4 = self.t*np.dot(a.T, np.dot(D0,a))+ D3
        
        Hred_up = np.concatenate((Hred_1, Hred_2), axis=1)
        Hred_down = np.concatenate((Hred_3, Hred_4), axis=1)        
        Hred = np.concatenate((Hred_up, Hred_down), axis=0)
        
        gred_up = 1*g1
        gred_down = g2 - np.dot(D2, np.dot(invD1, g3))
        gred = np.concatenate((gred_up, gred_down), axis=0)
        
        # compute search direction
        dv_dw = -np.dot(np.linalg.inv(Hred), gred)
        dw = 1*dv_dw[1:,]
        du = -np.dot( invD1, (g3 + np.dot(D2,dw)) ) 
        
        # concatenate the delta and gradient
        dv_dw_du = np.concatenate((dv_dw, du), axis=0)
        gv_gw_gu = np.concatenate((g1, g2, g3), axis=0)
        return dv_dw_du, gv_gw_gu
    
    
    def phi_t(self, x, b, v, w, u):
        a = b*x 
        z = np.dot(a, w) + v*b
        fz = np.log(1+np.exp(-z))
        l_avg = np.mean(fz)
        phi_t = self.t*l_avg + self.t*self.lambda_l1*np.sum(u)\
                - np.sum(np.log(u**2-w**2))
        return phi_t
    
    
    def bls_main(self, x, b, dv_dw_du, gv_gw_gu):
        '''
        Parameters
        ----------
        dv_dw_du : [2N+1] array.
            search direction which are concatenated.
        
        gv_gw_gu : [2N+1] array.
            gradient of the main loss function phi_t.  

        Returns
        -------
        lr : beta^k, learning rate, scalar

        '''
        lr = 1
        for k in range(self.max_bls_iter):
            # old phi_t
            phi_t_old = self.phi_t(x, b, self.v, self.w, self.u)\
                      + self.alpha*lr*np.dot(gv_gw_gu.T, dv_dw_du)
            # new phi_t
            dv = 1*dv_dw_du[0:1,0:1]
            dw = 1*dv_dw_du[1:self.N+1,0:1]
            du = 1*dv_dw_du[self.N+1:2*self.N+1,0:1]
            phi_t_new = self.phi_t(x, b, self.v+lr*dv,\
                                   self.w+lr*dw, self.u+lr*du)
            if phi_t_old >= phi_t_new:
                return lr
            lr *= self.beta  
            if k == self.max_bls_iter-1:
                print('Backtracking line search in main loop fails.')
                return lr  


    def optimize_v(self, x, b):
        # Newton method
        v_bar = self.v*1
        fz = lambda z: np.log(1+np.exp(-z))
        for i in range(self.max_nt_iter):
            # intermediate variables
            a = b*x 
            z = np.dot(a, self.w) + v_bar*b
            p_log = np.exp(z)/(1+np.exp(z))            
            df = p_log-1
            d2f = -df*p_log
            
            # compute search direction (eliminate t in g1 and Hred_1)
            gv = -(1/self.M)*np.dot(b.T, (1-p_log))
            D0 = 1/self.M*np.diag(d2f.reshape(-1))
            Hv = np.dot(b.T, np.dot(D0,b))
            dv = -gv/Hv
            
            # newton decrement (actually positive here)
            nd = np.abs(gv**2/Hv/2) 
            if nd < self.tol_nt:
                return v_bar
            
            # backtracking line search to update
            lr = 1
            for k in range(self.max_bls_iter):
                l_avg_old = np.mean(fz(z))
                l_avg_new = np.mean(fz(z+lr*dv*b))+self.alpha*lr*gv*dv
                if l_avg_old>=l_avg_new:
                    break
                lr *= self.beta
                if k == self.max_nt_iter:
                    print('Backtracking line search in v optimization fails.')
            
            # update v_bar
            v_bar += lr*dv
            
            if i==self.max_nt_iter-1:
                print('Newton method fails to optimize v.')
                return v_bar
  
    
    def dual_point_theta(self, x, b, v_bar):
        a = b*x 
        z_bar = np.dot(a, self.w) + v_bar*b
        p_log_vbar = np.exp(z_bar)/(1+np.exp(z_bar))  
        s = np.minimum( self.M*self.lambda_l1/\
                       np.linalg.norm(np.dot(a.T,(1-p_log_vbar)), ord=np.inf), 1)
        theta = s/self.M*(1-p_log_vbar)
        return theta
        
    
    def f_conjugate(self, y):
        f_conj = -y*np.log(-y) + (1+y)*np.log(1+y)
        return f_conj
    
    
    def dual_function_G(self, theta):
        f_conjugate = lambda y: -y*np.log(-y) + (1+y)*np.log(1+y)
        # avoid numerical error
        theta[theta==0] = theta[theta==0] + 1e-40
        theta[theta==1] = theta[theta==1] - 1e-40
        # compute G
        G = -np.mean(f_conjugate(-self.M*theta))
        return G
    
    
    def dual_gap_eta(self, x, b):
        # l_avg
        a = b*x 
        z = np.dot(a, self.w) + self.v*b
        fz = np.log(1+np.exp(-z))
        l_avg = np.mean(fz)
        # l1_norm
        l1_norm = np.sum( np.abs(self.w) )
        # G(theta)
        v_bar = self.optimize_v(x, b)
        theta_bar = self.dual_point_theta(x, b, v_bar)
        eta = l_avg + self.lambda_l1*l1_norm \
            - self.dual_function_G(theta_bar)
        return eta
        
    
    def update_t(self, lr, eta):
        t_hat = 2*self.N/eta
        s = lr
        if s>=self.s_min:
            self.t = np.maximum(self.mu*np.minimum(t_hat, self.t), self.t)
        else:
            pass
        
    
    def train(self, x, b):
        for i in range(self.max_ml_iter):
            # compute search direction
            dv_dw_du, gv_gw_gu = self.compute_search_direc(x, b, self.lambda_l1)
            # backtracking line search
            lr = self.bls_main(x, b, dv_dw_du, gv_gw_gu)
            # update
            dv = 1*dv_dw_du[0:1,0:1]
            dw = 1*dv_dw_du[1:self.N+1,0:1]
            du = 1*dv_dw_du[self.N+1:2*self.N+1,0:1]       
            self.v += lr*dv
            self.w += lr*dw
            self.u += lr*du
            # compute dual gap
            eta = self.dual_gap_eta(x, b)
            # update t (lr = beta^k here)
            self.update_t(lr, eta)
            print("\rMain loop iteration: {}, dual gap: {:.2e}".format(i, eta), end='')            
            if np.abs(eta) < self.tol_eta:
                break
            
            