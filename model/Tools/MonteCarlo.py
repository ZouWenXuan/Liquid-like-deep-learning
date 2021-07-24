# =============================================================================
# Metropolis Monte Carlo sampling
# =============================================================================
import numpy as np

class MonteCarlo:
    '''
    
    Parameters
    ----------
    beta : scalar
        inverse temperature.
    relaxation : scalar
        Number of Markov steps before sampling to relax
    interval : scalar
        Number of Markov steps between two samplings
    M : scalar
        Number of samples.

    '''
    
    def __init__(self, beta, relaxation, interval, M):
        self.beta = beta
        self.relaxation = relaxation
        self.interval = interval
        self.M = M
          

    def MarkovStep(self, J, h, sigma, algorithm='Metropolis-Hastings'):  
        '''
        
        Parameters
        ----------
        J : [N,N] array
            pair-wise coupling.
        h : [N,1] array
            external field.
        sigma : [N,1] array
            configuration at last step.
        algorithm : str
            Markov algorithms "Metropolis-Hastings" or "Heat-bath".
            
        Returns
        -------
        sigma : [N,1] array
            configuration at this step.
            
        '''
        
        N = J.shape[0]
        index_permutation = np.random.permutation(N)
        for spin in index_permutation:
            H = np.dot(J[spin:spin+1], sigma) + h[spin]
            if algorithm == 'Metropolis-Hastings':
                Delta_E = 2*H*sigma[spin]
                Trans = np.minimum(1, np.exp(-self.beta*Delta_E))
            elif algorithm == 'Heat-bath':
                Trans = 1/2*(1-sigma[spin]*np.tanh(self.beta*H))
            else:
                raise ValueError("Choose Metropolis-Hastings or Heat-bath!")  
            unirand = np.random.uniform(low=0, high=1)          
            Trans_bool = np.sign(unirand-Trans)
            sigma[spin] = sigma[spin]*Trans_bool
        return sigma
        
       

    def Sampling(self, J, h, algorithm = 'Metropolis-Hastings'):
        '''
        Parameters
        ----------
        J : [N,N] array
            pair-wise coupling.
        h : [N,1] array
            external field.           
        algorithm : str
            Markov algorithms "Metropolis-Hastings" or "Heat-bath".
            
        Returns
        -------
        sigmas : [M,N] array
            Monte Carlo samples array.

        '''
        N = J.shape[0]
        sigma = np.random.choice([1,-1], (N,1), p=[0.5,0.5])            
        for step in range(self.relaxation):
            sigma = self.MarkovStep(J, h, sigma, algorithm)
            print('\rRelaxation step: {}/{}'.format(step+1, self.relaxation), end='')  
        print('\nRelaxation finish. Begin sampling...')
        sigmas = []
        for i in range(self.M):
            for step in range(self.interval):
                sigma = self.MarkovStep(J, h, sigma)
            sigmas.append(sigma.reshape(-1)*1)
            print('\rSampling quantity: {}/{}'.format(len(sigmas),self.M),end='')
        print('\nSampling finish.')
        sigmas = np.array(sigmas)
        return sigmas
    
    