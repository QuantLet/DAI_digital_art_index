import numpy as np
# import autograd.numpy as np 

import pandas as pd
import time
import datetime
import os
import logging



class Kalman_filtering(object):
    
    def __init__(self, list_of_date, ols_resid, discR, phi):
        self.list_of_date = list_of_date
        self.ols_resid = ols_resid
        self.discR = discR
        self.phi = phi
        self.Time = []
        self.residualsReg = []
        self.datePrecision = []
        self.a = []
        self.n_t = []
        self.Beta = []
        self.sigmaUsq = []
        self.sigmaXiSqOLS = []
        self.sigmaBSq = []
        self.zeta = []
        
    def DataPreparation(self):
        T = self.list_of_date.astype("float")
        combo = np.c_[T,  self.ols_resid]
        T = combo[:,0]
        T = np.round(T/self.discR) * self.discR
        self.residualsReg = combo[:,1]
        self.datePrecision = np.repeat(1, len(self.residualsReg))
        
        T1 = T[0:len(T)-1]
        T2 = T[1:len(T)]
        actualDate = np.unique(T)
        diff_T1T2 = T2 - T1
        decide = np.array(diff_T1T2, dtype=bool).astype(int)
        
        for i in range(0, len(self.datePrecision)-1):
            self.datePrecision[i+1] = self.datePrecision[i] + decide[i]
        # num_obs = len(self.residualsReg)
        self.Time = max(self.datePrecision)
        
        return(actualDate)
    
    def ParameterEstimate(self):
        n_t_inv = []
        resSq = []
        
        for i in range(0, self.Time):
            logi1 = np.array(self.datePrecision == i+1).astype(int)
            self.n_t.append(sum(logi1))
            
            # fixed effect beta
            self.Beta.append((1/self.n_t[i]) * np.dot(self.residualsReg, logi1))
            resSq.append(np.square(np.dot(self.residualsReg, logi1) - self.Beta[i]))
            self.a.append(logi1)
            n_t_inv.append(1/self.n_t[i])
            
        self.a = np.stack(self.a)
        xi = np.array(self.Beta[1:]) - np.array(self.Beta[0:self.Time-1])
        
        self.sigmaUsq = (1/(1-1/self.Time * sum(n_t_inv))) * (1/self.Time) * np.dot(n_t_inv, resSq)
        
        self.sigmaXiSqOLS = ((self.Time-1)/self.Time) * np.var(xi) - self.sigmaUsq * 1/self.Time * sum(n_t_inv)
        
        self.sigmaBSq = (self.Time-1)/self.Time * np.dot(np.var(self.Beta), np.repeat(1, self.Time)) 
        
        self.zeta = np.dot(self.sigmaXiSqOLS, np.repeat(1, self.Time)) + self.sigmaBSq
        
        return self.Beta 
    
    def UpdateBeta(self, sigmaXiSq):
        for t in range(1, self.Time): 
            # prediction step
            current = np.where(self.datePrecision==t+1)
            aCurrent = self.a[t][current]
            
            ### i add this
            self.Beta[t] = self.phi * self.Beta[t-1]
            self.sigmaBSq[t] = np.square(self.phi) * self.sigmaBSq[t-1] + sigmaXiSq
            
            # etaCurrent = np.dot(aCurrent, self.Beta[t-1])
            etaCurrent = np.dot(aCurrent, self.Beta[t])
            e_t = self.residualsReg[current] - etaCurrent
            nSp = self.n_t[t]
            aTranspose = aCurrent.T
            
            sigmaEtaSq = aCurrent.dot(self.sigmaBSq[t]).dot(aTranspose) + self.sigmaUsq * np.identity(nSp)
            
            # correction step
            m = 10^-6
            self.Beta[t] = self.Beta[t] + np.dot(self.sigmaBSq[t], aTranspose).dot(np.linalg.inv(sigmaEtaSq+ np.eye(sigmaEtaSq.shape[1])*m)).dot(e_t)
            
            self.sigmaBSq[t] = self.sigmaBSq[t] - (np.square(self.sigmaBSq[t]) * aTranspose).dot(np.linalg.inv(sigmaEtaSq)).dot(aCurrent)
            
        # BetaT = self.Beta
        # for t in range(self.Time-2, -1, -1):
        #     BetaT[t] = self.Beta[t] +  self.phi * self.sigmaBSq[t] / self.sigmaBSq[t+1] * (BetaT[t+1] - self.Beta[t+1])
        # self.Beta = BetaT
        return self.Beta
        
    def ComputeLikelihood(self, sigmaXiSq):
        likLi = []
        for t in range(1, self.Time):
            # prediction step
            current = np.where(self.datePrecision==t+1)
            aCurrent = self.a[t][current]
            aTranspose = aCurrent.T
            nSp = self.n_t[t]
            
            ### i add this
            self.Beta[t] = self.phi * self.Beta[t-1]
            
            self.sigmaBSq[t] = np.square(self.phi) * self.sigmaBSq[t-1] + sigmaXiSq
            sigmaEtaSq = aCurrent.dot(self.sigmaBSq[t]).dot(aTranspose) + self.sigmaUsq * np.identity(nSp)
            
            #  etaCurrent = np.dot(aCurrent, self.Beta[t-1])
            etaCurrent = np.dot(aCurrent, self.Beta[t])
            e_t = self.residualsReg[current] - etaCurrent
            
            # correction step
            m = 10^-6
            self.Beta[t] = self.phi * self.Beta[t]+ np.dot(self.sigmaBSq[t], aTranspose).dot(np.linalg.inv(sigmaEtaSq+ np.eye(sigmaEtaSq.shape[1])*m)).dot(e_t)
            
            self.sigmaBSq[t] = self.sigmaBSq[t] - (np.square(self.sigmaBSq[t]) * aTranspose).dot(np.linalg.inv(sigmaEtaSq)).dot(aCurrent)
            
            self.zeta[t] = self.sigmaBSq[t] + sigmaXiSq
            zetaTT_1 = self.zeta[t-1]
            sigma_t_zeta = abs(2*(nSp-1) * np.log(np.sqrt(self.sigmaUsq))  + np.log(nSp*zetaTT_1+self.sigmaUsq))
            
            sigma_t_1_zeta = 1/(nSp*zetaTT_1+self.sigmaUsq)*np.dot(aCurrent,aTranspose)/nSp + 1/self.sigmaUsq*(np.identity(nSp)-np.dot(aCurrent,aTranspose)/nSp)
            
            likLi.append(0.5 * (sigma_t_zeta + e_t.dot(sigma_t_1_zeta).dot(e_t.T)))
            
            
        return sum(likLi)
