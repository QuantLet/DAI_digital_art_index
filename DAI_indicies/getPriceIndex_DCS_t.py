# import numpy as np
import autograd.numpy as np 
from autograd_gamma import gammaln


class DCSt_filtering(object):
    
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
        
        # self.sigmaVsq = [] 
        # self.sigmaBSq = [] 
        
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
        
        return actualDate
    
    def ParameterEstimate(self):
        n_t_inv = []
        resSq = []
        for i in range(0, self.Time):
            logi1 = np.array(self.datePrecision == i+1).astype(int)
            # a_t = (1,...,1)'
            self.n_t.append(sum(logi1))
            # fixed effect estimated beta
            self.Beta.append((1/self.n_t[i]) * np.dot(self.residualsReg, logi1))
            
            # fixed effect estimated residuals 
            resSq.append(np.square(np.dot(self.residualsReg, logi1) - self.Beta[i]))
            self.a.append(logi1)
            n_t_inv.append(1/self.n_t[i])
        
        # a_it
        self.a = np.stack(self.a)
    
    def UpdateBeta(self,  params):
        sigmaVsq, kappa ,nu = params
        e_list=[]
        uPrevious = np.zeros(self.Time)
        
        for t in range(1, self.Time):
            # for t-1
            previous = np.where(self.datePrecision==t)
            aPrevious = self.a[t-1][previous]
            aPreviousT = aPrevious.T
            etaPrevious = np.dot(aPrevious, self.Beta[t-1])
            ePrevious = self.residualsReg[previous] - etaPrevious
            ePreviousT = ePrevious.T
            nSp_pre = self.n_t[t-1]
            
            self.Beta[t] = self.phi * self.Beta[t-1] + kappa * uPrevious[t-1]
            
            wPrevious = 1 + (1 / (nu*sigmaVsq)) * np.dot(ePrevious, ePreviousT)
            uPrevious[t] = (1 / wPrevious) * ((nu+nSp_pre) / nu) * (1/sigmaVsq) * np.dot(aPreviousT, ePrevious)
           
            
            e_list.append(ePrevious)
        
        return self.Beta, uPrevious, e_list
        
    def ComputeLikelihood(self, params):
        sigmaVsq, kappa ,nu = params
        likLi = []
        uPrevious = np.zeros(self.Time)
        
        for t in range(1, self.Time):
                        
            previous = np.where(self.datePrecision==t)
            aPrevious = self.a[t-1][previous]
            aPreviousT = aPrevious.T
            etaPrevious = np.dot(aPrevious, self.Beta[t-1])
            ePrevious = self.residualsReg[previous] - etaPrevious
            ePreviousT = ePrevious.T
            nSp_pre = self.n_t[t-1]
            
            self.Beta[t] = self.phi * self.Beta[t-1] + kappa * uPrevious[t-1]
            
            wPrevious = 1 + (1 / (nu*sigmaVsq)) * np.dot(ePrevious, ePreviousT)
            uPrevious[t] = (1 / wPrevious) * ((nu+nSp_pre) / nu) * (1/sigmaVsq) * np.dot(aPreviousT, ePrevious)
            
            likLi.append(gammaln((nu+nSp_pre)/2)-gammaln(nu/2)-(nSp_pre/2)*np.log(np.pi*nu* sigmaVsq)-(nu+nSp_pre)/2 * np.log(wPrevious))
            
        
        return -1*sum(likLi)
