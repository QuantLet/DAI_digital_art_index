from math import gamma
import numpy as np
import pandas as pd
import time
import datetime
import os
import logging

from scipy.special import gammaln
import mpmath



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
        
        self.sigmaVsq = [] 
        
        # self.sigmaKUsqOLS = [] # sigmaXiSqOLS
        # self.u = []
        # self.k_u = []
        
        self.sigmaBSq = [] 
        
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
        
        # kappa * u_t
        # self.k_u = np.array(self.Beta[1:]) - np.array(self.Beta[0:self.Time-1])
        # self.u =self.k_u / self.kappa
        
        # # estimate of sigmaVsq 
        self.sigmaVsq = (1/(1-1/self.Time * sum(n_t_inv))) * (1/self.Time) * np.dot(n_t_inv, resSq)
        
        # # estimate sigmaKUsqOLS via beta_t - beta_(t-1)
        # self.sigmaKUsqOLS = ((self.Time-1)/self.Time) * np.var(self.k_u) - self.sigmaVsq * 1/self.Time * sum(n_t_inv)
        
        # estimate sigmaBSq
        self.sigmaBSq = (self.Time-1)/self.Time * np.dot(np.var(self.Beta), np.repeat(1, self.Time)) 
        # return(self.Beta)
    
    def UpdateBeta(self,  params):
        sigmaVsq, kappa ,nu = params
        u_list = []
        for t in range(2, self.Time):
            # prediction step
            # for t-1
            previous = np.where(self.datePrecision==t-1)
            aPrevious = self.a[t-2][previous]
            aPreviousT = aPrevious.T
            etaPrevious = np.dot(aPrevious, self.Beta[t-1])

            ePrevious = self.residualsReg[previous] - etaPrevious
            ePreviousT = ePrevious.T
            nSp_pre = self.n_t[t-2]

            wPrevious = 1 + (1 / (nu*sigmaVsq)) * np.dot(ePrevious, ePreviousT)
            uPrevious = (1 / wPrevious) * ((nu+nSp_pre) / nu) * (1/sigmaVsq) * np.dot(aPreviousT, ePrevious)
            u_list.append(wPrevious)
            
            self.Beta[t] = self.phi * self.Beta[t-1] + kappa * uPrevious
            
            self.sigmaBSq[t] = np.square(self.phi) * self.sigmaBSq[t-1] + np.square(kappa) * (np.square(nu)/((nu+3)*(nu+1)))      
            
                      
                                                                       
            # for t
            current = np.where(self.datePrecision==t)
            aCurrent = self.a[t-1][current]
            aCurrentT = aCurrent.T
            etaCurrent = np.dot(aCurrent, self.Beta[t])
            eCurrent = self.residualsReg[current] - etaCurrent
            eCurrentT = eCurrent.T
            
            nSp = self.n_t[t-1]

            sigmaEtaSq = aCurrent.dot(self.sigmaBSq[t]).dot(aCurrentT) + sigmaVsq * np.identity(nSp) 
            
            # correction step
            m = 10^-6
            self.Beta[t] = self.Beta[t] + np.dot(self.sigmaBSq[t], aCurrentT).dot(np.linalg.inv(sigmaEtaSq+ np.eye(sigmaEtaSq.shape[1])*m)).dot(self.residualsReg[current]-etaCurrent)
            # self.Beta[t] = self.Beta[t] + np.dot(self.sigmaBSq[t], aCurrentT).dot(np.linalg.inv(sigmaEtaSq)).dot(self.residualsReg[current]-etaCurrent)
            self.sigmaBSq[t] = self.sigmaBSq[t] - (np.square(self.sigmaBSq[t]) * aCurrentT).dot(np.linalg.inv(sigmaEtaSq)).dot(aCurrent)


        BetaT = self.Beta
        for t in range(self.Time-2, -1, -1):
            BetaT[t] = self.Beta[t] +  self.phi * self.sigmaBSq[t] / self.sigmaBSq[t+1] * (BetaT[t+1] - self.Beta[t+1])
        self.Beta = BetaT

        return(self.Beta)
        
    def ComputeLikelihood(self, params):
        sigmaVsq, kappa ,nu = params
        likLi = []
        for t in range(2, self.Time):
            # prediction step
            # for t-1
            previous = np.where(self.datePrecision==t-1)
            aPrevious = self.a[t-2][previous]
            aPreviousT = aPrevious.T
            etaPrevious = np.dot(aPrevious, self.Beta[t-1])
            ePrevious = self.residualsReg[previous] - etaPrevious
            ePreviousT = ePrevious.T
            nSp_pre = self.n_t[t-2]

            wPrevious = 1 + (1 / (nu*sigmaVsq)) * np.dot(ePrevious, ePreviousT)
            uPrevious = (1 / wPrevious) * ((nu+nSp_pre) / nu) * (1/sigmaVsq) * np.dot(aPreviousT, ePrevious)
            
            self.Beta[t] = self.phi * self.Beta[t-1] + kappa * uPrevious
            self.sigmaBSq[t] = np.square(self.phi) * self.sigmaBSq[t-1] +  np.square(kappa) * (np.square(nu)/((nu+3)*(nu+1)))
            
            # for t
            current = np.where(self.datePrecision==t)
            aCurrent = self.a[t-1][current]
            aCurrentT = aCurrent.T
            etaCurrent = np.dot(aCurrent, self.Beta[t])
            eCurrent = self.residualsReg[current] - etaCurrent
            eCurrentT = eCurrent.T
            nSp = self.n_t[t-1]
            
            sigmaEtaSq = aCurrent.dot(self.sigmaBSq[t]).dot(aCurrentT) + sigmaVsq * np.identity(nSp)
            
            # correction step
            m = 10^-6
            self.Beta[t] = self.Beta[t] + np.dot(self.sigmaBSq[t], aCurrentT).dot(np.linalg.inv(sigmaEtaSq+ np.eye(sigmaEtaSq.shape[1])*m)).dot(self.residualsReg[current]-etaCurrent)
            self.sigmaBSq[t] = self.sigmaBSq[t] - (np.square(self.sigmaBSq[t]) * aCurrentT).dot(np.linalg.inv(sigmaEtaSq)).dot(aCurrent)
            
            # analytics shortcut
            likLi.append(-(gammaln((nu+nSp_pre)/2)- gammaln(nu/2) - (nSp_pre/2)*np.log(np.pi*nu* sigmaVsq)-(nu+nSp)/2 * np.log(wPrevious)))            
        # return np.count_nonzero(np.isinf(likLi))
        return sum(likLi)
