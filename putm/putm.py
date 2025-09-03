import numpy as np
from sklearn.base import BaseEstimator
from sympy import ff
from utils import prepare_weighted_pu_data


class PUbasic(BaseEstimator):
   
    def __init__(self, clf):
        self.clf = clf
    def fit(self, X, y):
        self.clf.fit(X,y)
        return self

    def predict(self, X):
        return self.clf.predict()
        
    def predict_proba(self, Xtest):
        return self.clf.predict_proba(Xtest)    
        
    
class PUtm(BaseEstimator):

    def __init__(self, clf, clf_ex, epochs=100, epsilon=1e-4):
        self.clf = clf
        self.clf_ex = clf_ex
        self.epochs = epochs
        self.epsilon = epsilon

        
    def fit(self, X, s):
        
        model_naive = PUbasic(self.clf)
        model_naive.fit(X,s)
        sx = model_naive.predict_proba(X)[:,1]
        
        ex = (sx+1)/2
        
        prev_loss = 0 #Cas convergence stop
        for i in np.arange(self.epochs):
        # Model for posterior probability:
            Xtemp, stemp, weights = prepare_weighted_pu_data(X,s,ex,sx)
            self.clf.fit(Xtemp,stemp,sample_weight=weights)
            yx = self.clf.predict_proba(X)[:,1]
            
            
            hat_c = np.mean(s) 
            
            yx1 = yx[np.where(s==1)]    
            val_thrs = np.quantile(yx1,q=hat_c)
            sel1 = np.where(yx>val_thrs)    
            sel2 = np.where(s==1)    
            sel = np.union1d(sel1,sel2)    
            
            if sel.shape[0]>0:
                Xsel = X[sel,:]
                ssel = s[sel]
            else:
                Xsel = X
                ssel = s
                
            # ToDo Cas all labels are 1 cannot fit, therefore this hacks
            if np.sum(ssel)==np.shape(ssel):
                print(f"PUtm cannot fit all labels are 1 {i}")
                break
            # Model for propensity score: 
            self.clf_ex.fit(Xsel,ssel)
            ex = self.clf_ex.predict_proba(X)[:,1] 
                
            too_small = np.where(ex<sx)[0]
            if too_small.shape[0]>0:
                ex[too_small] = sx[too_small]
            
            sx = ex*yx 

            #Cas implement convergence stop         
            eps = 1e-7
            loss = -np.mean(s * np.log(sx + eps) + (1 - s) * np.log(1 - sx + eps))
            
                

            if (np.abs(prev_loss - loss) < self.epsilon):
                print(f"PUtm converged at epoch {i}")
                break

            prev_loss = loss

        #Build final model:
        Xtemp, stemp, weights = prepare_weighted_pu_data(X,s,ex,sx)
        self.clf.fit(Xtemp,stemp,sample_weight=weights)
            
            
        return self

    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_proba(self, Xtest):
        return self.clf.predict_proba(Xtest)        
    
    
    
