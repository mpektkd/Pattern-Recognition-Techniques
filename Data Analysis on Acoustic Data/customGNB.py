# %% [code]
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

   
def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    #X_digit_avg = []   # list to append average values
    #
    #index = np.where(y == digit)  # find where y is equal to 'digit'
    #for i in range (0,16,1):
    #    for j in range (0,16,1):  # for each pixel
    #        pixel = (i,j)
    #        avg_tmp = digit_mean_at_pixel(X, y, digit, pixel) # calculate avg 
    #        X_digit_avg.append(avg_tmp)
    #
    #return np.array(X_digit_avg)    # transform list to ndarray and return
    return np.mean(X[y==digit], axis=0)



def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    #X_digit_var = []   # list to append variance values
    #
    #index = np.where(y == digit)  # find where y is equal to 'digit'
    #for i in range (0,16,1):
    #    for j in range (0,16,1):  # for each pixel
    #        pixel = (i,j)
    #        var_tmp = digit_variance_at_pixel(X, y, digit, pixel) # calculate var
    #        X_digit_var.append(var_tmp)
    #
    #return np.array(X_digit_var)    # transform list to ndarray and return
    return np.var(X[y==digit], axis=0)


  

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance

        self.n_samples = 0  #number of samples  
        self.n_feats = 0    # number of features
        self.n_classes = 0   # number of classes
        self.classes_ = np.array([])    # tags of the classes

        self.X_mean_ = np.array([]) # array of means for each class
        self.X_var_ = np.array([])  # array of variance for each class

        self.pC = np.zeros((9,)) # pC is a vector with the probability of each class
        #self.pxC = np.zeros((x.shape[-1],9)) # pxC is an array with all probabilities p(xi|C)
        
        

    def _mean(self, X, y): 
        X_temp = [] # list to append means 
        for i in range (0,int(self.n_classes),1):    # for all digits
            X_temp.append(digit_mean(X,y,i))    # calculate means

        return np.array(X_temp)


    def _var(self, X, y): 
        X_temp = [] # list to append variance 
        for i in range (0,int(self.n_classes),1):    # for all digits
            X_temp.append(digit_variance(X,y,i))    # calculate variance

        return np.array(X_temp)
    

    def _prior(self,y): 
        """ Prior probability, P(y) for each y """
        pC = np.zeros((int(self.n_classes),))  # create an array to save prior probabilities
        for dig in y: 
            pC[int(dig)-1] += 1   
        pC /= y.shape[0]  # calculate probabilities for every class

        return pC


    def _normal(self,x,mean,var): 
        """ Gaussian Normal Distribution """
        try:
            multiplier = (1/ float(np.sqrt(2 * np.pi * var))) 
            exp = np.exp(-((x - mean)**2 / float(2 * var)))     # create pdf of normal distribution
            product = multiplier * exp
        except:
            product = 0.0   # if var is the nearest value to zero that is not 0 -> we get product=0

        return product


    def _observation(self, x, c):
        """Uses Normal Distribution to get, P(x|C) = P(x1|C) * P(x2|C) .. * P(xn|C)
        
        Args:
            x (np.ndarray): 1D-Array (nfeatures) 
            c (int): class

        Returns:
            (int): Observation Probability for that class    
        """
        pdfs = []
        for i in range(self.n_feats):
            mu = self.X_mean_[c][i]     # mean
            var = self.X_var_[c][i] if self.X_var_[c][i] > 1e-2 else 1e-2   # threshold variance for better results

            pdfs.append( self._normal(x[i],mu,var) )    # calculate pdfs

        pxC = np.prod(pdfs)

        return pxC
        
        
    def fit(self, X, y):
        self.n_samples, self.n_feats = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = np.array([int(x) for x in np.unique(y)])

        self.X_mean_ = self._mean(X,y) 
        self.X_var_ = ( self._var(X,y) if self.use_unit_variance is False else np.ones((int(self.n_classes),int(self.n_feats))) ) 

        self.pC = self._prior(y)
        #self.pxC = self._observation(X)

        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        samples, feats = X.shape
        result = []

        for i in range(samples):    # for every sample
            posterior = []
            for c in self.classes_:
                posterior.append( self._observation(X[i],c-1)*self.pC[c-1] )    # calculate posterior

            idx = np.argmax(posterior)+1  #find argmax
            result.append(self.classes_[idx])
        
        return np.array(result)

   
    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        y_pred = self.predict(X)    # calculate predictions 
        cor = 0   # variable to calculate correct predictions
        for i in range(0, len(y),1): # for all the test set
            if (y_pred[i] == y[i]): # compare prediction with actual result
                cor += 1  # if correct, count it

        return cor/len(y)  # calculate accuracy and return
