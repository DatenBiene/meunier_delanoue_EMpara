import numpy as np
from EMbase_c import expectation, covariance
from scipy.stats import multivariate_normal

__all__ = ['EM']


class EM(object):

    def __init__(self, n_components, tol=1e-3, max_iter = 100, random_state=None, means_init=None, covariances_init=None):

        self.n_components=n_components
        self.tol = tol
        self.max_iter = max_iter
        # Initialization ?
        self.random_state = None
        self.fitted=False

        if means_init is not None: # keep init ? 
            self.means_=means_init
            self.covariances_=covariances_init

        assert self.n_components==self.means_.shape[0]


    def _m_step(self, X, resp):

        self.weights_ = resp.mean(axis=0)
        self.means_ = np.array([np.average(X, axis=0, weights=resp[:,i]) for i in range(resp.shape[1]) ])
        self.covariances_ = np.array([covariance(X, resp[:,i], self.means_[i]) for i in range(resp.shape[1]) ])


    def _e_step(self, X):
        return expectation(X,self.means_,self.covariances_)


    def _initialise(self):
        pass

    def fit(self,X):
        self.fit_predict(X)
        return self


    def fit_predict(self, X):

        log_likelihood = -np.infty

        for n_iter in range(1, self.max_iter+1):
            prev_log_likelihood = log_likelihood

            resp = self._e_step(X)

            self._m_step(X, resp)

            # Cythonize !!!
            log_likelihood = np.log(np.sum([k*multivariate_normal(self.means_[i],self.covariances_[j]).pdf(X) for k,i,j in zip(self.weights_,range(self.means_.shape[0]),range(self.covariances_.shape[0]))]))
            print(log_likelihood)
            if abs(log_likelihood-prev_log_likelihood)<self.tol:
                print("Convergence in {}".format(n_iter))
                break

        self.fitted = True

        return resp.argmax(axis=1)


    def predict_proba(self, X):
        if not self.fitted:
            print("Classifier is not fitted, please call fit first")
            return
        else:
            prediction = np.zeros((X.shape[0], self.n_components))
            for idx in range(self.n_components):
                # Cythonize !!!
                prediction[:,idx] = multivariate_normal(mean=self.means_[idx],cov=self.covariances_[idx]).pdf(X)/np.sum([multivariate_normal(mean=self.means_[i],cov=self.covariances_[i]).pdf(X) for i in range(self.n_components)])

            return prediction


    def predict(self,X):
        if not self.fitted:
            print("Classifier is not fitted, please call fit first")
            return
        else:
            return self.predict_proba(X).argmax(axis=1)
