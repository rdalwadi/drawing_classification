import numpy as np
import scipy as sp
from copy import deepcopy, copy
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as normalize_func
from vectorizer import TfIdfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn import preprocessing
from sklearn.svm import SVC


class GaussianMixtureGenerativeModel(object):
    def __init__(self, confidence=.9, n_components_gm=1, covariance_type='full'):
        self.n_components_gm = n_components_gm
        self.covariance_type = covariance_type
        self.clf = None
        self.confidence = confidence
        self.threshold = None
        
    def fit(self, X):
        if self.n_components_gm > 1:
            n_components_gm = self.n_components_gm
        else:
            n_components_gm = max(1,int(len(X)*self.n_components_gm))
        self.clf = GaussianMixture(n_components=n_components_gm, covariance_type=self.covariance_type).fit(X)
        scores = self.clf.score_samples(X).reshape(-1)
        self.threshold = np.percentile(scores, self.confidence*100)
        return self
 
    def predict_proba(self, X):
        scores = self.clf.score_samples(X).reshape(-1)
        return scores 

    def predict(self, X):
        scores = self.predict_proba(X)
        preds = scores >= self.threshold
        return preds

    
class GaussianMixtureClassifier(object):
    def __init__(self, n_components_svd=300, n_components_gm=1, covariance_type='full', normalize=True):
        self.n_components_svd = n_components_svd
        self.n_components_gm = n_components_gm
        self.covariance_type = covariance_type
        self.normalize = normalize

    def fit(self, X,y):
        if X.ndim==2 and X.shape[1] > self.n_components_svd:
            self._svd = TruncatedSVD(n_components=self.n_components_svd).fit(X)
            X_low = self._svd.transform(X)
        else:
            X_low = X
        if self.normalize:
            X_low = normalize_func(X_low)
        self._clfs = []
        for i in sorted(set(y)):
            X_i = X_low[y==i]
            if self.n_components_gm > 1:
                clf = GaussianMixture(n_components=self.n_components_gm, covariance_type=self.covariance_type).fit(X_i)
            else:
                n_components_gm = max(1,int(len(X_i)*self.n_components_gm))
                clf = GaussianMixture(n_components=n_components_gm, covariance_type=self.covariance_type).fit(X_i)
            self._clfs.append(clf)
        return self
    
    def predict_proba(self, X):
        if X.ndim==2 and X.shape[1] > self.n_components_svd:
            X_low = self._svd.transform(X)
        else:
            X_low = X
        if self.normalize:
            X_low = normalize_func(X_low)
        scores = np.hstack([clf.score_samples(X_low).reshape(-1,1) for clf in self._clfs])
        return scores 

    def predict(self, X):
        scores = self.predict_proba(X)
        preds = np.argmax(scores,axis=1).reshape(-1)
        return preds
    
    
class BayesianGaussianMixtureClassifier(object):
    def __init__(self, n_components_svd=300, n_components_gm=1, covariance_type='full', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=1e5, normalize=True):
        self.n_components_svd = n_components_svd
        self.n_components_gm = n_components_gm
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.normalize = normalize

    def fit(self, X,y):
        if X.ndim==2 and X.shape[1] > self.n_components_svd:
            self._svd = TruncatedSVD(n_components=self.n_components_svd).fit(X)
            X_low = self._svd.transform(X)
        else:
            X_low = X
        if self.normalize:
            X_low = normalize_func(X_low)
        self._clfs = []
        for i in sorted(set(y)):
            X_i = X_low[y==i]
            if self.n_components_gm > 1:
                nc = self.n_components_gm
            else:
                nc = max(1,int(len(X_i)*self.n_components_gm))
            clf = BayesianGaussianMixture(n_components=nc, 
                                          weight_concentration_prior_type=self.weight_concentration_prior_type,
                                          weight_concentration_prior=self.weight_concentration_prior,
                                          covariance_type=self.covariance_type).fit(X_i)
            self._clfs.append(clf)
        return self
    
    def predict_proba(self, X):
        if X.ndim==2 and X.shape[1] > self.n_components_svd:
            X_low = self._svd.transform(X)
        else:
            X_low = X
        if self.normalize:
            X_low = normalize_func(X_low)
        scores = np.hstack([clf.score_samples(X_low).reshape(-1,1) for clf in self._clfs])
        return scores 

    def predict(self, X):
        scores = self.predict_proba(X)
        preds = np.argmax(scores,axis=1).reshape(-1)
        return preds
    
    
class TextClassifier(object):
    def __init__(self, classifier, vectorizer=TfIdfVectorizer()):
        self.classifier = classifier
        self.vectorizer = vectorizer
    
    def fit(self, documents, targets):
        X = self.vectorizer.fit_transform(documents, targets)
        self.classifier.fit(X, targets)
        return self

    def predict_proba(self, documents):
        X = self.vectorizer.transform(documents)
        scores = self.classifier.predict_proba(X)
        return scores

    def predict(self, X):
        scores = self.predict_proba(X)
        preds = np.argmax(scores,axis=1).reshape(-1)
        return preds
    

class MultiLabelTextClassifier(object):
    def __init__(self, classifier=SVC(C=1e2, kernel='rbf', gamma='scale', probability=True), vectorizer=TfIdfVectorizer()):
        self.vectorizer = vectorizer
        self.classifier = ClassifierChain(classifier)
        
    def fit_vectorizer(self, documents, targets):
        flat_targets = [hash(tuple(t)) for t in targets]
        flat_targets = preprocessing.LabelEncoder().fit_transform(flat_targets)
        self.vectorizer = self.vectorizer.fit(documents, flat_targets)
        return self
        
    def fit(self, documents, targets):
        self.fit_vectorizer(documents, targets)
        self.classifier.fit(self.transform(documents), targets)
        return self
    
    def transform(self, documents):
        return self.vectorizer.transform(documents)
    
    def predict(self, documents):
        preds = self.classifier.predict(self.transform(documents))
        return preds
    
    def predict_proba(self, documents):
        preds = self.classifier.predict_proba(self.transform(documents))
        return preds

    def score(self, documents, targets):
        mean_score = self.classifier.score(self.transform(documents), targets)
        return mean_score