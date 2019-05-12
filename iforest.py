
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix


class Node(object):
    def __init__(self, X, e, q, p, left, right, n_type=''):
        self.X = X  # input data
        self.e = e  # depth
        self.q = q  # split on attribute q
        self.p = p  # decision value p
        self.size = len(X)
        self.left = left
        self.right = right
        self.n_type = n_type  # InNode or ExNode


class IsolationTree:
    def __init__(self, X,e,height_limit):
        self.X = X  # input data
        self.e = 0  # current height
        self.height_limit = height_limit  # l in the paper
        self.n_nodes = 0
        # self.num_instances = len(X)
        self.num_features = X.shape[1]  # Q in the paper
        self.root = self.fit(X,e)

    def fit(self, X:np.ndarray, e, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.e = e
        if self.e >= self.height_limit or len(X) <= 1 or (X == X[0]).all():
            self.n_nodes += 1
            return Node(X, e, None, None, None, None,n_type='exNode')
        else:
            # randomly select an attribute q
            q = np.random.randint(self.num_features)
            compare = X[:,q]
            low = min(compare)
            high = max(compare)
            if low == high:
                # randomly select a split point p from max and min
                attri_list = np.random.permutation(self.num_features)
                for i in range(self.num_features):
                    q = attri_list[i]
                    compare = X[:,q]
                    low = compare.min()
                    high = compare.max()
                    if low != high:
                        break
            p = np.random.uniform(low=low, high=high)
            # Xl <-- filter(X,q < p)
            # Xl = X[X[:,q]<p]
            left_index = X[:, q] < p
            # xr <-- filter(X,q >= p)
            # xr = X[X[:,q]>=p]
            right_index = np.invert(left_index)
            self.n_nodes += 1
            # return inNode{Left <-- iTree(Xl,e+1,l),Right <-- iTree(Xr,e+1,l),splitAtt<--q, SplitValue<--p}
            return Node(X, e, q, p, left=self.fit(X[left_index], e+1, improved=improved),right=self.fit(X[right_index],e+1, improved=improved), n_type='inNode')
        return self.root


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.height_limit = np.ceil(np.log2(self.sample_size)) - 1
        
    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.size = len(X)
        # set height limit l = ceiling(log2sub-samplingsize)
        # for i =1 to t do
        for i in range(self.n_trees):
            # X' <-- sample(X,sub-samplingsize)
            sampled = random.sample(range(len(X)), self.sample_size)
            X_sub = X[sampled]
            
            # forest <-- forest union itree(X',0,l)
            itree = IsolationTree(X_sub,0,self.height_limit)
            self.trees.append(itree)
        return self
    
    def cn(self,n):
        """
        Average path length of unsuccessful search in BST
        n: a data set of n instances
        """
        return 2*(np.log(n-1)+0.5772156649) - (2*(n-1)/n)

    def path(self,x_i,node,e=0):
        """
        x_i is an instance in X
        """      
        if node.n_type == 'exNode':
            if node.size == 1:
                return node.e
            else:
                return node.e + self.cn(node.size)
        else:
            if x_i[node.q] < node.p:
                return self.path(x_i,node.left,e+1)
            else:
                return self.path(x_i,node.right,e+1)

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        nd_X = []
        for x_i in X:
            x_i_path_len = []
            for itree in self.trees:
                x_i_path_len.append(self.path(x_i, itree.root))
            nd_X.append(np.mean(x_i_path_len))
        return np.array(nd_X).reshape(len(X), 1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        c_whole = self.cn(self.sample_size)
        return np.exp2((-self.path_length(X)/c_whole))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return scores >= threshold

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)
                       

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    c = 1
    for i in range(100):
        comp = (scores>=c).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, comp).ravel()
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)
        if desired_TPR <= tpr:
            return c, fpr
        c -= 0.01
