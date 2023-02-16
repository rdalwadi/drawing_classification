import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cycler import cycler

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import KernelPCA
#import umap

def plot_dendrogram(X, **kwargs):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(X)
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
    
    plt.figure(figsize=(12,8))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    
    
def plot2d(X,y, targets_encoder=None, n_components=50, n_neighbors=20, mode='kpca-knn', view_3d=False):
    if mode=='tsne':
        X_low = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(X)
        X_embedded = TSNE(n_components=2, perplexity=n_neighbors, init='pca').fit_transform(X_low)
    elif mode == 'knn':
        X_embedded = NeighborhoodComponentsAnalysis(n_components=2).fit_transform(X,y)
    elif mode == 'knn-tsne':
        X_embedded = NeighborhoodComponentsAnalysis(n_components=n_components).fit_transform(X,y)
        X_embedded = TSNE(n_components=4, perplexity=n_neighbors, init='pca', method='exact').fit_transform(X_embedded)
        X_embedded = NeighborhoodComponentsAnalysis(n_components=2).fit_transform(X,y)
    elif mode == 'umap':
        X_embedded = umap.UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(X)
    elif mode == 'kpca':
        X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(X)
    elif mode == 'kpca-knn':
        X_embedded = KernelPCA(kernel='cosine').fit_transform(X)
        X_embedded = NeighborhoodComponentsAnalysis(n_components=2).fit_transform(X_embedded,y)
    else:
        assert False, 'Unknown mode:%s'%mode
        
    fig = plt.figure(figsize=(15,9))
    NUM_COLORS = len(set(y))+1
    ax = fig.add_subplot(111)
    if view_3d == True:
        ax = plt.axes(projection = '3d')
    cm = plt.get_cmap('tab20')
    ax.set_prop_cycle(cycler(color=[cm(1.*i/NUM_COLORS) for i in range(1, NUM_COLORS+1)]))
    for i in sorted(set(y)):
        X_i = X_embedded[y==i]
        if targets_encoder is None:
            labels = '%s'%i
        else:
            labels = targets_encoder.classes_[i]
        ax.scatter(*X_i.T,s=60,alpha=1,label='%d %s [#%d]'%(i,labels ,len(X_i)))
    plt.legend()
    if view_3d == True:
        plt.axis('auto')
    else:
        plt.axis('equal')
    plt.show()
    
def plot2d_continous_target(X,target, n_components=40):
    svd = TruncatedSVD(n_components=n_components, random_state=42).fit(X)
    X_low = svd.transform(X)
    X_embedded = TSNE(n_components=2, perplexity=30, init='pca').fit_transform(X_low)
    plt.figure(figsize=(15,9))
    plt.scatter(*X_embedded.T,s=20,alpha=.9, c=target,cmap='hot',lw=1,edgecolor='k')
    plt.axis('equal')
    plt.show()

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    if ylim is not None:
        axes[2].set_ylim(*ylim)
    axes[2].set_title("Performance of the model")

    return plt