from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def SVD_analysis(X):
    """
    X: numpy array of shape (n_samples, n_features)
    Output: numpy array of shape (n_samples, n_components)
    """
    svd = TruncatedSVD(n_components=2)
    svd.fit(X)
    return svd.transform(X)

def SVD_graph(X, name = None):
    """
    X: numpy array of shape (n_samples, n_features)
    Output: Figure
    """
    X_transformed = SVD_analysis(X)
    plt.figure(plt.figsize(10, 8))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('SVD Analysis')
    if name is not None:
        plt.savefig(name)
    else:
        plt.show()