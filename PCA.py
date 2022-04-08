from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_analysis(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca.transform(X)

def graph_PCA(X, y):
    X_transformed = PCA_analysis(X)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
    plt.show()