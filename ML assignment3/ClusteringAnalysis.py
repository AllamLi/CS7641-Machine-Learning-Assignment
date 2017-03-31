from time import time
import pandas
import numpy as np
import matplotlib.pyplot as plt

import pickle

from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.datasets import load_digits

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import GaussianRandomProjection

from sklearn.neural_network import MLPClassifier

def input(path):
    data = pandas.read_csv(path, sep=',')
    instances = len(data.values)
    features = len(data.values[0])-1
    X = data.values[:, 0:features]
    y = data.values[:, features]

    return X, y

def test_KMeans(estimator, digits, labels, name):
    train_X = scale(digits.data)
    estimator.fit(train_X)
    Y = estimator.labels_

    n_digits = len(np.unique(labels))

    #inertia = estimator.inertia_
    colors = plt.cm.rainbow(np.linspace(0, 1, n_digits))
    #shapes = ["o", "v", "s", "p", "*", "x", "+"]
    print(n_digits)
    shapes = ["o"] * n_digits

    plt.figure()

    f1 = 1
    f2 = 2


    print('%d, %d' % (n_digits, len(estimator.cluster_centers_)))

    for i in range(n_digits):
        for j in range(len(estimator.cluster_centers_)):
            msk = (digits.target == i) & (Y == j)
            print(len(msk))
            plt.scatter(digits.data[msk, i], digits.data[msk, j],
                        color=colors[i], marker=shapes[j])
    plt.title("K-means")
    plt.show()

def show_result(estimator, data, labels, Y, name):
    print('\n' + name + ': ')

    print('ARI: {}'.format(metrics.adjusted_rand_score(labels, Y)))
    print('AMI: {}'.format(metrics.adjusted_mutual_info_score(labels, Y)))

    # How good are the clusters?
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, Y)))
    print('Completeness: {}'.format(metrics.completeness_score(labels, Y)))
    print('Silhouette: {}'.format(metrics.silhouette_score(data, Y)))

def test1(data, labels, name):
    print('-' * 20 + 'experiment1: ' + name + '\n')
    #------K means
    n_labels = len(np.unique(labels))
    km = KMeans(n_clusters=n_labels)

    train_X = scale(data)
    km.fit(train_X)
    y = km.labels_


    #-------EM
    #em = GMM(n_components=n_labels)
    #y = em.fit_predict(train_X)
    em = GaussianMixture(n_components=n_labels)
    y = em.fit(train_X)


    return km, em


def test2(data, labels, N, name):
    print('\n' + '-'*20 + 'experiment2: ' + name + '\n')

    scaler = StandardScaler()
    scaler.fit(data)

    #------PCA
    for i in range(1, 64):
        print('\nkeep components = %d' % i)
        #print(data)
        print('\nPCA:')
        pca = PCA(n_components=i)
        output_X = pca.fit_transform(data)

        eigvals = np.linalg.eigvals(pca.get_covariance())
        expl_var = sum(pca.explained_variance_ratio_)
        R = scaler.inverse_transform(pca.inverse_transform(output_X))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R-data))
        # print('Eigenvalues:')
        # print('{}'.format(eigvals))
        print('Explained variance (%): {}'.format(expl_var))
        print('Reconstruction error: {}'.format(R_error))

        #------ICA
        print('ICA:')
        ica = FastICA(n_components=i)
        output_X = ica.fit_transform(data)

        R = scaler.inverse_transform(ica.inverse_transform(output_X))
        R_error = sum(map(np.linalg.norm, R - data))
        print('Reconstruction error: {}'.format(R_error))

        #-------RP
        print('RP: ')
        rp = GaussianRandomProjection(n_components=i)
        output_X = rp.fit_transform(data)

        inv = np.linalg.pinv(rp.components_)
        R = scaler.inverse_transform(np.dot(output_X, inv.T))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R))
        print('Reconstruction error: {}'.format(R_error))

        #-------LDA
        print('SVD: ')
        svd = TruncatedSVD(n_components=i)
        output_X = svd.fit_transform(data)
        R = scaler.inverse_transform(svd.inverse_transform(output_X))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R - data))
        print('Reconstruction error: {}'.format(R_error))



def test3(projection, data, labels, name):
    print('-' * 20 + 'experiment3: ' + name + '\n')
    input_data = projection.fit_transform(data)
    km, em = test1(input_data, labels, name)

    show_result(km, input_data, labels, km.labels_, name + ' km')
    show_result(em, input_data, labels, em.predict(input_data), name + ' em')


def test4(data, labels, name):
    print('\n' + '-' * 20 + 'experiment4: ' + name)
    NN = MLPClassifier()
    y = labels

    pca = PCA(n_components=5)
    X = pca.fit_transform(data)
    #X = data
    output = NN.fit(X, y).predict(X)
    accuracy = metrics.accuracy_score(y, output) * 100
    print('PCA accuracy: %.2f%%' % accuracy)

    ica = FastICA(n_components=5)
    X = ica.fit_transform(data)
    output = NN.fit(X, y).predict(X)
    accuracy = metrics.accuracy_score(y, output) * 100
    print('ICA accuracy: %.2f%%' % accuracy)

    rp = GaussianRandomProjection(n_components=5)
    X = rp.fit_transform(data)
    output = NN.fit(X, y).predict(X)
    accuracy = metrics.accuracy_score(y, output) * 100
    print('RP accuracy: %.2f%%' % accuracy)

    svd = TruncatedSVD(n_components=5)
    X = svd.fit_transform(data)
    output = NN.fit(X, y).predict(X)
    accuracy = metrics.accuracy_score(y, output) * 100
    print('SVD accuracy: %.2f%%' % accuracy)

def test5(data, labels, name):
    print('\n' + '-' * 20 + 'experiment5: ' + name)
    NN = MLPClassifier()
    n_components = 15
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(data)
    size = len(X)
    print(size)
    km, em = test1(X, labels, "PCA ")
    new_input = np.zeros((size, n_components+1))
    new_input[:, 0:n_components] = X
    new_input[:, n_components] = km.labels_

    output_y = NN.fit(new_input, labels).predict(new_input)
    accuracy = metrics.accuracy_score(labels, output_y) * 100

    output_y = NN.fit(X, labels).predict(X)
    ori_accuracy = metrics.accuracy_score(labels, output_y) * 100
    print('PCA accuracy %.2f%% vs %.2f%%' % (ori_accuracy, accuracy))

    # ica = FastICA()
    # X, y = ica.fit_transform(data, labels)
    #
    # km, em = test1(X, y, "ICA ")
    #
    # rca = GaussianRandomProjection()
    # X, y = rca.fit_transform(data, labels)
    # km, em = test1(X, y, "Random CA ")



if __name__ == '__main__':

   # np.random.seed(42)



    digits = load_digits()
    labels = digits.target

    data = scale(digits.data)


    #print(digits.data)

    #print("data.shape: \n")
    #print(data.shape)

    #
    # km, em = test1(data, labels, "test1")
    # show_result(km, data, labels, km.labels_, 'km')
    # show_result(em, data, labels, em.predict(data), 'em')

    # test2(data, labels, 10, 'PCA/ICA/RCA')

    # test3(PCA(n_components=61), data, labels, 'PCA')
    #
    # test3(FastICA(n_components=61), data, labels, 'ICA')
    #
    # test3(GaussianRandomProjection(n_components=20), data, labels, 'RCA')
    #
    # test3(TruncatedSVD(n_components=61), data, labels, 'SVD')

    # test4(data, labels, 'test4')

    test5(data, labels, "test5")

    # n_samples, n_features = data.shape
    #
    # n_digits = len(np.unique(digits.target))
    #
    # sample_size = 300
    #
    # km = KMeans(n_clusters=n_digits)
    #
    # test_KMeans(estimator=km, digits=digits, labels=labels, name="test")



