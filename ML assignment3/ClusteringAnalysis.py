from time import time
import pandas
import numpy as np
import matplotlib.pyplot as plt
import time

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

def read_data(path):
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

def plot_curves(x, Y, xlabel, ylabel, curve_labels, name, save_name, flag=False):
    print(x)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(Y)))

    plt.figure()
    plt.title(name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (flag):
        plt.xticks(x, x)

    for (y, label, c) in zip(Y, curve_labels, colors):
        plt.plot(x, y, color=c, label=label, lw=2.0)

    plt.legend(loc='best')
    plt.savefig(save_name)

    return plt


def test1(X, y, min_n, max_n):

    score0 = np.zeros(max_n-min_n+1)
    score1 = np.zeros(max_n-min_n+1)

    for i in range(min_n, max_n+1):
        index = i - min_n

        km = KMeans(n_clusters=i)
        output_y = km.fit_predict(X)
        score0[index] = metrics.silhouette_score(X, output_y)
        print('km: n_cluster = %d: silhouette_score = %f' % (i, score0[index]))

        em = GaussianMixture(n_components=i)
        em.fit(X)
        output_y = em.predict(X)
        score1[index] = metrics.silhouette_score(X, output_y)
        print('em: n_cluster = %d: silhouette_score = %f' % (i, score1[index]))

    plot_curves(np.arange(min_n, max_n+1), (score0, score1), 'k', 'Silouette Score', ('k-means', 'EM'),
                'test1 for dataset1:\nsilouette score curves for different clusters', 'figures/test1_dataset1.png', flag=True)


def test2(X, y, N, name):
    print('\n' + '-'*20 + 'experiment2: ' + name + '\n')

    scaler = StandardScaler()
    scaler.fit(X)

    err0 = np.zeros(N-1)
    err1 = np.zeros(N-1)
    err2 = np.zeros(N-1)
    err3 = np.zeros(N-1)

    for i in range(1, N):
        print('\nkeep components = %d' % i)
        #print(data)
        # ------PCA
        print('\nPCA:')
        pca = PCA(n_components=i)
        output_X = pca.fit_transform(X)

        eigvals = np.linalg.eigvals(pca.get_covariance())
        expl_var = sum(pca.explained_variance_ratio_)
        R = scaler.inverse_transform(pca.inverse_transform(output_X))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R-X))
        # print('Eigenvalues:')
        # print('{}'.format(eigvals))
        print('Explained variance (%): {}'.format(expl_var))
        print('Reconstruction error: {}'.format(R_error))

        err0[i-1] = R_error

        #------ICA
        print('ICA:')
        ica = FastICA(n_components=i)
        output_X = ica.fit_transform(X)

        R = scaler.inverse_transform(ica.inverse_transform(output_X))
        R_error = sum(map(np.linalg.norm, R - X))
        print('Reconstruction error: {}'.format(R_error))
        err1[i - 1] = R_error

        #-------RP
        print('RP: ')
        rp = GaussianRandomProjection(n_components=i)
        output_X = rp.fit_transform(X)

        inv = np.linalg.pinv(rp.components_)
        R = scaler.inverse_transform(np.dot(output_X, inv.T))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R-X))
        print('Reconstruction error: {}'.format(R_error))
        err2[i - 1] = R_error

        #-------SVD
        print('SVD: ')
        svd = TruncatedSVD(n_components=i)
        output_X = svd.fit_transform(X)
        R = scaler.inverse_transform(svd.inverse_transform(output_X))  # Reconstruction
        R_error = sum(map(np.linalg.norm, R - X))
        print('Reconstruction error: {}'.format(R_error))
        err3[i - 1] = R_error

    plot_curves(np.arange(1, N), (err0, err1, err2, err3), 'keep components', 'Reconstruction Error', ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'test2 for dataset1:\nReconstruction Error for components analysis', 'figures/test2_dataset1.png')

def test3(X, y, n_labels, interval, n_features):
    projections = (PCA, FastICA, GaussianRandomProjection, TruncatedSVD)

    arange = np.arange(1, n_features, interval)

    score0 = np.zeros(len(arange))
    score1 = np.zeros(len(arange))
    score2 = np.zeros(len(arange))
    score3 = np.zeros(len(arange))

    scores = (score0, score1, score2, score3)

    index = -1

    print(arange)

    print('-'*20 + 'start KM:\n')
    for i in arange:
        index = index+1
        for (pro, sco) in zip(projections, scores):
            projection = pro(n_components=i)
            new_X = projection.fit_transform(X)

            km = KMeans(n_clusters=n_labels)
            output_y = km.fit_predict(new_X)
            #sco[index] = metrics.adjusted_mutual_info_score(y, output_y)
            sco[index] = metrics.adjusted_rand_score(y, output_y)
            #sco[index] = metrics.v_measure_score(y, output_y)

    plot_curves(arange, (score0, score1, score2, score3), 'keep components', 'ARI score', ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'test3 for dataset1:\nARI scores for different projections in k-means', 'figures/test3_dataset1_km.png',flag=True)

    index = -1
    print('-'*20 + 'start EM:')
    for i in arange:
        index = index+1
        for (pro, sco) in zip(projections, scores):
            projection = pro(n_components=i)
            new_X = projection.fit_transform(X)

            em = GaussianMixture(n_components=n_labels)
            em.fit(new_X)
            output_y = em.predict(new_X)
            #sco[index] = metrics.adjusted_mutual_info_score(y, output_y)
            sco[index] = metrics.adjusted_rand_score(y, output_y)
            #sco[index] = metrics.v_measure_score(y, output_y)

    plot_curves(arange, (score0, score1, score2, score3), 'keep components', 'ARI score', ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'test3 for dataset1:\nARI scores for features reduction algorithms in EM', 'figures/test3_dataset1_em.png',flag=True)


    plt.show()


# def test3(projection, data, labels, name):
#     print('-' * 20 + 'experiment3: ' + name + '\n')
#     input_data = projection.fit_transform(data)
#     km, em = test1(input_data, labels, name)
#
#     show_result(km, input_data, labels, km.labels_, name + ' km')
#     show_result(em, input_data, labels, em.predict(input_data), name + ' em')

def make_array(clustering, X, n_components):
    size = len(X)
    new_X = np.zeros((size, n_components + 1))
    new_X[:, 0:n_components] = X
    new_X[:, n_components] = clustering.predict(X)

    return new_X

def test4and5(train_X, train_y, test_X, test_y, start, interval, n_features):
    projections = (PCA, FastICA, GaussianRandomProjection, TruncatedSVD)
    names = ('PCA', 'ICA', 'RP', 'SVD')
    arange = np.arange(start, n_features, interval)

    train0 = np.zeros(len(arange))
    train1 = np.zeros(len(arange))
    train2 = np.zeros(len(arange))
    train3 = np.zeros(len(arange))
    train_scores = (train0, train1, train2, train3)

    test0 = np.zeros(len(arange))
    test1 = np.zeros(len(arange))
    test2 = np.zeros(len(arange))
    test3 = np.zeros(len(arange))
    test_scores = (test0, test1, test2, test3)


    km_test0 = np.zeros(len(arange))
    km_test1 = np.zeros(len(arange))
    km_test2 = np.zeros(len(arange))
    km_test3 = np.zeros(len(arange))
    km_test_scores = (km_test0, km_test1, km_test2, km_test3)

    km_train0 = np.zeros(len(arange))
    km_train1 = np.zeros(len(arange))
    km_train2 = np.zeros(len(arange))
    km_train3 = np.zeros(len(arange))
    km_train_scores = (km_train0, km_train1, km_train2, km_train3)

    em_train0 = np.zeros(len(arange))
    em_train1 = np.zeros(len(arange))
    em_train2 = np.zeros(len(arange))
    em_train3 = np.zeros(len(arange))
    em_train_scores = (em_train0, em_train1, em_train2, em_train3)

    em_test0 = np.zeros(len(arange))
    em_test1 = np.zeros(len(arange))
    em_test2 = np.zeros(len(arange))
    em_test3 = np.zeros(len(arange))
    em_test_scores = (em_test0, em_test1, em_test2, em_test3)

    time0 = np.zeros(len(arange))
    time1 = np.zeros(len(arange))
    time2 = np.zeros(len(arange))
    time3 = np.zeros(len(arange))

    times = (time0, time1, time2, time3)

    index = -1

    for i in arange:
        index = index+1
        print('')
        print('keep components %d:' % i)
        for (pro, train_s, test_s, km_train, km_test, em_train, em_test, t, name) in zip(
                projections, train_scores, test_scores, km_train_scores,
                km_test_scores, em_train_scores, em_test_scores, times, names):

            print('--------%s:' % name)
            projection = pro(n_components=i)
            projection.fit(train_X)
            train_new_X = projection.transform(train_X)
            test_new_X = projection.transform(test_X)

            start_time = time.time()

            NN = MLPClassifier()
            NN.fit(train_new_X, train_y)
            output_y = NN.predict(train_new_X)
            train_s[index] = metrics.accuracy_score(train_y, output_y) * 100
            output_y = NN.predict(test_new_X)
            test_s[index] = metrics.accuracy_score(test_y, output_y) * 100

            end_time = time.time()

            t[index] = end_time - start_time

            print('test 4: train_score=%.2f%%, test_score=%.2f%%, time=%.2f' % (train_s[index], test_s[index], t[index]))

            cl = KMeans(n_clusters=10)
            cl.fit(train_new_X)
            test5_train_X = make_array(cl, train_new_X, i)

            NN = MLPClassifier()
            NN.fit(test5_train_X, train_y)
            output_y = NN.predict(test5_train_X)

            km_train[index] = metrics.accuracy_score(train_y, output_y) * 100

            test5_test_X = make_array(cl, test_new_X, i)
            output_y = NN.predict(test5_test_X)
            km_test[index] = metrics.accuracy_score(test_y, output_y) * 100

            cl = GaussianMixture(n_components=10)
            cl.fit(train_new_X)
            test5_train_X = make_array(cl, train_new_X, i)

            NN = MLPClassifier()
            NN.fit(test5_train_X, train_y)
            output_y = NN.predict(test5_train_X)

            em_train[index] = metrics.accuracy_score(train_y, output_y) * 100

            test5_test_X = make_array(cl, test_new_X, i)
            output_y = NN.predict(test5_test_X)
            em_test[index] = metrics.accuracy_score(test_y, output_y) * 100

            print('test 5: KM: train_score=%.2f%%, test_score=%.2f%%' % (km_train[index], km_test[index]))
            print('test 5: EM: train_score=%.2f%%, test_score=%.2f%%' % (em_train[index], em_test[index]))




    plot_curves(arange, (time0, time1, time2, time3), 'keep components', 'time(s)',
                ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'test4:\ntotal running time for different features reduction algorithms',
                'figures/test4_time.png', flag=True)

    plot_curves(arange, (train0, test0, km_train0, km_test0, em_train0, em_test0),
                'keep_components', 'accuracy(%%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for PCA',
                'figures/test4_PCA.png', flag=True)

    plot_curves(arange, (train1, test1, km_train1, km_test1, em_train1, em_test1),
                'keep_components', 'accuracy(%%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for ICA',
                'figures/test4_ICA.png', flag=True)

    plot_curves(arange, (train2, test2, km_train2, km_test2, em_train2, em_test2),
                'keep_components', 'accuracy(%%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for Random Projection',
                'figures/test4_RP.png', flag=True)

    plot_curves(arange, (train3, test3, km_train3, km_test3, em_train3, em_test3),
                'keep_components', 'accuracy(%%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for SVD',
                'figures/test4_SVD.png', flag=True)








def dataset1_test():
    train_X, train_y = read_data('data/optdigits.tra')
    test_X, test_y = read_data('data/optdigits.tes')

    #test1(train_X, train_y, 2, 20)

    n_features = len(train_X[0])
    scale_X = scale(train_X)
    #test2(scale_X, train_y, n_features, 'test')

    #test3(train_X, train_y, 10, 4, 64)

    test4and5(train_X, train_y, test_X, test_y, 9, 10, 64)


if __name__ == '__main__':

    np.random.seed(15)

    dataset1_test()

    # digits = load_digits()
    # labels = digits.target
    #
    # data = scale(digits.data)


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

    #test5(data, labels, "test5")

    # n_samples, n_features = data.shape
    #
    # n_digits = len(np.unique(digits.target))
    #
    # sample_size = 300
    #
    # km = KMeans(n_clusters=n_digits)
    #
    # test_KMeans(estimator=km, digits=digits, labels=labels, name="test")



