import pandas
import numpy as np
import matplotlib.pyplot as plt
import time


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

from sklearn import metrics

def read_dataset1(path):
    data = pandas.read_csv(path, sep=',')
    instances = len(data.values)
    features = len(data.values[0])-1
    X = data.values[:, 0:features]
    y = data.values[:, features]

    return X, y

def read_dataset2(path):
    data = pandas.read_csv(path)
    instances = len(data.values)
    features = len(data.values[0])-1
    X = data.values[:, 0:features]
    tmp = data.values[:, features]
    uni = np.unique(tmp)
    #print(uni)
    y = np.zeros(instances)
    for i in range(0, instances):
        y[i] = 3
        for j in range(0, 4):
            if tmp[i] == uni[j]:
                y[i] = j;

    #print(y)
    #print(tmp)
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

def show_result(estimator, data, labels, Y, name):
    print('\n' + name + ': ')

    print('ARI: {}'.format(metrics.adjusted_rand_score(labels, Y)))
    print('AMI: {}'.format(metrics.adjusted_mutual_info_score(labels, Y)))

    # How good are the clusters?
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, Y)))
    print('Completeness: {}'.format(metrics.completeness_score(labels, Y)))
    print('Silhouette: {}'.format(metrics.silhouette_score(data, Y)))

def plot_curves(x, Y, xlabel, ylabel, curve_labels, name, save_name, flag=False, dotline=0, line_label='', show100=False):
    print(x)

    colors = plt.cm.rainbow(np.linspace(1, 0, len(Y)))

    plt.figure()
    plt.title(name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (flag):
        plt.xticks(x, x)

    if (dotline!=0):
        tmp = np.zeros(len(x))
        tmp[:] = dotline
        print(tmp)
        plt.plot(x, tmp, color='black', label=line_label, lw=0.7, ls='dashed')
        tmp100 = np.zeros(len(x))
        tmp100[:] = 100
        if (show100):
            plt.plot(x, tmp100, color='black', label='train accuracy of raw data', lw=0.7, ls='dotted')

    for (y, label, c) in zip(Y, curve_labels, colors):
        plt.plot(x, y, color=c, label=label, lw=2.0)

    plt.legend(loc='best')
    plt.savefig(save_name)

    return plt


def test1(X, y, min_n, max_n, dataset):

    score0 = np.zeros(max_n-min_n+1)
    score1 = np.zeros(max_n-min_n+1)

    for i in range(min_n, max_n+1):
        index = i - min_n

        km = KMeans(n_clusters=i, random_state=10)
        output_y = km.fit_predict(X)
        score0[index] = metrics.silhouette_score(X, output_y)
        print('km: n_cluster = %d: silhouette_score = %f' % (i, score0[index]))

        em = GaussianMixture(n_components=i, random_state=10)
        em.fit(X)
        output_y = em.predict(X)
        score1[index] = metrics.silhouette_score(X, output_y)
        print('em: n_cluster = %d: silhouette_score = %f' % (i, score1[index]))

    plot_curves(np.arange(min_n, max_n+1), (score0, score1), 'k', 'Silhouette Score', ('k-means', 'EM'),
                'test1 for ' + dataset + ':\nsilhouette score curves for different clusters', 'figures/test1_' + dataset + 'sc.png', flag=True)

    for i in range(min_n, max_n+1):
        index = i - min_n

        km = KMeans(n_clusters=i, random_state=10)
        output_y = km.fit_predict(X)
        score0[index] = metrics.adjusted_rand_score(y, output_y)
        print('km: n_cluster = %d: ARI = %f' % (i, score0[index]))

        em = GaussianMixture(n_components=i, random_state=10)
        em.fit(X)
        output_y = em.predict(X)
        score1[index] = metrics.adjusted_rand_score(y, output_y)
        print('em: n_cluster = %d:ARI = %f' % (i, score1[index]))

    plot_curves(np.arange(min_n, max_n+1), (score0, score1), 'k', 'ARI Score', ('k-means', 'EM'),
                'test1 for ' + dataset + ':\nARI scores curves for different clusters', 'figures/test1_' + dataset + 'ari.png', flag=True)


def test2(X, y, N, name, dataset):
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
                'test2 for '+ dataset + ':\nReconstruction Error for components analysis', 'figures/test2_'+ dataset + '.png')

def test3(X, y, n_labels, interval, n_features, dataset):

    projections = (PCA, FastICA, GaussianRandomProjection, TruncatedSVD)

    arange = np.arange(1, n_features, interval)

    score0 = np.zeros(len(arange))
    score1 = np.zeros(len(arange))
    score2 = np.zeros(len(arange))
    score3 = np.zeros(len(arange))

    scores = (score0, score1, score2, score3)

    index = -1

    print(arange)
    km = KMeans(n_clusters=n_labels, random_state=10)
    output_y = km.fit_predict(X)
    sc = metrics.adjusted_rand_score(y, output_y)
    print(sc)

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
                'test3 for ' + dataset + ':\nARI scores for different projections in k-means', 'figures/test3_'+ dataset + '_km.png',flag=True, dotline=sc, line_label='raw data')

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

    em = GaussianMixture(n_components=n_labels, random_state=10)
    em.fit(X)
    output_y = em.predict(X)
    sc = metrics.adjusted_rand_score(y, output_y)
    print(sc)
    plot_curves(arange, (score0, score1, score2, score3), 'keep components', 'ARI score', ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'test3 for '+ dataset + ':\nARI scores for features reduction algorithms in EM', 'figures/test3_' + dataset + '_em.png',flag=True, dotline=sc, line_label='raw data')


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

def test4and5(train_X, train_y, test_X, test_y, start,n_features, interval):
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



    start_time = time.time()
    NN = MLPClassifier()
    NN.fit(train_X, train_y)
    output_y = NN.predict(train_X)
    acc = metrics.accuracy_score(train_y, output_y) * 100
    print('training accuracy = %.2f%%' % acc)
    output_y = NN.predict(test_X)
    acc = metrics.accuracy_score(test_y, output_y) * 100
    tot_time = time.time() - start_time
    print('testing accuracy = %.2f%%' % acc)
    print('total time = %fs' % tot_time)

    plot_curves(arange, (time0, time1, time2, time3), 'keep components', 'time(s)',
                ('PCA', 'ICA', 'Random Projection', 'SVD'),
                'test4:\ntotal running time for different features reduction algorithms',
                'figures/test4_time.png', flag=True, dotline=tot_time, line_label='raw data')

    plot_curves(arange, (train0, test0, km_train0, km_test0, em_train0, em_test0),
                'keep_components', 'accuracy(%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for PCA',
                'figures/test4_PCA.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)

    plot_curves(arange, (train1, test1, km_train1, km_test1, em_train1, em_test1),
                'keep_components', 'accuracy(%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for ICA',
                'figures/test4_ICA.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)

    plot_curves(arange, (train2, test2, km_train2, km_test2, em_train2, em_test2),
                'keep_components', 'accuracy(%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for Random Projection',
                'figures/test4_RP.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)

    plot_curves(arange, (train3, test3, km_train3, km_test3, em_train3, em_test3),
                'keep_components', 'accuracy(%)',
                ('test4: train accuracy', 'test4: test accuracy',
                 'test5: train accuracy(k-means)', 'test5: test accuracy(k-means)',
                 'test5: train accuracy(EM)', 'test5: test accuracy(EM)'),
                'test4 and test5:\ntraining and testing accuracy for SVD',
                'figures/test4_SVD.png', flag=True, dotline=acc, line_label='test accuracy of raw data', show100=True)



def plot_projection_graph(data, n_clusters, dataset, savename):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the '+ dataset + ' (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('figures/' + savename)




def dataset1_test():
    train_X, train_y = read_dataset1('data/optdigits.tra')
    test_X, test_y = read_dataset1('data/optdigits.tes')

    test1(train_X, train_y, 2, 20, 'dataset1')

    n_features = len(train_X[0])
    scale_X = scale(train_X)
    test2(scale_X, train_y, n_features, 'test', 'dataset1')

    test3(train_X, train_y, 10, 4, 64, 'dataset1')

    test4and5(train_X, train_y, test_X, test_y, 1, 64, 4)
    plot_projection_graph(train_X, 10, 'dataset1', 'dataset1_graph.png')

def dataset2_test():
    X, y = read_dataset2('data/data2.csv')

    test1(X, y, 2, 10, 'dataset2')

    n_features = len(X[0])
    scale_X = scale(X)

    test2(scale_X, y, n_features, 'test', 'dataset2')

    test3(X, y, 4, 1, 5, 'datset2')

    plot_projection_graph(X, 4, 'dataset2', 'dataset2_graph.png')

if __name__ == '__main__':

    np.random.seed(15)

    dataset1_test()

    dataset2_test()

    plt.show()



