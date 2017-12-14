import matplotlib.pyplot as plt
import numpy as np
import input

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Model_Validation:
    def __init__(self, train_x, train_y, model, features):
        self.train_x = train_x
        self.train_y = train_y

        self.estimator = model

        self.features = features
        self.has_test = False

    def set_estimator(self, estimator):
        self.estimator = estimator

    def set_test_data(self, test_x, test_y):
        self.has_test = True
        self.test_x = test_x
        self.test_y = test_y

    def plot_learning_curve(self, title, ylim= None, cv = None, n_jobs = 1, train_sizes=np.linspace(.1, 1.0, 10)):
        print('training'+title)
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training samples")
        plt.ylabel("Error")
        train_sizes, train_scores, test_scores = learning_curve(
            self.estimator, self.train_x, self.train_y, cv = cv, n_jobs = n_jobs, train_sizes=train_sizes)

        train_scores = 1 - train_scores

        test_scores = 1 - test_scores

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        print(train_sizes)
        print(train_scores_mean)

        plt.grid()

        # plt.fill_between(
        #     train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color="r")
        #
        # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
        #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, color="r",
                 label="Training Error")
        plt.plot(train_sizes, test_scores_mean, color="g",
                 label="Cross-validation Error")

        plt.legend(loc="best")

        # print(test_scores_mean)
        return plt

    def plot_validation_curve_data(self, title, x_label, train_scores, test_scores, new_test_scores, param_range, minx=-1, maxx=-1, plot = True):

        train_scores = 1 - train_scores
        test_scores = 1 - test_scores
        new_test_scores = 1 - new_test_scores

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Error")

        if (minx != -1 & maxx != -1):
            plt.xlim(minx, maxx)

        lw = 2



        # plt.fill_between(param_range, train_scores_mean - train_scores_std,
        #                  train_scores_mean + train_scores_std, alpha=0.5,
        #                  color="r", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="g", lw=lw)

        if (plot == False):
            plt.semilogx(param_range, train_scores_mean, label="Training Error",
                        color="r", lw=lw)
            plt.semilogx(param_range, test_scores_mean, label="Cross-validation Error",
                        color="g", lw=lw)
            if (self.has_test):
                plt.semilogx(param_range, new_test_scores, label="Test Data Error",
                        color="b", lw=lw)
        else:
            plt.plot(param_range, train_scores_mean, label="Training Error",
                        color="r")
            plt.plot(param_range, test_scores_mean, label="Cross-validation Error",
                        color="g")
            if (self.has_test):
                plt.plot(param_range, new_test_scores, label="Test Data Error",
                        color="b")

        plt.legend(loc="best")


    def get_validation_curve_data(self, param_name, cv = None, n_jobs = 1, param_range = np.linspace(1, 10, 10)):
        train_scores, test_scores = validation_curve(
            self.estimator, self.train_x, self.train_y, param_name=param_name, param_range=param_range,
            cv=cv, scoring="accuracy", n_jobs=n_jobs)
        return train_scores, test_scores

    def get_scores(self, estimator):
        import time

        start_time = time.time()
        estimator.fit(self.train_x, self.train_y)
        predict = estimator.predict(self.test_x)
        tot_time = time.time() - start_time

        accuracy = metrics.accuracy_score(self.test_y, predict)

        return accuracy, tot_time

    def normalize(self):
        scaler = StandardScaler()
        scaler.fit(self.train_x)
        self.train_x = scaler.transform(self.train_x)
        if self.has_test:
            self.test_x = scaler.transform(self.test_x)

def get_SVM_learning_curve(data, dataset, gamma, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()

    title = dataset + ":\nLearning Curve(SVM)"
    print("training " + title)
    plt.title(title)

    plt.xlabel("Training samples")
    plt.ylabel("Error")

    train_sizes, train_scores, test_scores = learning_curve(
        SVC(gamma=gamma), data.train_x, data.train_y, cv=6, n_jobs=-1, train_sizes=train_sizes)

    train_scores = 1 - train_scores
    test_scores = 1 - test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()

    plt.plot(train_sizes, train_scores_mean, color="r",
             label=("Training Error(gamma=%.4f)" % gamma))
    plt.plot(train_sizes, test_scores_mean, color="g",
             label=("Cross-validation Error(gamma=%.4f)" % gamma))

    train_sizes, train_scores, test_scores = learning_curve(
        SVC(), data.train_x, data.train_y, cv=6, n_jobs=-1, train_sizes=train_sizes)

    train_scores = 1 - train_scores
    test_scores = 1 - test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, color="c",
             label="Training Error(default gamma)")
    plt.plot(train_sizes, test_scores_mean, color="b",
             label="Cross-validation Error(default gamma)")

    plt.legend()

    plt.savefig(dataset + " Learning Curve(SVM).png")
    # print(test_scores_mean)
    return plt

def get_NN_VC(data, dataset):
    step = np.arange(1, 50, 2)
    print("training NN " + dataset)
    # print(step)
    new_test_scores = np.zeros(len(step))
    index = 0
    # input_range = np.zeros(shape=(len(step), 2), dtype='int32')
    input_range = []
    data.normalize()

    for i in step:
        # arr = np.zeros(shape = (i), dtype='int32')
        # for j in range(0, i):
        #     arr[j] = 2
        # print(arr)
        arr = (i, )
        input_range.append(arr)

        if data.has_test:
            new_test_scores[index], t = data.get_scores(MLPClassifier(hidden_layer_sizes=arr))

        # input_range.append((i,))

        index = index+1

    print(input_range)

    # print(' ')
    # print(new_test_scores)

    data.set_estimator(MLPClassifier())
    train_scores, test_scores = data.get_validation_curve_data(param_name="hidden_layer_sizes", cv = 6, n_jobs = -1, param_range=input_range)
    data.plot_validation_curve_data(title=dataset+"\nModel Complexity Curve(Neural Network with 1 hidden layer)", x_label="Hidden layer Size",
                                    train_scores=train_scores, test_scores=test_scores, new_test_scores=new_test_scores, param_range=step,
                                    plot = True)

    filename = dataset + " Model Complexity Curve(NeuralNetwork).png"

    plt.savefig(filename)

    return plt

def get_Boost_VC(data, dataset):
    print("training Boost " + dataset)
    step = np.arange(1, 50, 2)
    # print(step)
    new_test_scores = np.zeros(len(step))
    index = 0

    for i in step:
        if data.has_test:
            new_test_scores[index], t = data.get_scores(AdaBoostClassifier(n_estimators=i))
        index = index+1

    # print(' ')
    # print(new_test_scores)

    data.set_estimator(AdaBoostClassifier())
    train_scores, test_scores = data.get_validation_curve_data(param_name="n_estimators", cv = 6, n_jobs = -1, param_range=step)
    data.plot_validation_curve_data(title=dataset + "\nModel Complexity Curve(AdaBoost)", x_label="Number of estimators",
                                    train_scores=train_scores, test_scores=test_scores, new_test_scores=new_test_scores, param_range=step,
                                    plot = True)

    filename = dataset + " Model Complexity Curve(AdaBoost).png"

    plt.savefig(filename)

    return plt

def get_KNN_VC(data, dataset):
    print("training KNN " + dataset)
    step = np.arange(9, 0, -1)
    for i in range(0, len(step)):
        step[i] = (2 << (step[i]))/4
    # print(step)
    new_test_scores = np.zeros(len(step))
    index = 0

    for i in step:
        if data.has_test:
            new_test_scores[index], t = data.get_scores(KNeighborsClassifier(n_neighbors=i))
        index = index+1

    # print(' ')
    # print(new_test_scores)

    param_range = 1.0 / step
    data.set_estimator(KNeighborsClassifier())
    train_scores, test_scores = data.get_validation_curve_data(param_name="n_neighbors", cv = 6, n_jobs = -1, param_range=step)
    data.plot_validation_curve_data(title=dataset + "\nModel Complexity Curve(KNN)", x_label="1/k",
                                    train_scores=train_scores, test_scores=test_scores, new_test_scores=new_test_scores, param_range=param_range,
                                    plot = False)
    filename = dataset + " Model Complexity Curve(KNN).png"

    plt.savefig(filename)

    return plt

def get_DT_VC(data, dataset):
    print("training DT " + dataset)
    step = np.arange(1, 40, 2)
    # print(step)
    new_test_scores = np.zeros(len(step))
    index = 0

    for i in step:
        if data.has_test:
            new_test_scores[index], t = data.get_scores(DecisionTreeClassifier(max_depth=i))
        index = index+1

    # print(' ')
    # print(new_test_scores)

    data.set_estimator(DecisionTreeClassifier(criterion='gini', splitter='random'))
    train_scores, test_scores = data.get_validation_curve_data(param_name="max_depth", cv = 6, n_jobs = -1, param_range=step)
    data.plot_validation_curve_data(dataset + "\nModel Complexity Curve(Decision Tree)", "Max depth",
                                    train_scores=train_scores, test_scores=test_scores, new_test_scores=new_test_scores, param_range=step,
                                    plot = True)

    filename = dataset + " Model Complexity Curve(Decision Tree).png"

    plt.savefig(filename)

    return plt

def get_SVM_VC(data, dataset):
    print("training SVM " + dataset)
    param_range = np.logspace(-6, 1.2, 20)

    # print(param_range)
    new_test_scores = np.zeros(len(param_range))
    index = 0

    for i in param_range:
        if data.has_test:
            new_test_scores[index], t = data.get_scores(SVC(gamma=i))
        index = index+1

    # print(' ')
    # print(new_test_scores)

    data.set_estimator(SVC())
    train_scores, test_scores = data.get_validation_curve_data(param_name="gamma", cv = 6, n_jobs = -1, param_range=param_range)
    data.plot_validation_curve_data(dataset + "\nModel Complexity Curve(SVM)", "Gamma of kernel",
                                    train_scores=train_scores, test_scores=test_scores, new_test_scores=new_test_scores, param_range=param_range,
                                    plot = False)

    filename = dataset + " Model Complexity Curve(SVM).png"

    plt.savefig(filename)

    return plt

# def autolabel(ax, rects, pos):
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., height,
#                 '%.1f' % height,
#                 ha=pos, va='bottom')

def get_overview(data, dataset):
    print("training overview " + dataset)
    accuracy = np.zeros(5)
    times = np.zeros(5)

    # accuracy = (0.1, 0.2, 0.3, 0.4, 0.5)
    # times = (20, 30, 80, 40, 4)

    accuracy[0], times[0] = data.get_scores(DecisionTreeClassifier())
    accuracy[1], times[1] = data.get_scores(KNeighborsClassifier())
    accuracy[2], times[2] = data.get_scores(SVC())
    accuracy[3], times[3] = data.get_scores(AdaBoostClassifier())
    accuracy[4], times[4] = data.get_scores(MLPClassifier())

    accuracy = accuracy * 100


    name = ('Decision Tree', 'KNN', 'SVM', 'AdaBoost', 'Neural Network')

    plt.figure()
    title = dataset + ": Overview of accuracy and performance"
    plt.title(title)

    ax1 = plt.subplot()
    ax2 = ax1.twinx()

    opacity = 0.6
    ax1.set_ylabel('Accuracy', color = (0, 0, 1, opacity))
    ax2.set_ylabel('Running time(s)', color = (1, 0, 0, opacity))
    ax1.plot(color = (0, 0, 1, opacity))
    ax2.plot(color = (1, 0, 0, opacity))
    index = np.arange(5)+1
    bar_width = 0.25


    bar1 = ax1.bar(index-0.06, accuracy, bar_width, alpha = opacity, color = 'b', label = 'Accuracy')
    bar2 = ax2.bar(index + bar_width+0.06, times, bar_width, alpha = opacity, color = 'r', label = 'Running time')
    # autolabel(ax1, bar1, 'left')
    # autolabel(ax2, bar2, 'right')

    for rect in bar1:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f%%' % height,
                ha='center', va='bottom')

    for rect in bar2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                '%.2fs' % height,
                ha='center', va='bottom')

    plt.xticks(index + bar_width/2, name)

    # plt.legend((bar1, bar2), ('Accuracy', 'Running time'))
    plt.tight_layout()

    plt.savefig(dataset+" overview.png")

    return plt

def get_SVM_compare():
    print("training SVM comparison " + dataset)
    group = 4

    accuracy1 = np.zeros(4)
    times1 = np.zeros(4)

    train_x, train_y = input.read_data('optdigits.tra')
    test_x, test_y = input.read_data('optdigits.tes')


    data = Model_Validation(train_x, train_y, DecisionTreeClassifier(), len(train_x[0]))
    data.set_test_data(test_x, test_y)

    accuracy1[0], times1[0] = data.get_scores(SVC())
    print('rbf')
    accuracy1[1], times1[1] = data.get_scores(SVC(kernel='linear'))
    print('linear')
    accuracy1[2], times1[2] = data.get_scores(SVC(kernel='poly'))
    print('poly')
    accuracy1[3], times1[3] = data.get_scores(SVC(kernel='sigmoid'))
    print('sigmoid')

    accuracy1 = accuracy1 * 100

    name = ('rbf', 'linear', 'poly', 'sigmoid')

    plt.figure()
    title = "dataset 1: \nSVM kernels comparison"
    plt.title(title)

    ax1 = plt.subplot()
    ax2 = ax1.twinx()

    opacity = 0.6
    ax1.set_ylabel('Accuracy', color = (0, 0, 1, opacity))
    ax2.set_ylabel('Running time(s)', color = (1, 0, 0, opacity))
    ax1.plot(color = (0, 0, 1, opacity))
    ax2.plot(color = (1, 0, 0, opacity))
    index = np.arange(4)+1
    bar_width = 0.25


    bar11 = ax1.bar(index, accuracy1, bar_width, alpha = opacity, color = 'b')

    bar21 = ax2.bar(index + bar_width, times1, bar_width, alpha = opacity, color = 'r')
    # autolabel(ax1, bar1, 'left')
    # autolabel(ax2, bar2, 'right')

    for rect in bar11:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f%%' % height,
                ha='center', va='bottom')

    for rect in bar21:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                '%.2fs' % height,
                ha='center', va='bottom')


    plt.xticks(index + bar_width/2, name)

    plt.legend((bar11, bar21), ('Accuracy', 'Running time'))
    plt.tight_layout()

    plt.savefig("SVM kernels comparison.png")

    return plt

def get_ROC_curve():
    print("training ROC " + dataset)
    file1 = 'magic04.data'
    train_x, train_y = input.read_data1(file1)


    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)
    colors = ('r', 'g', 'b', 'c', 'darkorange')
    estimators = (DecisionTreeClassifier(), KNeighborsClassifier(), SVC(gamma=0.001, probability=True), AdaBoostClassifier(), MLPClassifier())
    names = ('Decision tree', 'KNN', 'SVM', 'AdaBoost', 'Neural network')
    plt.figure()

    lw = 2
    for estimator, color, name in zip(estimators, colors, names):
        classifier = estimator.fit(train_x, train_y)
        y_score = classifier.predict_proba(test_x)
        # y_score = classifier.fit(train_x, train_y).decision_function(test_x)
        fpr, tpr, _ = roc_curve(test_y, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, label=(name + (' (area = %0.2f)' % roc_auc)) )


    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('dataset 2:\nROC curves')
    plt.legend(loc="lower right")
    plt.savefig('datase 2 ROC curves.png')

if __name__ == '__main__':


    # dataset 1 start

    train_x, train_y = input.read_data('optdigits.tra')
    test_x, test_y = input.read_data('optdigits.tes')

    # print(len(train_x))
    data = Model_Validation(train_x, train_y, DecisionTreeClassifier(), len(train_x[0]))
    data.set_test_data(test_x, test_y)

    dataset = "Dataset 1"

    print(dataset + " start")

    #get_overview(data, dataset)

    #get_SVM_learning_curve(data, dataset, 0.0025)

    data.set_estimator(AdaBoostClassifier())
    data.plot_learning_curve(title=dataset+ ":\nLearning Curve(AdaBoost)", cv=6, n_jobs=-1)
    plt.savefig(dataset + " Learning Curve(AdaBoost).png")

    data.set_estimator(KNeighborsClassifier())
    data.plot_learning_curve(title = dataset+ ":\nLearning Curve(KNN)", cv = 6, n_jobs=-1)
    plt.savefig(dataset + " Learning Curve(KNN).png")
    plt.show()

    #
    # data.set_estimator(DecisionTreeClassifier())
    # data.plot_learning_curve(title = dataset+ ":\nLearning Curve(Decision Tree)", cv = 6, n_jobs=-1)
    # plt.savefig(dataset + " Learning Curve(Decision Tree).png")
    #
    # get_SVM_VC(data, dataset)
    #
    # get_DT_VC(data, dataset)
    #
    # get_Boost_VC(data, dataset)
    #
    # get_KNN_VC(data, dataset)
    #
    # get_NN_VC(data, dataset)
    #
    # data.set_estimator(MLPClassifier())
    # data.plot_learning_curve(title=dataset + ":\nLearning Curve(NeuralNetwork)", cv=6, n_jobs=-1)
    # plt.savefig(dataset + " Learning Curve(NeuralNetwork).png")

    # dataset 2 start

    # file1 = 'magic04.data'
    #
    # dataset = "Dataset 2"
    # print(dataset + " start")
    #
    #
    # train_x, train_y = input.read_data1(file1)
    #
    # data = Model_Validation(train_x, train_y, DecisionTreeClassifier(), len(train_x[0]))
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)
    # data.set_test_data(test_x, test_y)
    #
    # get_overview(data, dataset)
    #
    # train_x, train_y = input.read_data1(file1)
    #
    # data = Model_Validation(train_x, train_y, DecisionTreeClassifier(), len(train_x[0]))
    #
    # get_SVM_learning_curve(data, dataset, 0.001, train_sizes=np.linspace(.8, 1.0, 6))
    #
    # data.set_estimator(AdaBoostClassifier())
    # data.plot_learning_curve(title=dataset+ ":\nLearning Curve(AdaBoost)", cv=6, n_jobs=-1)
    # plt.savefig(dataset + " Learning Curve(AdaBoost).png")
    #
    # data.set_estimator(KNeighborsClassifier())
    # data.plot_learning_curve(title = dataset+ ":\nLearning Curve(KNN)", cv = 6, n_jobs=-1)
    # plt.savefig(dataset + " Learning Curve(KNN).png")
    #
    # data.set_estimator(DecisionTreeClassifier())
    # data.plot_learning_curve(title = dataset+ ":\nLearning Curve(Decision Tree)", cv = 6, n_jobs=-1)
    # plt.savefig(dataset + " Learning Curve(Decision Tree).png")
    #
    #
    #
    # get_SVM_VC(data, dataset)
    #
    # get_DT_VC(data, dataset)
    #
    # get_Boost_VC(data, dataset)
    #
    # get_KNN_VC(data, dataset)
    #
    # get_NN_VC(data, dataset)
    #
    # data.set_estimator(MLPClassifier())
    # data.plot_learning_curve(title=dataset + ":\nLearning Curve(NeuralNetwork)", cv=6, n_jobs=-1)
    # plt.savefig(dataset + " Learning Curve(NeuralNetwork).png")
    #
    # get_ROC_curve()
    # get_SVM_compare()
