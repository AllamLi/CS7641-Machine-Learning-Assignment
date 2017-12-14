import matplotlib.pyplot as plt
import numpy as np
import pandas

plot_colors = ('r', 'g', 'b', 'c', 'darkorange', 'black')

def read_data3(path_name):
    data = pandas.read_csv(path_name, sep=",");
    size = len(data.values)
    x0 = data.values[0:size, 0]
    x1 = data.values[0:size, 1]
    x2 = data.values[0:size, 2]
    return x0, x1, x2

def read_data5(path_name):
    data = pandas.read_csv(path_name, sep=",");
    size = len(data.values)
    x0 = data.values[0:size, 0]
    x1 = data.values[0:size, 1]
    x2 = data.values[0:size, 2]
    x3 = data.values[0:size, 3]
    x4 = data.values[0:size, 4]
    return x0, x1, x2, x3, x4

def get_compare_figure(title, label, accuracy, times):

    plt.figure()

    plt.title(title)

    ax1 = plt.subplot()
    ax2 = ax1.twinx()

    opacity = 0.6
    ax1.set_ylabel(label, color=(0, 0, 1, opacity))
    ax2.set_ylabel('Running time(s)',color=(1, 0, 0, opacity))
    ax1.plot(color=(0, 0, 1, opacity))
    ax2.plot(color=(1, 0, 0, opacity))
    index = np.arange(4)+1
    bar_width = 0.25

    bar11 = ax1.bar(index, accuracy, bar_width, alpha = opacity, color = 'b')

    bar21 = ax2.bar(index + bar_width, times, bar_width, alpha = opacity, color = 'r')

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

    name = ('BackProp', 'RHC', 'SA', 'GA')

    plt.xticks(index + bar_width/2, name)

    plt.legend((bar11, bar21), (label, 'Running time'), loc="best")
    plt.tight_layout()

    plt.savefig(title + ".png")

    return plt

def get_comparison(title, label1, label2, name, accuracy, times):

    plt.figure()

    plt.title(title)

    ax1 = plt.subplot()
    ax2 = ax1.twinx()

    opacity = 0.6
    ax1.set_ylabel(label1, color=(0, 0, 1, opacity))
    ax2.set_ylabel(label2,color=(1, 0, 0, opacity))
    ax1.plot(color=(0, 0, 1, opacity))
    ax2.plot(color=(1, 0, 0, opacity))
    index = np.arange(6)+1
    bar_width = 0.3

    bar11 = ax1.bar(index - 0.1, accuracy, bar_width, alpha = opacity, color = 'b')

    bar21 = ax2.bar(index + bar_width + 0.1, times, bar_width, alpha = opacity, color = 'r')

    for rect in bar11:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f%%' % height,
                ha='center', va='bottom')

    for rect in bar21:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f%%' % height,
                ha='center', va='bottom')


    plt.xticks(index + bar_width/2, name)

    #plt.legend((bar11, bar21), (label1, label2), loc="best")
    plt.tight_layout()

    plt.savefig("png/" + title + ".png")

    return plt

class Figures:
    def __init__(self, title, x_label, y_label):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.num = 0;

    def start(self):
        plt.figure()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

    def finish(self):
        plt.legend(loc="best")
        plt.savefig("png/" + self.title + ".png")
        return plt

    def plot_curve(self, label, param_range, values, plot=True):
        if (plot==True):
            plt.plot(param_range, values, label=label, color=plot_colors[self.num])
        else:
            plt.semilogx(param_range, values, label=label, color=plot_colors[self.num])

        self.num += 1



if __name__ == '__main__':
    x, x_RHC, t_RHC = read_data3("csv/TSP_RHC.csv")
    x, x_SA, t_SA = read_data3("csv/TSP_SA.csv")

    x0, x_GA, t_GA = read_data3("csv/TSP_GA.csv")
    x1, x_MIMIC, t_MIMIC = read_data3("csv/TSP_MIMIC.csv")

    f = Figures("TSP Evaluation Curves", "Iterations", "Evaluation Values")
    f.start()
    f.plot_curve("RHC", x, x_RHC)
    f.plot_curve("SA", x, x_SA)
    f.plot_curve("GA", x0, x_GA)
    f.plot_curve("MIMIC", x1, x_MIMIC)
    f.finish()

#--------------------------------------------------Flip Flop

    x, x_RHC, t_RHC = read_data3("csv/FF_RHC.csv")
    x, x_SA, t_SA = read_data3("csv/FF_SA.csv")

    x0, x_GA, t_GA = read_data3("csv/FF_GA.csv")
    x1, x_MIMIC, t_MIMIC = read_data3("csv/FF_MIMIC.csv")

    f = Figures("Flip Flop Evaluation Curves", "Iterations", "Evaluation Values")
    f.start()
    f.plot_curve("RHC", x, x_RHC)
    f.plot_curve("SA", x, x_SA)
    f.plot_curve("GA", x0, x_GA)
    f.plot_curve("MIMIC", x1, x_MIMIC)
    f.finish()

#--------------------------------------------------Knapsack

    x, x_RHC, t_RHC = read_data3("csv/KNS_RHC.csv")
    x, x_SA, t_SA = read_data3("csv/KNS_SA.csv")

    x0, x_GA, t_GA = read_data3("csv/KNS_GA.csv")
    x1, x_MIMIC, t_MIMIC = read_data3("csv/KNS_MIMIC.csv")

    f = Figures("Knapsack Problem Evaluation Curves", "Iterations", "Evaluation Values")
    f.start()
    f.plot_curve("RHC", x, x_RHC)
    f.plot_curve("SA", x, x_SA)
    f.plot_curve("GA", x0, x_GA)
    f.plot_curve("MIMIC", x1, x_MIMIC)
    f.finish()

#--------------------------------------------------finish opt

    x0, RHC_error, RHC_train, RHC_test, RHC_time = read_data5("csv/RHCError.csv")
    x1, SA_error, SA_train, SA_test, SA_time = read_data5("csv/SAError.csv")
    x2, GA_error, GA_train, GA_test, GA_time = read_data5("csv/GAError.csv")

    f = Figures("Learning Curves(Accuracy)", "Iterations", "Accuracy")
    f.start()
    f.plot_curve("RHC(train)", x0, RHC_train)
    f.plot_curve("RHC(test)", x0, RHC_test)
    f.plot_curve("SA(train)", x1, SA_train)
    f.plot_curve("SA(test)", x1, SA_test)
    f.plot_curve("GA(train)", x2, GA_train)
    f.plot_curve("GA(test)", x2, GA_test)
    f.finish()

    f = Figures("Learning Square Error Curves", "Iterations", "Square Error")
    f.start()
    f.plot_curve("RHC", x0, RHC_error)
    f.plot_curve("SA", x1, SA_error)
    f.plot_curve("GA", x2, GA_error)
    f.finish()

    x0, error, train_acc, test_acc, time = read_data5("csv/GA_test0.csv")
    size = len(x0) - 1
    a0 = train_acc[size]
    b0 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/GA_test1.csv")
    a1 = train_acc[size]
    b1 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/GA_test2.csv")
    a2 = train_acc[size]
    b2 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/GA_test3.csv")
    a3 = train_acc[size]
    b3 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/GA_test4.csv")
    a4 = train_acc[size]
    b4 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/GA_test5.csv")
    a5 = train_acc[size]
    b5 = test_acc[size]

    get_comparison("GA Training Accuracy Comparison(iterations = 1000)", "Training Accuracy", "Testing Accuracy",
                   ("P=100\nM=50", "P=150\nM=75", "P=200\nM=100","P=250\nM=125", "P=300\nM=150", "P=350\nM=175"),
                   (a0, a1, a2, a3, a4, a5), (b0, b1, b2, b3, b4, b5))

    x0, error, train_acc, test_acc, time = read_data5("csv/SA0_test0.csv")
    size = len(x0) - 1
    a0 = train_acc[size]
    b0 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/SA0_test1.csv")
    a1 = train_acc[size]
    b1 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/SA0_test2.csv")
    a2 = train_acc[size]
    b2 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/SA0_test3.csv")
    a3 = train_acc[size]
    b3 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/SA0_test4.csv")
    a4 = train_acc[size]
    b4 = test_acc[size]

    x0, error, train_acc, test_acc, time = read_data5("csv/SA0_test5.csv")
    a5 = train_acc[size]
    b5 = test_acc[size]

    get_comparison("SA Training Accuracy Comparison(iterations = 6000 cooling = 0.95)", "Training Accuracy", "Testing Accuracy",
                   ("T=0.01", "T=1", "T=100", "T=1E4", "T=1E6", "T=1E8"),
                   (a0, a1, a2, a3, a4, a5), (b0, b1, b2, b3, b4, b5))

    f = Figures("SA training Accuracy(T = 100)", "Iterations", "Training Accuracy")
    f.start()
    x0, error, train_acc, test_acc, time = read_data5("csv/SA1_test0.csv")
    f.plot_curve("Cooling=0.95", x0, train_acc)

    x0, error, train_acc, test_acc, time = read_data5("csv/SA1_test1.csv")
    f.plot_curve("Cooling=0.80", x0, train_acc)

    x0, error, train_acc, test_acc, time= read_data5("csv/SA1_test2.csv")
    f.plot_curve("Cooling=0.65", x0, train_acc)

    f.finish()

    f = Figures("Flip Flop Curves of Different T", "Iterations", "Evaluation Values")
    f.start()
    x0, train_acc, time = read_data3("csv/FF_SA0.csv")
    f.plot_curve("T = 0.01", x0, train_acc)

    x0, train_acc, time = read_data3("csv/FF_SA1.csv")
    f.plot_curve("T = 1.0", x0, train_acc)

    x0, train_acc, time = read_data3("csv/FF_SA2.csv")
    f.plot_curve("T = 100", x0, train_acc)

    f.finish()

    plt.show()
