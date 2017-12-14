import numpy
import pandas
import random
def read_data0(data_file):
    data = pandas.read_csv(data_file, sep=';')
    #data = open(data_file, "r")
    train_num = 4000
    size = len(data.values[0])
    print('data size: %d' % len(data.values))
    print(size)

    train_x = data.values[0:train_num, 0:size-1]
    train_y = data.values[0:train_num, size-1]

    test_x = data.values[0:, 0:size-1]
    test_y = data.values[0:, size-1]


    # for y in train_y:
    #     y = int(y)
    #
    # for y in test_y:
    #     y = int(y)

    #print(test_y)

    return train_x, train_y, test_x, test_y

def read_data1(data_file):
    data = pandas.read_table(data_file, sep=',')
    # size = len(data.values[0])
    # print('data size: %d' % len(data.values))
    # print(size)

    features = len(data.values[0])
    tmp = data.values

    random.seed(0)
    random.shuffle(tmp)


    train_x = tmp[0:, 0:features-1]
    train_y = tmp[0:, features-1]

    # for i in range(0, len(train_x)):
    #     for j in range(0, features-1):
    #         if (train_x[i][j] == '?'):
    #             train_x[i][j] = numpy.nan
    #
    #
    # from sklearn.preprocessing import Imputer
    # imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    # imp.fit(train_x)
    # train_x = imp.transform(train_x)
    # print(train_x)
    #

    arr = numpy.zeros(shape=len(train_y), dtype='int')

    for i in range(0, len(train_y)):
        # print(train_y[i])
        if (train_y[i] == 'g'):
            arr[i] = 0
        else:
            arr[i] = 1
    print(arr)

    # arr = train_y

    # test_x = train_x
    # test_y = arr


    return train_x, arr

def read_data(data_file):
    data = pandas.read_table(data_file, sep=',')
    # size = len(data.values[0])
    # print('data size: %d' % len(data.values))
    # print(size)

    features = len(data.values[0])


    train_x = data.values[0:, 0:features-1]
    train_y = data.values[0:, features-1]

    return train_x, train_y

# data_file = 'letter-recognition.data'
#
# read_data0(data_file)

# train_x, train_y, test_x, test_y = read_data(data_file)
#
# print(train_x[0])
# print(test_x[0])
#
# print(train_y)
# print(test_y)