import tensorflow as tf
import numpy as np
import shutil
import os
import math
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def safe_sqrt(x, lbound=1e-10):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def OBJ_func(w_I, w_C, w_A, I=8, C=16, A=24):
    OBJ = (np.sum(w_I[:I]) - np.sum(w_I[I:])) \
    + (np.sum(w_C[I:C]) - np.sum(w_C[:I]) - np.sum(w_C[C:])) \
    + (np.sum(w_A[C:A]) - np.sum(w_A[A:]) - np.sum(w_A[:C]))

    return OBJ

class Load_Data(object):
    def __init__(self, train_valid_test = [7,2,1], seed = 123):
        self.data = None
        self.num = 0
        self.rng = np.random.RandomState(seed)
        self.train_valid_test = train_valid_test
    
    def reinit(self):
        self.data = None
        self.num = 0

    def load_csv(self, file_path):
        data = np.loadtxt(file_path, delimiter=',')
        t = data[:, 0:1]
        y = data[:, 1:2]
        ycf = data[:, 2:3]
        mu0 = data[:, 3:4]
        mu1 = data[:, 4:5]
        x = data[:, 5:]
        self.num = self.num + x.shape[0]

        data_list = [x,t,y,ycf,mu0,mu1]

        if self.data == None:
            self.data = data_list
        else:
            for i, _ in enumerate(self.data):
                self.data[i] = np.concatenate((self.data[i], data_list[i]),axis=0)
    
    def load_npz(self, file_path, ind=0):
        data = np.load(file_path)

        t = data['t'][:,ind:ind+1]
        y = data['yf'][:,ind:ind+1]
        ycf = data['ycf'][:,ind:ind+1]
        mu0 = data['mu0'][:,ind:ind+1]
        mu1 = data['mu1'][:,ind:ind+1]
        x = data['x'][:,:,ind]
        self.num = self.num + x.shape[0]

        data_list = [x,t,y,ycf,mu0,mu1]

        if self.data == None:
            self.data = data_list
        else:
            for i, _ in enumerate(self.data):
                self.data[i] = np.concatenate((self.data[i], data_list[i]),axis=0) 

    def split_data(self, train_valid_test=None):
        if train_valid_test==None:
            train_valid_test = (self.num * np.array(self.train_valid_test) / sum(self.train_valid_test)).astype(np.int)
        else:
            self.train_valid_test = train_valid_test
            train_valid_test = (self.num * np.array(train_valid_test) / sum(train_valid_test)).astype(np.int)

        self.train = [d[0:train_valid_test[0], :] for d in self.data]
        self.valid = [d[train_valid_test[0]:int(sum(train_valid_test[0:2])), :] for d in self.data]
        self.test = [d[int(sum(train_valid_test[0:2])):, :] for d in self.data]

        self.train_I, self.valid_I, self.test_I = train_valid_test

        self.to_dict()

    def to_dict(self):
        self.train = self.list_2_dict(self.train)
        self.valid = self.list_2_dict(self.valid)
        self.test = self.list_2_dict(self.test)

        self.train['I'] = np.array(range(self.train_I))
        self.valid['I'] = np.array(range(self.valid_I))
        self.test['I'] = np.array(range(self.test_I))

    def list_2_dict(self, data_list):
        data_dict = {}
        data_dict['x'] = data_list[0]
        data_dict['t'] = data_list[1]
        data_dict['y'] = data_list[2]
        data_dict['ycf'] = data_list[3]
        data_dict['mu0'] = data_list[4]
        data_dict['mu1'] = data_list[5]

        return data_dict

    def shuffle(self):
        p = self.rng.permutation(self.num)
        self.data = [d[p] for d in self.data]
        self.split_data()

def plot_pic(w_I, w_C, w_A, save_path):

    n = w_I.shape[0]
    x = range(0, n)

    # size
    plt.figure(figsize=(20, 8), dpi=80)

    # color
    plt.plot(x, w_I, label="w_I", color="#F08080")
    plt.plot(x, w_C, label="w_C", color="#0000FF", linestyle="--")
    plt.plot(x, w_A, label="w_A", color="#102020", linestyle="-.")

    # x axis
    _xtick_labels = range(0, n)
    plt.xticks(x, _xtick_labels)

    # gird
    plt.grid(alpha=0.4, linestyle=':')

    # legend
    plt.legend(loc="upper left")

    # save
    plt.savefig(save_path)

    # show
    plt.show()

class batch_G(object):
    def __init__(self, data, batch_size, shuffle_=True, seed=123):
        self.data = data
        if batch_size == 0:
            self.batch_size = self.data[0].shape[0]
        else:
            self.batch_size = batch_size
        self.shuffle_ = shuffle_
        self.rng = np.random.RandomState(seed)
        self.batch = self.batch_generator()
        self.num = self.data[0].shape[0]
        self.batch_num = math.ceil(self.num // self.batch_size)

    def shuffle(self):
        num = self.data[0].shape[0]
        p = self.rng.permutation(num)
        self.data = [d[p] for d in self.data]

    def batch_generator(self):
        if self.shuffle_:
            self.shuffle()

        batch_count = 0
        while True:
            if batch_count * self.batch_size + self.batch_size >= len(self.data[0]):
                batch_count = 0
                if self.shuffle_:
                    self.shuffle()

            start = batch_count * self.batch_size
            end = start + self.batch_size
            batch_count += 1

            yield [d[start:end] for d in self.data]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_param():
    '''
    Print all parameters
    :return:
    '''
    for i in tf.compat.v1.global_variables():
        print(i)

def get_params(sess, var_name):
    '''
    The neural network parameters
    :param sess: tf Session
    :param var_name: Neural network parameter names
    :return: Parameter value of neural network
    '''
    variables = tf.trainable_variables()
    params = {}
    for i in range(len(variables)):
        if var_name in variables[i].name:
            name = variables[i].name
            params[name] = sess.run(variables[i])
    return params

def remove(path):
    '''
    Delete Directory or file
    :param path: Directory name or file name
    :return:
    '''
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

def create(path):
    '''
    Create Directory or file
    :param path: Directory name or file name
    :return:
    '''
    if '.' in path:
        p, _ = os.path.split(path)
        if not os.path.exists(p):
            os.makedirs(p)
        file = open(path, 'w')
        file.close()
    elif not os.path.exists(path):
        os.makedirs(path)

def to_one_hot(x, N = -1):
    '''
    to one-hot
    :param x: label
    :param N: the number of classes
    :return: label,one-hot
    '''
    # enocde the label
    x = x.astype('int32')
    if np.min(x) !=0 and N == -1:
        x = x - np.min(x)
    x = x.reshape(-1)

    # The number of classes
    if N == -1:
        N = np.max(x) + 1

    # one-hot
    label = np.zeros((x.shape[0],N))
    idx = range(x.shape[0])
    label[idx,x] = 1
    return label.astype('float32')

def vars_from_scopes(scopes):
    '''
    Parameters list from the variable_scope
    :param scopes: tf.variable_scope
    :return: Trainable parameters
    '''
    current_scope = tf.compat.v1.get_variable_scope().name
    print(current_scope)
    if current_scope != '':
        scopes = [current_scope + '/' + scope for scope in scopes]
    var = []
    for scope in scopes:
        for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            var.append(v)
    return var

def shuffle_aligned_list(data):
    '''
    Shuffle
    :param data: list(data, labels, ...)
    :return: list(data, labels, ...)
    '''
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    '''
    Batch generation of data
    :param data: data
    :param batch_size: batch_size
    :param shuffle: Ture of False
    :return: generator.__next__()
    '''

    # shuffle
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1

        # iterative output
        yield [d[start:end] for d in data]

def get_accuracy(pred, label):
    '''
    Predice accuracy
    :param pred: prediction
    :param label: label
    :return: accuracy
    '''
    if len(pred.shape) > 1:
        pred = np.argmax(pred,axis = 1)
    if len(label.shape) > 1:
        label = np.argmax(label, axis = 1)
        #pdb.set_trace()
    acc = (pred == label).sum().astype('float32')
    return acc/label.shape[0]

class Log(object):
    def __init__(self, log_single, log_all = 'results'):
        '''
        Log：save log file
        :param log_single: single log file
        :param log_all: log file
        '''
        self.log_single = os.path.join(log_single, 'log.txt')
        self.log_all = os.path.join(log_all, 'log.txt')
        create(self.log_single)
        if not os.path.exists(self.log_all):
            file = open(self.log_all, 'w')
            file.close()

    def log(self, str):
        '''
        Write to log file
        :param str: context
        :return:
        '''
        with open(self.log_single, 'a') as f:
            f.write(str + '\n')
        with open(self.log_all, 'a') as f:
            f.write(str + '\n')
        print(str)