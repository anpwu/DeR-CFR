from Module import Net
import tensorflow as tf
import utils
import numpy as np
import random
import os
from my_evaluate import evaluation, ACCURACY_func, PEHE_func_v2, PEHE_func_nn, MSE_func
import csv 

# hyperparameter (parameter_name, default_value, parameter_description)
FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_float('p_alpha', 1e-1, "loss-A")
tf.compat.v1.app.flags.DEFINE_float('p_beta', 1, "loss-I")
tf.compat.v1.app.flags.DEFINE_float('p_gamma', 1, "loss-C_B")
tf.compat.v1.app.flags.DEFINE_float('p_mu', 10, "loss-O")
tf.compat.v1.app.flags.DEFINE_float('p_lambda', 1, "los-Reg")
tf.compat.v1.app.flags.DEFINE_float('lr', 0.001, "Learning rate")
tf.compat.v1.app.flags.DEFINE_float('decay_rate', 1.0, "Weight decay rate")
tf.compat.v1.app.flags.DEFINE_integer('seed', 888, "seed")
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 256, "batch_size")
tf.compat.v1.app.flags.DEFINE_integer('num_experiments', 1, "num_experiments")
tf.compat.v1.app.flags.DEFINE_integer('rep_dim', 256, "The dimension of representation network")
tf.compat.v1.app.flags.DEFINE_integer('rep_layer', 2, "The number of representation network layers")
tf.compat.v1.app.flags.DEFINE_integer('t_dim', 128, "The dimension of treatment network")
tf.compat.v1.app.flags.DEFINE_integer('t_layer', 5, "The number of treatment network layers")
tf.compat.v1.app.flags.DEFINE_integer('y_dim', 128, "The dimension of outcome network")
tf.compat.v1.app.flags.DEFINE_integer('y_layer', 5, "The number of outcome network layers")
tf.compat.v1.app.flags.DEFINE_integer('select_layer', 0, "contribution layer")
tf.compat.v1.app.flags.DEFINE_string('activation', 'elu', "Activation function")
tf.compat.v1.app.flags.DEFINE_string('data_path', 'data/Syn_cont_4_16_16_16_3000/data.csv', "data")
tf.compat.v1.app.flags.DEFINE_string('output_dir', 'DeR_CFR', "output")
tf.compat.v1.app.flags.DEFINE_string('var_from', 'get_variable', "get_variable/Variable")
tf.compat.v1.app.flags.DEFINE_integer('t_is_binary', 1, "The treatment is binary")
tf.compat.v1.app.flags.DEFINE_integer('y_is_binary', 1, "The outcome is binary")
tf.compat.v1.app.flags.DEFINE_integer('reweight_sample', 1, "sample balance")
tf.compat.v1.app.flags.DEFINE_integer('use_p_correction', 1, "fix coef")
tf.compat.v1.app.flags.DEFINE_integer('batch_norm', 0, "batch normalization")
tf.compat.v1.app.flags.DEFINE_integer('rep_normalization', 0, "representation normalization")

def run(train, valid, test, output_dir, FLAGS):
    t_threshold = 0.5

    ys = np.concatenate((train['y'], valid['y']), axis=0)
    y_threshold = np.median(ys)

    ''' Output log file '''
    log = utils.Log('results/' + output_dir)

    ''' Set random seed '''
    log.log("Set random seed: {}".format(FLAGS.seed))
    random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Session '''
    log.log("Session: Open! ")
    tf.compat.v1.reset_default_graph()
    graph = tf.compat.v1.get_default_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=graph, config=config)
    log.log("Session: Create！")

    ''' load data '''
    log.log("load data")
    D = [train['x'], train['t'], train['y'], train['I'],
         train['ycf'], train['mu0'], train['mu1']]
    G = utils.batch_G(D, batch_size=FLAGS.batch_size)

    ''' Initialize all parameters '''
    log.log("Initialize all parameters")
    net = Net(train['x'].shape[0], train['x'].shape[1], FLAGS)
    tf.compat.v1.global_variables_initializer().run(session=sess)
    log.log("tf: Complete!")

    ''' dict '''
    log.log("dict")
    train_dict = {net.x: train['x'], net.t: train['t'], net.y: train['y'],
                  net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: train['I'],
                  net.t_threshold: t_threshold, net.y_threshold: y_threshold}

    valid_dict = {net.x: valid['x'], net.t: valid['t'], net.y: valid['y'],
                  net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: valid['I'],
                  net.t_threshold: t_threshold, net.y_threshold: y_threshold}
    
    if FLAGS.y_is_binary:
        test_t = test['t']
    else: 
        test_t = test['t'] - test['t']

    test_dict = {net.x: test['x'], net.t: test_t, net.y: test['y'],
                 net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: test['I'],
                 net.t_threshold: t_threshold, net.y_threshold: y_threshold}
    
    log.log("metric and setting")
    E = evaluation(train, valid, test, 'pehe_nn')
    train_steps = 3000
    num = 1
    if FLAGS.y_is_binary:
        valid_best = 0
        show_best = 0
    else:
        valid_best = 999
        show_best = 999

    log.log("traing ---- ")
    for i in range(train_steps):

        x, t, y, I, yc, mu0, mu1 = G.batch.__next__()

        batch_dict = {net.x: x, net.t: t, net.y: y, net.do_in: 1.0, net.do_out: 1.0, net.p_t: 0.5, net.I: I, net.t_threshold: t_threshold, net.y_threshold: y_threshold}

        sess.run(net.train, feed_dict=batch_dict)
        if FLAGS.reweight_sample:
            sess.run(net.train_balance, feed_dict=batch_dict)

        if i % 100 == 0:
            if FLAGS.t_is_binary:
                if FLAGS.y_is_binary:
                    loss, y_hat, t_hat = sess.run([net.loss, net.y_pred, net.t_pred_I], feed_dict=batch_dict)
                    valid_y_hat = sess.run(net.y_pred, feed_dict=valid_dict)
                    test_y_hat, test_t_hat = sess.run([net.y_pred, net.t_pred_I], feed_dict=test_dict)
                    valid_y_acc = ACCURACY_func(valid['y'], valid_y_hat) 
                    y_acc = ACCURACY_func(test['y'], test_y_hat) 
                    t_acc = ACCURACY_func(test['t'], test_t_hat) 
                    print(np.concatenate((y[:num, :], y_hat[:num, :], t[:num, :], t_hat[:num, :]), axis=1),
                        "{}, loss: {}, y_acc: {}, t_acc: {}".format(i, loss, y_acc, t_acc))
                    if valid_best < valid_y_acc:
                        valid_best = valid_y_acc
                        show_best = y_acc
                else:
                    loss, y_hat, t_hat = sess.run([net.loss, net.y_pred, net.t_pred_I], feed_dict=batch_dict)
                    valid_y_hat, valid_ycf_hat = sess.run([net.y_pred, net.ycf_pred], feed_dict=valid_dict)
                    test_y0, test_y1, test_t_hat = sess.run([net.y_pred, net.ycf_pred, net.t_pred_I], feed_dict=test_dict)
                    valid_pehe = PEHE_func_nn(valid['x'], valid['t'], valid['y'], valid_y_hat, valid_ycf_hat)
                    pehe = PEHE_func_v2(test['mu1'], test['mu0'], test_y1, test_y0)
                    t_acc = ACCURACY_func(test['t'], test_t_hat) 
                    print(np.concatenate((y[:num, :], y_hat[:num, :], t[:num, :], t_hat[:num, :]), axis=1),
                        "{}, loss: {}, Pehe: {}, t_acc: {}".format(i, loss, pehe, t_acc))
                    if valid_best > valid_pehe:
                        valid_best = valid_pehe
                        show_best = pehe
            else:
                if FLAGS.y_is_binary:
                    pass
                else:
                    loss, y_hat, t_hat = sess.run([net.loss, net.y_pred, net.t_pred_I], feed_dict=batch_dict)
                    valid_y_hat, valid_ycf_hat = sess.run([net.y_pred, net.ycf_pred], feed_dict=valid_dict)
                    test_y0, test_y1, test_t_hat = sess.run([net.y_pred, net.ycf_pred, net.t_pred_I], feed_dict=test_dict)
                    
                    
                    valid_mse = MSE_func(valid['y'], valid_y_hat)
                    test_mse = MSE_func(test['y'], test_y0)
                    t_mse = MSE_func(test['t'], test_t_hat) 

                    print(np.concatenate((y[:num, :], y_hat[:num, :], t[:num, :], t_hat[:num, :]), axis=1),
                        "{}, loss: {}, test_y_mse: {}, test_t_mse: {}".format(i, loss, test_mse, t_mse))
                    if valid_best > valid_mse:
                        valid_best = valid_mse
                        show_best = test_mse

    w_I, w_C, w_A = sess.run([net.w_I_mean, net.w_C_mean, net.w_A_mean], feed_dict=train_dict)
    weight = {'w_I': w_I, 'w_C': w_C, 'w_A': w_A}

    if os.path.exists('results/{}/used_configs.txt'.format(output_dir)):
        with open('results/{}/used_configs.txt'.format(output_dir), 'r') as f:
            ind = len(f.readlines())
    else:
        ind = 0
    
    return show_best, log, ind

def train():

    D_load = utils.Load_Data([63, 27, 10])

    datas = FLAGS.data_path.split(',')
    for path in datas:
        if path[-3:] == 'npz':
            D_load.load_npz(path)
        elif path[-3:] == 'csv':
            D_load.load_csv(path)

    D_load.split_data()
    
    results = []
    for _ in range(FLAGS.num_experiments):
        D_load.shuffle()
        result, log, ind = run(D_load.train, D_load.valid, D_load.test, FLAGS.output_dir, FLAGS)
        results.append(result)
    
    results = np.array(results)
    
    log.log('{}: {} +/- {}'.format(ind, np.mean(results), np.std(results)))



if __name__ == '__main__':
    train()

