import numpy as np
import scipy.special
import csv
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from generator.csv2npz import np2npz
from generator.plotting import draw_w

storage_path = '.'

def lindisc(X, t, p):
    ''' Linear MMD '''

    it = np.where(t > 0)
    ic = np.where(t < 1)

    Xc = X[ic]
    Xt = X[it]

    mean_control = np.mean(Xc, axis=0)
    mean_treated = np.mean(Xt, axis=0)

    c = np.square(2 * p - 1) * 0.25
    f = np.sign(p - 0.5)

    mmd = np.sum(np.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + np.sqrt(c + mmd)

    return mmd


def get_multivariate_normal_params(m, dep, seed=0):
    np.random.seed(seed)

    if dep:
        mu = np.random.normal(size=m) / 10.
        ''' sample random positive semi-definite matrix for cov '''
        temp = np.random.uniform(size=(m, m))
        temp = .5 * (np.transpose(temp) + temp)
        sig = (temp + m * np.eye(m)) / 100.

    else:
        mu = np.zeros(m)
        sig = np.eye(m)

    return mu, sig


def get_latent(m, seed, n, dep):
    L = np.array((n * [[]]))
    if m != 0:
        mu, sig = get_multivariate_normal_params(m, dep, seed)
        L = np.random.multivariate_normal(mean=mu, cov=sig, size=n)
    return L


def plot(z, pi0_t1, t, y, data_path, file_name):
    gridspec.GridSpec(3, 1)

    z_min = np.min(z)  # - np.std(z)
    z_max = np.max(z)  # + np.std(z)
    z_grid = np.linspace(z_min, z_max, 100)

    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ind = np.where(t == 0)
    plt.plot(z[ind], np.squeeze(y[ind, 0]), '+', color='r')
    ind = np.where(t == 1)
    plt.plot(z[ind], np.squeeze(y[ind, 1]), '.', color='b')
    plt.legend(['t=0', 't=1'])

    ax = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    ind = np.where(t == 0)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='r', linewidth=2)
    ind = np.where(t == 1)
    mu, std = norm.fit(z[ind])
    p = norm.pdf(z_grid, mu, std)
    plt.plot(z_grid, p, color='b', linewidth=2)

    plt.savefig(data_path + 'info/' + file_name + '.png')
    plt.close()


def run(run_dict, size=0):
    ''' Setting... '''
    mA = run_dict["mA"]  # Dimensions of instrumental variables
    mB = run_dict["mB"]  # Dimensions of confounding variables
    mC = run_dict["mC"]  # Dimensions of adjustment variables
    mD = run_dict["mD"]  # Dimensions of irrelevant variables
    sc = run_dict["sc"]  # 1
    sh = run_dict["sh"]  # 0
    init_seed = run_dict["init_seed"] # Fixed random seed

    # Dataset size
    if size ==0:
        size = run_dict["size"]

    # File path
    file = run_dict["file"]

    # continuous or binary
    binary = run_dict["binary"]

    # Number of repeated experiments
    num = run_dict["num"]

    # Randomly generated coefficients
    random_coef = run_dict["random_coef"]

    # 固定的系数值
    coef_t_AB = run_dict["coef_t_AB"]
    coef_y1_BC = run_dict["coef_y1_BC"]
    coef_y2_BC = run_dict["coef_y2_BC"]

    # Fixed coefficient value
    use_one = run_dict["use_one"]

    # harder datasets
    dep = 0  # overwright; dep=0 generates harder datasets

    # Dataset size
    n_trn = size

    # Coefficient of random seed allocation
    seed_coef = 10

    # all dimension
    max_dim = mA + mB + mC + mD

    # Output folder
    which_benchmark = 'Syn_' + '_'.join(str(item) for item in [sc, sh, dep])

    # Variables
    temp = get_latent(max_dim, seed_coef * init_seed + 4, n_trn, dep)

    # Divide I, C, A, D
    A = temp[:, 0:mA]
    B = temp[:, mA:mA + mB]
    C = temp[:, mA + mB:mA + mB + mC]
    D = temp[:, mA + mB + mC:mA + mB + mC + mD]

    # X: all data; AB: variable related T; BC: variable related Y
    x = np.concatenate([A, B, C, D], axis=1)
    AB = np.concatenate([A, B], axis=1)
    BC = np.concatenate([B, C], axis=1)

    # coef_t_AB 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
    np.random.seed(1 * seed_coef * init_seed)
    if random_coef == "True" or random_coef == "T":
        coefs_1 = np.random.normal(size=mA + mB)
    else:
        coefs_1 = np.array(coef_t_AB)
    if use_one == "True" or use_one == "T":
        coefs_1 = np.ones(shape=mA + mB)

    # generate t_cont and t_binary
    z = np.dot(AB, coefs_1)
    if random_coef == "True" or random_coef == "T" or use_one == "True" or use_one == "T":   
        pass
    else:
        z = z / run_dict["coef_devide_1"]
    per = np.random.normal(size=n_trn)
    pi0_t1 = scipy.special.expit(sc * (z + sh + per))
    t_cont = pi0_t1
    t = np.array([])
    for p in pi0_t1:
        t = np.append(t, np.random.binomial(1, p, 1))
    

    # coef_y_BC 1: Random normal generation; 2: fixed coefficient; 3: 1 coefficient
    np.random.seed(2 * seed_coef * init_seed)  # <--
    if random_coef == "True" or random_coef == "T":
        coefs_2 = np.random.normal(size=mB + mC)
    else:
        coefs_2 = np.array(coef_y1_BC)
    if use_one == "True" or use_one == "T":
        coefs_2 = np.ones(shape=mB + mC)
    if random_coef == "True" or random_coef == "T":
        coefs_3 = np.random.normal(size=mB + mC)
    else:
        coefs_3 = np.array(coef_y2_BC)
    if use_one == "True" or use_one == "T":
        coefs_3 = np.ones(shape=mB + mC)
    
    # base mu_0, mu_1
    if random_coef == "True" or random_coef == "T" or use_one == "True" or use_one == "T":   
        mu_0 = np.dot(BC ** 1, coefs_2) / (mB + mC)
        mu_1 = np.dot(BC ** 2, coefs_3) / (mB + mC)
    else:
        mu_0 = np.dot(BC ** 1, coefs_2) / (mB + mC) / run_dict["coef_devide_2"]
        mu_1 = np.dot(BC ** 2, coefs_3) / (mB + mC) / run_dict["coef_devide_3"]
    
        # print("up",mu_0)

    # continuous y
    y_cont = mu_0 + t_cont * mu_1 + np.random.normal(loc=0., scale=.1, size=n_trn)

    # binary y
    y = np.zeros((n_trn, 2))
    if binary == "False" or binary == "F":
        np.random.seed(3 * seed_coef * init_seed)  # <--
        y[:, 0] = mu_0 + np.random.normal(loc=0., scale=.1, size=n_trn)
        np.random.seed(3 * seed_coef * init_seed)  # <--
        y[:, 1] = mu_1 + np.random.normal(loc=0., scale=.1, size=n_trn)
    else:
        mu_0_ = np.dot(BC ** 1, coefs_2)
        mu_1_ = np.dot(BC ** 2, coefs_3)
        if random_coef == "True" or random_coef == "T" or use_one == "True" or use_one == "T":   
            mu_0 = mu_0_ 
            mu_1 = mu_1_ 
        else:
            mu_0 = mu_0_ / (mB + mC) / run_dict["coef_devide_2"]
            mu_1 = mu_1_ / (mB + mC) / run_dict["coef_devide_3"]
            mu_00 = mu_0_ / (mB + mC) / run_dict["coef_devide_2"]
            mu_11 = mu_1_ / (mB + mC) / run_dict["coef_devide_3"]

            # print("bottom",mu_0)
            
        # median__ = np.median(np.concatenate((mu_0,mu_1),axis=0))
        # mu_0[mu_0 < median__] = 0.
        # mu_0[mu_0 >= median__] = 1.
        # mu_1[mu_1 < median__] = 0.
        # mu_1[mu_1 >= median__] = 1.

        median_0 = np.median(mu_0)
        median_1 = np.median(mu_1)
        mu_0[mu_0 >= median_0] = 1.
        mu_0[mu_0 < median_0] = 0.    
        mu_1[mu_1 < median_1] = 0.
        mu_1[mu_1 >= median_1] = 1.

        print("the number of t  :", np.sum(t))
        print("the number of y_0:", np.sum(mu_0))
        print("the number of y_1:", np.sum(mu_1))

        y_0 = mu_0
        y_1 = mu_1

        y[:, 0] = y_0
        y[:, 1] = y_1
    
    yf = np.array([])
    ycf = np.array([])
    for i, t_i in enumerate(t):
        yf = np.append(yf, y[i, int(t_i)])
        ycf = np.append(ycf, y[i, int(1 - t_i)])

    ##################################################################
    # data_path = storage_path+'/data/'+which_benchmark
    data_path = storage_path + '/data/'

    if not os.path.exists(storage_path + '/data/'):
        os.mkdir(storage_path + '/data/')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # dataset
    which_dataset = '_'.join(str(item) for item in [mA, mB, mC]) + '_' + str(size)
    data_path = data_path + file + '_' + str(init_seed) + '_' + which_dataset + '/'

    if not os.path.exists(data_path + 'info/'):
        os.makedirs(data_path + 'info/')

    # clear
    f = open(data_path + 'info/config.txt', 'w')
    f.write(which_benchmark + '_' + which_dataset)
    f.close()

    # write coefs
    f = open(data_path + 'info/coefs.txt', 'w')
    f.write(str(coefs_1) + '\n')
    f.write(str(coefs_2) + '\n')
    f.write(str(coefs_3) + '\n')
    f.close()

    # seed
    np.random.seed(4 * seed_coef * init_seed + init_seed)

    # data.csv
    file_name = 'data'
    with open(data_path + file_name + '.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for i in np.random.permutation(n_trn):
            if binary == "False" or binary == "F":
                temp = [t[i], yf[i], ycf[i], mu_0[i], mu_1[i]]
                temp.extend(x[i, :])
                csv_writer.writerow(temp)
            else:
                temp = [t[i], yf[i], ycf[i], y[i, 0], y[i, 1]]
                temp.extend(x[i, :])
                csv_writer.writerow(temp)
    
    # data_cont.csv
    file_cont_name = 'data_cont'
    with open(data_path + file_cont_name + '.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for i in np.random.permutation(n_trn):
            temp = [t_cont[i], y_cont[i], mu_0[i], mu_1[i], mu_0[i] + t_cont[i] * mu_1[i]]
            temp.extend(x[i, :])
            csv_writer.writerow(temp)

    if binary == "False" or binary == "F":
        pass
    else:
        file_name = 'probability'
        with open(data_path + file_name + '.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            for i in np.random.permutation(n_trn):
                temp = [t[i], y[i, 0], y[i, 1], mu_00[i], mu_11[i]]
                csv_writer.writerow(temp)

    num_pts = 250
    plot(z[:num_pts], pi0_t1[:num_pts], t[:num_pts], y[:num_pts], data_path, file_name)

    with open(data_path + 'info/specs.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        temp = [np.mean(t), np.min(pi0_t1), np.max(pi0_t1), np.mean(pi0_t1), np.std(pi0_t1)]
        temp.append(lindisc(x, t, np.mean(t)))
        csv_writer.writerow(temp)

    print('*' * 20)

    np2npz(data_path, num)
    draw_w(coefs_1,coefs_2,coefs_3,data_path + 'info/')


