import numpy as np

def MSE_func(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def PEHE_func_v1(y, ycf, y_pred, ycf_pred):
    eff = y - ycf
    eff_pred = y_pred - ycf_pred

    return np.sqrt(np.mean(np.square(eff-eff_pred)))

def PEHE_func_v2(y1, y0, y1_pred, y0_pred):
    eff = y1 - y0
    eff_pred = y1_pred - y0_pred

    return np.sqrt(np.mean(np.square(eff - eff_pred)))

def PEHE_func_nn(x, t, y, y_pred, ycf_pred):
    nn_0, nn_1, nn = x_NN(x, t)
    ycf = 1.0 * y[nn]

    return PEHE_func_v1(y, ycf, y_pred, ycf_pred)

def ATE_func(y, ycf, t):
    eff = y - ycf
    eff[t < 0.5] = -eff[t < 0.5]

    return np.mean(eff)

def ATT_func(y, ycf, t):
    eff = y - ycf
    eff[t < 0.5] = -eff[t < 0.5]

    return np.mean(eff[t > 0.5])

def ATC_func(y, ycf, t):
    eff = y - ycf
    eff[t < 0.5] = -eff[t < 0.5]

    return np.mean(eff[t < 0.5])

def ACCURACY_func(label, pred):
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    acc = (pred == label).sum().astype('float32')

    return acc/label.shape[0]

def Distance_M(X,Y):
    C = -2*X.dot(Y.T)
    nx = np.sum(np.square(X),1,keepdims=True)
    ny = np.sum(np.square(Y),1,keepdims=True)
    D = (C + ny.T) + nx
    return np.sqrt(D + 1e-8)

def x_NN(x, t):
    I1 = np.array(np.where(t > 0.5))[0,:]
    I0 = np.array(np.where(t < 0.5))[0,:]

    x_1 = x[I1,:]
    x_0 = x[I0,:]

    D = Distance_M(x_0, x_1)

    nn_1 = I0[np.argmin(D,0)]       # 为T=1找到的T0中的最近邻居
    nn_0 = I1[np.argmin(D,1)]       # 为T=0找到的T1中的最近邻居

    nn = np.zeros((t.shape[0]), dtype=np.int32)
    nn[I0] = nn_0
    nn[I1] = nn_1

    return nn_0, nn_1, nn


class evaluation(object):
    def __init__(self, train, valid, test, target='pehe_nn'):
        self.target = target
        self.value = 99999
        self.step = 0

        self.train = train
        self.valid = valid
        self.test  = test

        self.results = None

    def re_init(self, train, valid, test, target):
        self.target = target
        self.value = 99999
        self.step = 0

        self.train = train
        self.valid = valid
        self.test = test

        self.results = None

    def choose_early_stoppint(self, x=None, t=None, t_pred=None, y=None, ycf=None, y_pred=None, ycf_pred=None, step=None, loss=None):
        if self.target == 'pehe_nn':
            new_value = PEHE_func_nn(x, t, y, y_pred, ycf_pred)
            if new_value < self.value:
                self.value = new_value
                self.step = step

                return True


        elif self.target == 'pehe':
            new_value = PEHE_func_v1(y, ycf, y_pred, ycf_pred)
            if new_value < self.value:
                self.value = new_value
                self.step = step

                return True

        elif self.target == 'loss':
            new_value = loss
            if new_value < self.value:
                self.value = new_value
                self.step = step

                return True

        return False

    def figure_out(self, x=None, t=None, t_pred=None, y=None, ycf=None, y_pred=None, ycf_pred=None):
        result = {}
        result['y_mse'] = MSE_func(y, y_pred)
        result['ycf_mse'] = MSE_func(ycf, ycf_pred)
        result['pehe_nn'] = PEHE_func_nn(x, t, y, y_pred, ycf_pred)
        result['pehe'] = PEHE_func_v1(y, ycf, y_pred, ycf_pred)
        result['ATE'] = ATE_func(y, ycf, t)
        result['ATT'] = ATT_func(y, ycf, t)
        result['ATC'] = ATC_func(y, ycf, t)
        result['pred_ATE'] = ATE_func(y_pred, ycf_pred, t)
        result['pred_ATT'] = ATT_func(y_pred, ycf_pred, t)
        result['pred_ATC'] = ATT_func(y_pred, ycf_pred, t)
        result['bia_ATE'] = np.abs(ATE_func(y, ycf, t) - ATE_func(y_pred, ycf_pred, t))
        result['bia_ATT'] = np.abs(ATT_func(y, ycf, t) - ATT_func(y_pred, ycf_pred, t))
        result['bia_ATC'] = np.abs(ATC_func(y, ycf, t) - ATT_func(y_pred, ycf_pred, t))
        result['acc'] = ACCURACY_func(t, t_pred)

        return result

    def get_results(self):
        results = {}

        train_result = self.figure_out(
            x=self.train['x'], t=self.train['t'], t_pred=self.train['t_pred'],
            y=self.train['y'], ycf=self.train['ycf'],
            y_pred=self.train['y_pred'], ycf_pred=self.train['ycf_pred'])

        valid_result = self.figure_out(
            x=self.valid['x'], t=self.valid['t'], t_pred=self.valid['t_pred'],
            y=self.valid['y'], ycf=self.valid['ycf'],
            y_pred=self.valid['y_pred'], ycf_pred=self.valid['ycf_pred'])

        test_result = self.figure_out(
            x=self.test['x'], t=self.test['t'], t_pred=self.test['t_pred'],
            y=self.test['y'], ycf=self.test['ycf'],
            y_pred=self.test['y_pred'], ycf_pred=self.test['ycf_pred'])

        results['train'] = train_result
        results['valid'] = valid_result
        results['test'] = test_result
        results['step'] = self.step

        self.results = results

        return results






