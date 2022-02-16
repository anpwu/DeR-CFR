import tensorflow as tf
import numpy as np
import utils

class Net(object):
    def __init__(self, n, x_dim, FLAGS):
        self.n = n
        self.x_dim = x_dim
        self.FLAGS = FLAGS

        self.wd_loss = 0
        self.lr = FLAGS.lr

        self.x = tf.compat.v1.placeholder('float', shape=[None, x_dim], name='x')
        self.t = tf.compat.v1.placeholder('float', shape=[None, 1], name='t')
        self.y = tf.compat.v1.placeholder('float', shape=[None, 1], name='y')
        self.do_in = tf.compat.v1.placeholder("float", name='dropout_in')
        self.do_out = tf.compat.v1.placeholder("float", name='dropout_out')
        self.p_t = tf.compat.v1.placeholder("float", name='p_treated')
        self.t_threshold = tf.compat.v1.placeholder("float", name='treatment_threshold')
        self.y_threshold = tf.compat.v1.placeholder("float", name='outcome_threshold')
        self.I = tf.compat.v1.placeholder("int32", shape=[None, ], name='I')

        if FLAGS.activation.lower() == 'elu':
            self.activation = tf.nn.elu
        elif FLAGS.activation.lower() == 'tanh':
            self.activation = tf.nn.tanh
        else:
            self.activation = tf.nn.relu

        self.i0 = tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32)
        self.i1 = tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32)
        self.initializer = tf.contrib.layers.xavier_initializer(seed=FLAGS.seed)

        self.build_graph()
        self.ICA_W_setting()
        self.IPM()
        self.calculate_loss()
        self.setup_train_ops()
    
    def FC_layer(self, dim_in, dim_out, name, wd=0, b=True, weight_init = 0.01):

        bias = 0

        if self.FLAGS.var_from == 'get_variable':
            weight = tf.get_variable(name='weight' + name, shape=[dim_in, dim_out], initializer=self.initializer)
            if b:
                bias = tf.get_variable(name='bias' + name, shape=[1, dim_out], initializer=tf.constant_initializer())
        else:
            weight = tf.Variable(tf.random.normal([dim_in, dim_out], stddev=weight_init / np.sqrt(dim_in)), name='weight' + name)
            if b:
                bias = tf.Variable(tf.zeros([1, dim_out]), name='bias' + name)

        if wd>0:
            self.wd_loss += wd * tf.nn.l2_loss(weight)

        return weight, bias
    
    def build_graph(self):

        with tf.compat.v1.variable_scope('weight'):
            if self.FLAGS.reweight_sample:
                sample_weight = tf.get_variable(name='sample_weight', shape=[self.n, 1], initializer=tf.constant_initializer(1))
                self.sample_weight = tf.gather(sample_weight, self.I)
            else:
                self.sample_weight = 1.0
        
        with tf.compat.v1.variable_scope('representation'):
            self.rep_I, self.reps_I, self.w_I, self.b_I = self.representation(input=self.x,
                                                                              dim_in=self.x_dim,
                                                                              dim_out=self.FLAGS.rep_dim,
                                                                              layer=self.FLAGS.rep_layer,
                                                                              name='Instrument')
            self.rep_C, self.reps_C, self.w_C, self.b_C = self.representation(input=self.x,
                                                                              dim_in=self.x_dim,
                                                                              dim_out=self.FLAGS.rep_dim,
                                                                              layer=self.FLAGS.rep_layer,
                                                                              name='Confounder')
            self.rep_A, self.reps_A, self.w_A, self.b_A = self.representation(input=self.x,
                                                                              dim_in=self.x_dim,
                                                                              dim_out=self.FLAGS.rep_dim,
                                                                              layer=self.FLAGS.rep_layer,
                                                                              name='Adjustment')
        
        with tf.compat.v1.variable_scope('treatment'):
            self.mu_T, self.mus_T, self.w_muT, self.b_muT = self.treatment(
                input=tf.concat((self.rep_I, self.rep_C), axis=1),
                dim_in=self.FLAGS.rep_dim * 2,
                dim_out=self.FLAGS.t_dim,
                layer=self.FLAGS.t_layer,
                name='Mu_Treatment')
            
            self.mu_T_I, self.mus_T_I, self.w_muT_I, self.b_muT_I = self.treatment(
                input=tf.concat(self.rep_I, axis=1),
                dim_in=self.FLAGS.rep_dim,
                dim_out=self.FLAGS.t_dim,
                layer=self.FLAGS.t_layer,
                name='Mu_Treatment_I')
        
        if self.FLAGS.t_is_binary:
            with tf.compat.v1.variable_scope('outcome'):
                self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(
                    input=tf.concat((self.rep_C, self.rep_A), axis=1),
                    dim_in=self.FLAGS.rep_dim * 2,
                    dim_out=self.FLAGS.y_dim,
                    layer=self.FLAGS.y_layer,
                    name='Mu_ytx')
                
                self.mu_Y_A, self.mu_YCF_A, self.mus_Y_A, self.w_muY_A, self.b_muY_A = self.output(
                    input=tf.concat(self.rep_A, axis=1),
                    dim_in=self.FLAGS.rep_dim,
                    dim_out=self.FLAGS.y_dim,
                    layer=self.FLAGS.y_layer,
                    name='Mu_ytx_A')    
        else:
            with tf.compat.v1.variable_scope('outcome'):
                self.mu_Y, self.mu_YCF, self.mus_Y, self.w_muY, self.b_muY = self.output(
                    input=tf.concat((self.rep_C, self.rep_A, self.t), axis=1),
                    dim_in=self.FLAGS.rep_dim * 2 + 1,
                    dim_out=self.FLAGS.y_dim,
                    layer=self.FLAGS.y_layer,
                    name='Mu_ytx')
                
                self.mu_Y_A, self.mu_YCF_A, self.mus_Y_A, self.w_muY_A, self.b_muY_A = self.output(
                    input=tf.concat((self.rep_A, self.t), axis=1),
                    dim_in=self.FLAGS.rep_dim + 1,
                    dim_out=self.FLAGS.y_dim,
                    layer=self.FLAGS.y_layer,
                    name='Mu_ytx_A')

    def ICA_W_setting(self):

        if self.FLAGS.select_layer == 0:
            layer_num = len(self.w_I)
        else:
            layer_num = self.FLAGS.select_layer

        w_I_sum, w_C_sum, w_A_sum = self.w_I[0], self.w_C[0], self.w_A[0]
        for i in range(1, layer_num):
            w_I_sum = tf.matmul(w_I_sum, self.w_I[i])
            w_C_sum = tf.matmul(w_C_sum, self.w_C[i])
            w_A_sum = tf.matmul(w_A_sum, self.w_A[i])
        self.w_I_mean = tf.reduce_mean(tf.abs(w_I_sum), axis=1)
        self.w_C_mean = tf.reduce_mean(tf.abs(w_C_sum), axis=1)
        self.w_A_mean = tf.reduce_mean(tf.abs(w_A_sum), axis=1)

    def IPM(self):

        if self.FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        ########################################## Instrumental ########################################

        i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y >= self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
        i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t >= self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y < self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
        i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y >= self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))
        i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.cast(tf.where(self.t < self.t_threshold)[:, 0], dtype=tf.int32), tf.cast(tf.where(self.y < self.y_threshold)[:, 0], dtype=tf.int32)],axis=0)))

        if self.FLAGS.reweight_sample:
            w_1_1 = tf.gather(self.sample_weight,i_1_1)
            w_1_0 = tf.gather(self.sample_weight,i_1_0)
            w_0_1 = tf.gather(self.sample_weight,i_0_1)
            w_0_0 = tf.gather(self.sample_weight,i_0_0)
        else:
            w_1_1 = 1
            w_1_0 = 1
            w_0_1 = 1
            w_0_0 = 1

        I_1_1 = tf.gather(self.rep_I,i_1_1)
        I_1_0 = tf.gather(self.rep_I,i_1_0)
        I_0_1 = tf.gather(self.rep_I,i_0_1)
        I_0_0 = tf.gather(self.rep_I,i_0_0)

        mean_1_1 = tf.reduce_mean(w_1_1 * I_1_1,axis=0)
        mean_1_0 = tf.reduce_mean(w_1_0 * I_1_0,axis=0)
        mean_0_1 = tf.reduce_mean(w_0_1 * I_0_1,axis=0)
        mean_0_0 = tf.reduce_mean(w_0_0 * I_0_0,axis=0)

        mmd_1 = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_1_1 - 2.0 * (1.0 - p_ipm) * mean_1_0))
        mmd_0 = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_0_1 - 2.0 * (1.0 - p_ipm) * mean_0_0))

        self.IPM_I = mmd_0 + mmd_1

        ########################################### Condounder #########################################

        self.rep_C_0 = tf.gather(self.rep_C, self.i0)
        self.rep_C_1 = tf.gather(self.rep_C, self.i1)

        if self.FLAGS.reweight_sample:
            self.sample_weight_0 = tf.gather(self.sample_weight, self.i0)
            self.sample_weight_1 = tf.gather(self.sample_weight, self.i1)
        else:
            self.sample_weight_0 = 1
            self.sample_weight_1 = 1

        mean_C_0 = tf.reduce_mean(self.sample_weight_0 * self.rep_C_0, reduction_indices=0)
        mean_C_1 = tf.reduce_mean(self.sample_weight_1 * self.rep_C_1, reduction_indices=0)

        self.IPM_C = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_C_1 - 2.0 * (1.0 - p_ipm) * mean_C_0))

        ########################################### Adjustment #########################################

        self.rep_A_0 = tf.gather(self.rep_A, self.i0)
        self.rep_A_1 = tf.gather(self.rep_A, self.i1)

        mean_A_0 = tf.reduce_mean(self.rep_A_0, reduction_indices=0)
        mean_A_1 = tf.reduce_mean(self.rep_A_1, reduction_indices=0)

        self.IPM_A = tf.reduce_sum(tf.square(2.0 * p_ipm * mean_A_1 - 2.0 * (1.0 - p_ipm) * mean_A_0))

    def calculate_loss(self):
        if self.FLAGS.t_is_binary:
            self.t_pred, _, self.loss_IC_T = self.log_loss(self.mu_T, self.t)
            self.t_pred_I, _, self.loss_I_T = self.log_loss(self.mu_T_I, self.t)
        else: 
            self.t_pred, self.t_pred_I = self.mu_T, self.mu_T_I
            self.loss_IC_T, _ = self.l2_loss(self.mu_T, self.t)
            self.loss_I_T, _ = self.l2_loss(self.mu_T_I, self.t)

        if self.FLAGS.y_is_binary:
            self.y_pred, _, self.loss_TCA_Y = self.log_loss(self.mu_Y, self.y, True)
            self.y_pred_A, _, self.loss_TA_Y = self.log_loss(self.mu_Y_A, self.y)
            self.ycf_pred, _, _ = self.log_loss(self.mu_YCF, self.y)
        else:
            self.y_pred, self.y_pred_A, self.ycf_pred = self.mu_Y, self.mu_Y_A, self.mu_YCF
            self.loss_TCA_Y, _ = self.l2_loss(self.mu_Y, self.y, True)
            self.loss_TA_Y, _ = self.l2_loss(self.mu_Y_A, self.y)


        self.loss_ICA = tf.reduce_sum(self.w_I_mean * self.w_C_mean) + tf.reduce_sum(self.w_I_mean * self.w_A_mean) + tf.reduce_sum(self.w_C_mean * self.w_A_mean)
        self.loss_ICA_1 = tf.square(tf.reduce_sum(self.w_I_mean) - 1.0) + tf.square(tf.reduce_sum(self.w_C_mean) - 1.0) + tf.square(tf.reduce_sum(self.w_A_mean) - 1.0)

        self.loss_I_T_Y = self.IPM_I
        self.loss_C_T = self.IPM_C
        self.loss_A_T = self.IPM_A

        self.loss_R = self.loss_TCA_Y
        self.loss_A = self.FLAGS.p_alpha * (self.loss_A_T + self.loss_TA_Y)
        self.loss_I = self.FLAGS.p_beta * (self.loss_I_T_Y + self.loss_I_T)
        self.loss_O = self.FLAGS.p_mu * (self.loss_ICA + self.loss_ICA_1)
        self.loss_Reg = self.FLAGS.p_lambda * (1e-3 * self.wd_loss)
        
        if self.FLAGS.reweight_sample:
            self.loss_w = tf.square(tf.reduce_sum(self.sample_weight_0)/tf.reduce_sum(1.0 - self.t) - 1.0) + tf.square(tf.reduce_sum(self.sample_weight_1)/tf.reduce_sum(self.t) - 1.0)
            self.loss_C_B = self.FLAGS.p_gamma * (self.loss_C_T + self.loss_w)
            self.loss_balance = self.loss_R + self.loss_C_B + self.loss_Reg

        self.loss = self.loss_R + self.loss_A + self.loss_I + self.loss_O + self.loss_Reg

    def setup_train_ops(self):
        W_vars = utils.vars_from_scopes(['weight'])
        R_vars = utils.vars_from_scopes(['representation'])
        T_vars = utils.vars_from_scopes(['treatment'])
        O_vars = utils.vars_from_scopes(['outcome'])

        self.train = tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss, var_list=R_vars+T_vars+O_vars)
        if self.FLAGS.reweight_sample:
            self.train_balance = tf.compat.v1.train.AdamOptimizer(self.lr, 0.5).minimize(self.loss_balance, var_list=W_vars)
    
    def representation(self, input, dim_in, dim_out, layer, name):
        rep, weight, bias = [input], [], []

        dim = np.around(np.linspace(dim_in, dim_out, layer+1)).astype(int)

        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i+1], name='_{}_{}'.format(name, i))
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(rep[i], weight[i], name='matmul_{}_{}'.format(name, i)), bias[i], name='add_{}_{}'.format(name, i))
            if self.FLAGS.batch_norm:
                batch_mean, batch_var = tf.nn.moments(out, [0])
                out = tf.nn.batch_normalization(out, batch_mean, batch_var, 0, 1, 1e-3)

            rep.append(tf.nn.dropout(self.activation(out), self.do_in))
        
        if self.FLAGS.rep_normalization:
            rep[-1] = rep[-1] / utils.safe_sqrt(tf.reduce_sum(tf.square(rep[-1]), axis=1, keep_dims=True))

        return rep[-1], rep, weight, bias
    
    def predict(self, input, dim_in, dim_out, layer, name, wd=0, class_num=1, mode='mu'):
        pred, weight, bias = [input], [], []

        dim = np.around(np.linspace(dim_in, dim_out, layer + 1)).astype(int)

        for i in range(0, layer):
            w, b = self.FC_layer(dim_in=dim[i], dim_out=dim[i + 1], name='_{}_{}'.format(name, i), wd=wd)
            weight.append(w)
            bias.append(b)
            out = tf.add(tf.matmul(pred[i], weight[i], name='matmul_{}_{}'.format(name, i)), bias[i], name='add_{}_{}'.format(name, i))
            pred.append(tf.nn.dropout(self.activation(out), self.do_out))

        w, b = self.FC_layer(dim_in=dim[-1], dim_out=class_num, name='_{}_{}'.format(name, 'pred'), wd=wd)
        weight.append(w)
        bias.append(b)
        out = tf.add(tf.matmul(pred[-1], weight[-1], name='matmul_{}_{}'.format(name, 'pred')), bias[-1],name='add_{}_{}'.format(name, 'pred'))
        if mode == 'mu':
            pred.append(out)
            # pred.append(tf.nn.dropout(out, self.do_out))
        else:
            pred.append(tf.nn.tanh(out))
            # pred.append(tf.nn.dropout(tf.nn.tanh(out), self.do_out))

        return pred[-1], pred, weight, bias
    
    def treatment(self, input, dim_in, dim_out, layer, name, mode='mu'):
        if self.FLAGS.t_is_binary:
            mu_T, mus_T, w_muT, b_muT = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, 2, mode)
        else:
            mu_T, mus_T, w_muT, b_muT = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, 1, mode)
        return mu_T, mus_T, w_muT, b_muT

    def output(self, input, dim_in, dim_out, layer, name, mode='mu'):
        if self.FLAGS.t_is_binary:
            if self.FLAGS.y_is_binary:
                mu_Y_0, mus_Y_0, w_muY_0, b_muY_0 = self.predict(input, dim_in, dim_out, layer, name+'0', self.FLAGS.decay_rate, 2, mode)
                mu_Y_1, mus_Y_1, w_muY_1, b_muY_1 = self.predict(input, dim_in, dim_out, layer, name+'1', self.FLAGS.decay_rate, 2, mode)
            else:
                mu_Y_0, mus_Y_0, w_muY_0, b_muY_0 = self.predict(input, dim_in, dim_out, layer, name+'0', self.FLAGS.decay_rate, 1, mode)
                mu_Y_1, mus_Y_1, w_muY_1, b_muY_1 = self.predict(input, dim_in, dim_out, layer, name+'1', self.FLAGS.decay_rate, 1, mode)

            mu_YF_0 = tf.gather(mu_Y_0, self.i0)
            mu_YF_1 = tf.gather(mu_Y_1, self.i1)

            mu_YCF_0 = tf.gather(mu_Y_0, self.i1)
            mu_YCF_1 = tf.gather(mu_Y_1, self.i0)

            mu_YF = tf.dynamic_stitch([self.i0, self.i1], [mu_YF_0, mu_YF_1])
            mu_YCF = tf.dynamic_stitch([self.i0, self.i1], [mu_YCF_1, mu_YCF_0])

            mus_Y = mus_Y_0 + mus_Y_1
            w_muY = w_muY_0 + w_muY_1
            b_muY = b_muY_0 + b_muY_1

            return mu_YF, mu_YCF, mus_Y, w_muY, b_muY
        else:
            if self.FLAGS.y_is_binary:
                pass
            else:
                mu_Y, mus_Y, w_muY, b_muY = self.predict(input, dim_in, dim_out, layer, name, self.FLAGS.decay_rate, 1, mode)
            return mu_Y, mu_Y, mus_Y, w_muY, b_muY


    
    def log_loss(self, pred, label, sample = False):
        sigma = 0.995 / (1.0 + tf.exp(-pred)) + 0.0025
        pi_0 = tf.multiply(label, sigma) + tf.multiply(1.0 - label, 1.0 - sigma)

        labels = tf.concat((1 - label, label), axis=1)
        logits = pred

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        if sample and self.FLAGS.reweight_sample:
            loss = tf.reduce_mean(self.sample_weight * loss)
        else:
            loss = tf.reduce_mean(loss)

        return sigma[:,1:2], pi_0, loss
    
    def l2_loss(self, pred, out, sample = False):

        if sample and self.FLAGS.reweight_sample:
            loss = tf.reduce_mean(self.sample_weight * tf.square(pred - out))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred - out)))
        else:
            loss = tf.reduce_mean(tf.square(pred - out))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(pred - out)))

        return loss, pred_error

