import tensorflow as tf

def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def _rnn_dropout(x, kp):
    if kp < 1.0:
        x = tf.nn.dropout(x, kp)
    return x

def rnn_cell(x, h, units, scope='rnn_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02),
    b_init=tf.constant_initializer(0),
    i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        w_dim = shape_list(x)[1] + shape_list(h)[1]
        w = tf.get_variable("w", [w_dim, units], initializer=w_init)
        b = tf.get_variable("b", [units], initializer=b_init)
        x = _rnn_dropout(x, i_kp)
        h = tf.tanh(tf.matmul(tf.concat([x, h], 1), w)+b)
        h = _rnn_dropout(h, o_kp)
        return h, h

def lstm_cell(x, c, h, units, scope='lstm_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02), 
    b_init=tf.constant_initializer(0),
    f_b=1.0, i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        w_dim = shape_list(x)[1] + shape_list(h)[1]
        w = tf.get_variable("w", [w_dim, units * 4], initializer=w_init)
        b = tf.get_variable("b", [units * 4], initializer=b_init)
        x = _rnn_dropout(x, i_kp)
        z = tf.matmul(tf.concat([x, h], 1), w) + b
        i, j, f, o = tf.split(z, 4, 1)
        c = tf.nn.sigmoid(f + f_b) * c + tf.nn.sigmoid(i) * tf.tanh(j)
        h = tf.nn.sigmoid(o) * tf.tanh(c)
        h = _rnn_dropout(h, o_kp)
        return h, c

def peep_lstm_cell(x, c, h, units, scope='peep_lstm_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02), 
    b_init=tf.constant_initializer(0), 
    f_b=1.0, i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        w_dim = shape_list(x)[1] + shape_list(h)[1]
        w = tf.get_variable("w", [w_dim, units * 4], initializer=w_init)
        b = tf.get_variable("b", [units * 4], initializer=b_init)
        w_f_diag = tf.get_variable("w_f_diag", [units], initializer=w_init)
        w_i_diag = tf.get_variable("w_i_diag", [units], initializer=w_init)
        w_o_diag = tf.get_variable("w_o_diag", [units], initializer=w_init)
        x = _rnn_dropout(x, i_kp)
        z = tf.matmul(tf.concat([x, h], 1), w)+b
        i, j, f, o = tf.split(z, num_or_size_splits=4, axis=1)
        c = tf.nn.sigmoid(f + f_b + w_f_diag * c) * c + tf.nn.sigmoid(i + w_i_diag * c) * tf.tanh(j)
        h = tf.nn.sigmoid(o + w_o_diag * c) * tf.tanh(c)
        h = _rnn_dropout(h, o_kp)
        return h, c

def mlstm_cell(x, c, h, units, scope='mlstm_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02), 
    b_init=tf.constant_initializer(0),
    f_b=1.0, i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        x_dim = shape_list(x)[1]
        wx = tf.get_variable("wx", [x_dim, units * 4], initializer=w_init)
        wh = tf.get_variable("wh", [units, units * 4], initializer=w_init)
        wmx = tf.get_variable("wmx", [x_dim, units], initializer=w_init)
        wmh = tf.get_variable("wmh", [units, units], initializer=w_init)
        b = tf.get_variable("b", [units * 4], initializer=b_init)
        x = _rnn_dropout(x, i_kp)
        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, j, f, o = tf.split(z, 4, 1)
        c = tf.nn.sigmoid(f + f_b) * c + tf.nn.sigmoid(i) * tf.tanh(j)
        h = tf.nn.sigmoid(o) * tf.tanh(c)
        h = _rnn_dropout(h, o_kp)
        return h, c

def peep_mlstm_cell(x, c, h, units, scope='peep_mlstm_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02), 
    b_init=tf.constant_initializer(0),
    f_b=1.0, i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        x_dim = shape_list(x)[1]
        wx = tf.get_variable("wx", [x_dim, units * 4], initializer=w_init)
        wh = tf.get_variable("wh", [units, units * 4], initializer=w_init)
        wmx = tf.get_variable("wmx", [x_dim, units], initializer=w_init)
        wmh = tf.get_variable("wmh", [units, units], initializer=w_init)
        w_f_diag = tf.get_variable("w_f_diag", [units], initializer=w_init)
        w_i_diag = tf.get_variable("w_i_diag", [units], initializer=w_init)
        w_o_diag = tf.get_variable("w_o_diag", [units], initializer=w_init)
        b = tf.get_variable("b", [units * 4], initializer=b_init)
        x = _rnn_dropout(x, i_kp)
        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, j, f, o = tf.split(z, 4, 1)
        c = tf.nn.sigmoid(f + f_b + w_f_diag * c) * c + tf.nn.sigmoid(i + w_i_diag * c) * tf.tanh(j)
        h = tf.nn.sigmoid(o + w_o_diag * c) * tf.tanh(c)
        h = _rnn_dropout(h, o_kp)
        return h, c

def l2_mlstm_cell(x, c, h, units, scope='l2_mlstm_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02), 
    b_init=tf.constant_initializer(0),
    f_b=1.0, i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        x_dim = shape_list(x)[1]
        wx = tf.get_variable("wx", [x_dim, units * 4], initializer=w_init)
        wh = tf.get_variable("wh", [units, units * 4], initializer=w_init)
        wmx = tf.get_variable("wmx", [x_dim, units], initializer=w_init)
        wmh = tf.get_variable("wmh", [units, units], initializer=w_init)
        b = tf.get_variable("b", [units * 4], initializer=b_init)
        gx = tf.get_variable("gx", [units * 4], initializer=w_init)
        gh = tf.get_variable("gh", [units * 4], initializer=w_init)
        gmx = tf.get_variable("gmx", [units], initializer=w_init)
        gmh = tf.get_variable("gmh", [units], initializer=w_init)
        wx = tf.nn.l2_normalize(wx, axis=0) * gx
        wh = tf.nn.l2_normalize(wh, axis=0) * gh
        wmx = tf.nn.l2_normalize(wmx, axis=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, axis=0) * gmh
        x = _rnn_dropout(x, i_kp)
        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        c = tf.nn.sigmoid(f + f_b) * c + tf.nn.sigmoid(i) * tf.tanh(u)
        h = tf.nn.sigmoid(o) * tf.tanh(c)
        h = _rnn_dropout(h, o_kp)
        return h, c

def gru_cell(x, h, units, scope='gru_cell', 
    w_init=tf.random_normal_initializer(stddev=0.02), 
    b_init=tf.constant_initializer(0), 
    i_kp=1.0, o_kp=1.0):
    with tf.variable_scope(scope):
        w_dim = shape_list(x)[1] + shape_list(h)[1]
        w_g = tf.get_variable("w_g", [w_dim, units * 2], initializer=w_init)
        b_g = tf.get_variable("b_g", [units * 2], initializer=b_init)
        w_c = tf.get_variable("w_c", [w_dim, units], initializer=w_init)
        b_c = tf.get_variable("b_c", [units], initializer=b_init)
        x = _rnn_dropout(x, i_kp)
        g = tf.nn.sigmoid(tf.matmul(tf.concat([x, h], 1), w_g)+b_g)
        r, z = tf.split(g, num_or_size_splits=2, axis=1)
        c = tf.tanh(tf.matmul(tf.concat([x, r * h], 1), w_c)+b_c)
        h = z * h + (1 - z) * c
        h = _rnn_dropout(h, o_kp)
        return h