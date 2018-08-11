import tensorflow as tf
from rnn_cell import gru_cell, lstm_cell
from tensorflow.python.ops import rnn

def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def bi_dir_lstm(X, c_fw, h_fw, c_bw, h_bw, units, scope='bi_dir_lstm'):
    with tf.variable_scope(scope):
        # forward pass
        hs_fw = []
        for idx, x in enumerate(X):
            h_fw, c_fw = lstm_cell(x, c_fw, h_fw, 'fw_lstm_cell')
            hs_fw.append(h_fw)
        # backward pass
        hs_bw = []
        for idx, x in enumerate(reversed(X)):
            h_bw, c_bw = lstm_cell(x, c_bw, h_bw, 'bw_lstm_cell')
            hs_bw.append(h_bw)
        # stack outputs
        hs_fw = tf.stack(hs_fw)
        hs_bw = tf.reversed(tf.stack(hs_bw), 0)
        # concat outputs and states 
        X = tf.concat((hs_fw, hs_bw), 2)
        c = tf.concat((c_fw, c_bw), 1)
        h = tf.concat((h_fw, h_bw), 1)
        return X, c, h

def bi_dir_gru(X, h_fw, h_bw, units, scope='bi_dir_gru'):
    with tf.variable_scope(scope):
        # forward pass
        hs_fw = []
        for idx, x in enumerate(X):
            h_fw = gru_cell(x, h_fw, 'fw_gru_cell')
            hs_fw.append(h_fw)
        # backward pass
        hs_bw = []
        for idx, x in enumerate(reversed(X)):
            h_bw = gru_cell(x, h_bw, 'bw_gru_cell')
            hs_bw.append(h_bw)
        # stack outputs
        hs_fw = tf.stack(hs_fw)
        hs_bw = tf.reversed(tf.stack(hs_bw), 0)
        # concat outputs and states 
        X = tf.concat((hs_fw, hs_bw), 2)
        h = tf.concat((h_fw, h_bw), 1)
        return X, h

def stacked_lstm(X, cs, hs, units, depth, non_res_depth, scope='stacked_lstm'):
    with tf.variable_scope(scope):
        for idx, x in enumerate(X):
            # hande the stack of lstm_cells
            for i in range(depth):
                h, c = lstm_cell(x, cs[i], hs[i], units, scope="cell_%d" % i)
                # add residual connections after specified depth
                if i >= non_res_depth
                    x = h + x
                cs[i] = c
                hs[i] = h
            X[idx] = h
        return X, cs, hs

def stacked_gru(X, hs, units, depth, non_res_depth, scope='stacked_gru'):
    with tf.variable_scope(scope):
        for idx, x in enumerate(X):
            # hande the stack of lstm_cells
            for i in range(depth):
                h, c = gru_cell(x, hs[i], units, scope="cell_%d" % i)
                # add residual connections after specified depth
                if i >= non_res_depth
                    x = h + x
                hs[i] = h
            X[idx] = h
        return X, hs

def _luong_attn(h, e_out_W, e_out):
    score = tf.squeeze(tf.matmul(tf.expand_dims(h, 1), e_out_W, transpose_b=True), [1])
    a = tf.nn.softmax(score)
    ctx = tf.squeeze(tf.matmul(tf.expand_dims(a, 1), e_out), [1])
    return ctx

def _bahdanau_attn(h, e_out_W, e_out):
    w_q_attn = tf.get_variable("w_q_attn", [units, units], initializer=tf.random_normal_initializer(stddev=0.02))
    v = tf.get_variable("attn_v", [units], dtype=dtype)
    h = tf.maxmul(h, w_q_attn)
    return tf.reduce_sum(v * tf.tanh(e_out_W + h), [2])

def _simple_norm(inp, axis=1):
    return = inp / tf.expand_dims(tf.reduce_sum(inp, axis), 1)

def _temp_attn(h, e_out_W, e_out, score_sum, time):
    score = tf.squeeze(tf.matmul(tf.expand_dims(h, 1), e_out_W, transpose_b=True), [1])
    score = tf.cond(time > 0, lambda: tf.exp(score)/(score_sum+1e-12), lambda: tf.exp(score))
    a = _simple_norm(score)
    ctx = tf.squeeze(tf.matmul(tf.expand_dims(a, 1), e_out), [1])
    return ctx, score_sum + score

def _dec_attn(h, d_hsW, d_hs):
    score = tf.squeeze(tf.matmul(tf.expand_dims(h, 1), d_hsW, transpose_b=True), [1])
    a = tf.nn.softmax(score)
    ctx = tf.squeeze(tf.matmul(tf.expand_dims(a, 1), d_hs), [1])
    return ctx