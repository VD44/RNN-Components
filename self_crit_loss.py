import tensorflow as tf
from tensorflow.python.ops.distributions import categorical

def shape_list(x):
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def seq2seq_self_critical_loss(logits, targets, metric_fn, gamma, target_seq_lens):

    # loss function to reward objective 'metric_fn'
    # sample from logits, if sampled ids achieve higher rewards from 'metric_fn'
    # than do greedy logits: reinforce sample ids; else: punish sample ids.

    # ml_losses = cross_entropy(logits, targets)
    # rl_losses = (metric(sampled_out) - metric(greedy_out)) * cross_entropy(logits, sample_ids)
    # losses = gamma * rl_losses + (1 - gamma) * ml_losses

    # args:
    #
    # logits: logits to calculate loss for, should be batch major order
    #
    # targets: ground truth values to compare logits to, should be batch major order
    #
    # metric_fn: metric which loss will reinforce (Rouge, Bleu, Gleu, etc)
    #
    # gamma: constant : final loss = gamma * rl_losses + (1 - gamma) * ml_losses,
    # gamma = 0.9984 used in https://arxiv.org/pdf/1705.04304.pdf with rouge_l as metric
    #
    # target_seq_lens : tensor of shape [batch_size] used to regularize loss for each sequence

    # mask used to regularize loss for each sequence
    M = tf.sequence_mask(target_seq_lens, shape_list(logits)[1], tf.float32)
    # calculate cross entropy between logits and targets and average across time axis
    # adjusting for each sequence length
    ml_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    ml_losses = tf.reduce_sum(ml_losses*M, 1)/tf.reduce_sum(M, 1)

    greedy_ids = tf.argmax(logits)
    sample_ids = categorical.Categorical(logits=logits).sample()

    metric_losses = metric_fn(sample_ids, targets) - metric_fn(greedy_ids, targets)
    # calculate cross entropy between logits and sample ids and average across time axis
    # adjusting for each sequence length
    rl_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sample_ids)
    rl_losses = tf.reduce_sum(rl_losses*M, 1)/tf.reduce_sum(M, 1)

    losses = gamma * rl_losses + (1 - gamma) * ml_losses
    return losses