import tensorflow as tf


def build_network(n_features, n_actions, lr):
    """ Build the network for RL algorithm.

    :param n_features: input layer's features
    :param n_actions:  How many actions to take in output.
    :param lr: learning rate
    :return:
        [eval_net_input, target_net_input, q_target, ISWeights]: input
        [q_eval_net_out, loss, _train_op, q_target_net_out, abs_errors]: output
        [e_params, t_params]: weights
    """
    # ------------------ all inputs --------------------------
    # input for eval net
    eval_net_input = tf.placeholder(tf.float32, [None, n_features], name='eval_net_input')
    # q_target for loss
    q_target = tf.placeholder(tf.float32, [None, sum(n_actions)], name='q_target')
    # input for target net
    target_net_input = tf.placeholder(tf.float32, [None, n_features], name='target_net_input')
    # ISWeights
    ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
    # initializer
    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

    # ------------------ build evaluate_net ------------------
    with tf.variable_scope('eval_net'):
        ef1 = tf.layers.dense(eval_net_input, 128, tf.nn.relu, kernel_initializer=w_initializer,
                              bias_initializer=b_initializer, name='ef1')
        ef2 = tf.layers.dense(ef1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                              bias_initializer=b_initializer, name='ef2')
        q_eval_net_out1 = tf.layers.dense(ef2, n_actions[0], kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_eval_1')
        eu0 = tf.stack([q_eval_net_out1] * 32)
        eu0 = tf.reshape(eu0, [-1, 128])
        eu1 = tf.concat([ef2, eu0], 1)
        q_eval_net_out2 = tf.layers.dense(eu1, n_actions[1], kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_eval_2')
        q_eval_net_out = tf.concat([q_eval_net_out1, q_eval_net_out2], 1)
        # e1 = tf.layers.dense(f2, 64, tf.nn.relu, kernel_initializer=w_initializer,
        #                      bias_initializer=b_initializer, name='e1')
        # eval_V1 = tf.layers.dense(e1, 1, None, kernel_initializer=w_initializer,
        #                           bias_initializer=b_initializer, name='eval_V1')
        # eval_A1 = tf.layers.dense(e1, n_actions[0], None, kernel_initializer=w_initializer,
        #                           bias_initializer=b_initializer, name='eval_A1')
        # q_eval_net_out1 = eval_V1 + (eval_A1 - tf.reduce_mean(eval_A1, axis=1, keep_dims=True))
        # # concat output and input
        # eu1 = tf.concat([e1, q_eval_net_out1], 1)
        # # as next input
        # eval_V2 = tf.layers.dense(eu1, 1, None, kernel_initializer=w_initializer,
        #                           bias_initializer=b_initializer, name='eval_V2')
        # eval_A2 = tf.layers.dense(eu1, n_actions[1], None, kernel_initializer=w_initializer,
        #                           bias_initializer=b_initializer, name='eval_A2')
        # q_eval_net_out2 = eval_V2 + (eval_A2 - tf.reduce_mean(eval_A2, axis=1, keep_dims=True))
        #
        # q_eval_net_out = tf.concat([q_eval_net_out1, q_eval_net_out2], 1)

    with tf.variable_scope('loss'):
        abs_errors = tf.reduce_sum(tf.abs(q_target - q_eval_net_out), axis=1)  # for updating Sumtree
        loss = tf.reduce_mean(ISWeights * tf.squared_difference(q_target, q_eval_net_out))

    with tf.variable_scope('train'):
        _train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # ------------------ build target_net ------------------
    with tf.variable_scope('target_net'):
        tf1 = tf.layers.dense(eval_net_input, 128, tf.nn.relu, kernel_initializer=w_initializer,
                              bias_initializer=b_initializer, name='tf1')
        tf2 = tf.layers.dense(tf1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                              bias_initializer=b_initializer, name='tf2')
        q_target_net_out1 = tf.layers.dense(tf2, n_actions[0], kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q_target_1')
        tu0 = tf.stack([q_target_net_out1]*32)
        tu0 = tf.reshape(tu0, [-1, 128])
        tu1 = tf.concat([tf2, tu0], 1)
        q_target_net_out2 = tf.layers.dense(tu1, n_actions[1], kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q_target_2')
        q_target_net_out = tf.concat([q_target_net_out1, q_target_net_out2], 1)
        # t1 = tf.layers.dense(f2, 64, tf.nn.relu, kernel_initializer=w_initializer,
        #                      bias_initializer=b_initializer, name='t1')
        # target_V1 = tf.layers.dense(t1, 1, None, kernel_initializer=w_initializer,
        #                             bias_initializer=b_initializer, name='target_V1')
        # target_A1 = tf.layers.dense(t1, n_actions[0], None, kernel_initializer=w_initializer,
        #                             bias_initializer=b_initializer, name='target_A1')
        # q_target_net_out1 = target_V1 + (target_A1 - tf.reduce_mean(target_A1, axis=1, keep_dims=True))
        # # concat output and input
        # tu1 = tf.concat([t1, q_target_net_out1], 1)
        # # as next input
        # target_V2 = tf.layers.dense(tu1, 1, None, kernel_initializer=w_initializer,
        #                             bias_initializer=b_initializer, name='target_V2')
        # target_A2 = tf.layers.dense(tu1, n_actions[1], None, kernel_initializer=w_initializer,
        #                             bias_initializer=b_initializer, name='target_A2')
        # q_target_net_out2 = target_V2 + (target_A2 - tf.reduce_mean(target_A2, axis=1, keep_dims=True))
        #
        # q_target_net_out = tf.concat([q_target_net_out1, q_target_net_out2], 1)

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    return [eval_net_input, target_net_input, q_target, ISWeights], \
           [q_eval_net_out, loss, _train_op, q_target_net_out, abs_errors], \
           [e_params, t_params]
