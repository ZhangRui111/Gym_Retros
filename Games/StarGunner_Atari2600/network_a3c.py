import tensorflow as tf

from Hyper_parameters.hp_a3c import Hyperparameters


def build_network(scope):
    """ Build the network for RL algorithm.

    :param rand -- used to identify different network.
    :return:
        [s, a_prob, v, a_params, c_params]
    """
    # init Hp
    hp = Hyperparameters()

    s = tf.placeholder(tf.float32, [None, hp.N_FEATURES], 'S')

    w_init = tf.random_normal_initializer(0., .1)
    with tf.variable_scope('actor'):
        l_a = tf.layers.dense(s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
        a_prob = tf.layers.dense(l_a, hp.N_ACTIONS, tf.nn.softmax, kernel_initializer=w_init, name='ap')
    with tf.variable_scope('critic'):
        l_c = tf.layers.dense(s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
        v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
    a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
    c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    return [s, a_prob, v, a_params, c_params]


















    # # ------------------ all inputs --------------------------
    # # input for target net
    # eval_net_input = tf.placeholder(tf.float32, [None, hp.IMAGE_LENGTH, hp.IMAGE_LENGTH, hp.AGENT_HISTORY_LENGTH],
    #                                 name='eval_net_input_' + str(rand))
    # # input for eval net
    # target_net_input = tf.placeholder(tf.float32, [None, hp.IMAGE_LENGTH, hp.IMAGE_LENGTH, hp.AGENT_HISTORY_LENGTH],
    #                                   name='target_net_input_' + str(rand))
    # # q_target for loss
    # q_target = tf.placeholder(tf.float32, [None, 2 ** hp.N_ACTIONS], name='q_target_' + str(rand))
    # # initializer
    # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0)
    #
    # # ------------------ build evaluate_net ------------------
    # with tf.variable_scope('eval_net'):
    #     # [None, 96, 96, 4] --> [None, 48, 48, 32]
    #     eval_conv1 = tf.layers.conv2d(eval_net_input, kernel_size=3, activation=tf.nn.relu,
    #                                   kernel_initializer=tf.random_normal_initializer(0., 0.01),
    #                                   filters=32, strides=1, padding='SAME', name='eval_c1_' + str(rand))
    #     eval_pool1 = tf.layers.max_pooling2d(eval_conv1, pool_size=2, strides=2, padding='VALID',
    #                                          name='eval_p1_' + str(rand))
    #     # [None, 48, 48, 32] --> [None, 24, 24, 64]
    #     eval_conv2 = tf.layers.conv2d(eval_pool1, kernel_size=3, activation=tf.nn.relu,
    #                                   kernel_initializer=tf.random_normal_initializer(0., 0.01),
    #                                   filters=64, strides=1, padding='SAME', name='eval_c2_' + str(rand))
    #     eval_pool2 = tf.layers.max_pooling2d(eval_conv2, pool_size=2, strides=2, padding='VALID',
    #                                          name='eval_p2_' + str(rand))
    #     # [None, 24, 24, 64] --> [None, 12, 12, 64]
    #     eval_conv3 = tf.layers.conv2d(eval_pool2, kernel_size=3, activation=tf.nn.relu,
    #                                   kernel_initializer=tf.random_normal_initializer(0., 0.01),
    #                                   filters=64, strides=1, padding='SAME', name='eval_c3_' + str(rand))
    #     eval_pool3 = tf.layers.max_pooling2d(eval_conv3, pool_size=2, strides=2, padding='VALID',
    #                                          name='eval_p3_' + str(rand))
    #
    #     length = eval_pool3.shape[1] * eval_pool3.shape[2] * eval_pool3.shape[3]
    #     eval_pool3_flat = tf.reshape(eval_pool3, [-1, length])
    #     f1 = tf.layers.dense(eval_pool3_flat, 512, tf.nn.relu, kernel_initializer=w_initializer,
    #                          bias_initializer=b_initializer, name='eval_f1' + str(rand))
    #     # f2 = tf.layers.dense(f1, 128, tf.nn.relu, kernel_initializer=w_initializer,
    #     #                      bias_initializer=b_initializer, name='f2' + str(rand))
    #     q_eval_net_out = tf.layers.dense(f1, 2 ** hp.N_ACTIONS, kernel_initializer=w_initializer,
    #                                      bias_initializer=b_initializer, name='q_e_' + str(rand))
    #
    # with tf.variable_scope('loss'):
    #     loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_net_out, name='TD_error_' + str(rand)))
    #
    # with tf.variable_scope('train'):
    #     _train_op = tf.train.RMSPropOptimizer(hp.LEARNING_RATE).minimize(loss)
    #
    # # ------------------ build target_net --------------------
    # with tf.variable_scope('target_net'):
    #     # [None, 96, 96, 4] --> [None, 48, 48, 32]
    #     target_conv1 = tf.layers.conv2d(target_net_input, kernel_size=3, activation=tf.nn.relu,
    #                                     kernel_initializer=tf.random_normal_initializer(0., 0.01),
    #                                     filters=32, strides=1, padding='SAME', name='target_c1_' + str(rand))
    #     target_pool1 = tf.layers.max_pooling2d(target_conv1, pool_size=2, strides=2, padding='VALID',
    #                                            name='target_p1_' + str(rand))
    #     # [None, 48, 48, 32] --> [None, 24, 24, 64]
    #     target_conv2 = tf.layers.conv2d(target_pool1, kernel_size=3, activation=tf.nn.relu,
    #                                     kernel_initializer=tf.random_normal_initializer(0., 0.01),
    #                                     filters=64, strides=1, padding='SAME', name='target_c2_' + str(rand))
    #     target_pool2 = tf.layers.max_pooling2d(target_conv2, pool_size=2, strides=2, padding='VALID',
    #                                            name='target_p2_' + str(rand))
    #     # [None, 24, 24, 64] --> [None, 12, 12, 64]
    #     target_conv3 = tf.layers.conv2d(target_pool2, kernel_size=3, activation=tf.nn.relu,
    #                                     kernel_initializer=tf.random_normal_initializer(0., 0.01),
    #                                     filters=64, strides=1, padding='SAME', name='target_c3_' + str(rand))
    #     target_pool3 = tf.layers.max_pooling2d(target_conv3, pool_size=2, strides=2, padding='VALID',
    #                                            name='target_p3_' + str(rand))
    #
    #     length = target_pool3.shape[1] * target_pool3.shape[2] * target_pool3.shape[3]
    #     target_pool3_flat = tf.reshape(target_pool3, [-1, length])
    #     f1 = tf.layers.dense(target_pool3_flat, 512, tf.nn.relu, kernel_initializer=w_initializer,
    #                          bias_initializer=b_initializer, name='target_f1' + str(rand))
    #     # f2 = tf.layers.dense(f1, 128, tf.nn.relu, kernel_initializer=w_initializer,
    #     #                      bias_initializer=b_initializer, name='f2' + str(rand))
    #     q_target_net_out = tf.layers.dense(f1, 2 ** hp.N_ACTIONS, kernel_initializer=w_initializer,
    #                                        bias_initializer=b_initializer, name='t_e_' + str(rand))
    #
    # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_' + str(rand))
    # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_' + str(rand))
    #
    # return [[eval_net_input, target_net_input, q_target], [q_eval_net_out, loss, _train_op, q_target_net_out],
    #         [e_params, t_params]]


if __name__ == '__main__':
    build_network(2)