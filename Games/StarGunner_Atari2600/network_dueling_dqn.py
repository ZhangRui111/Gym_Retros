import tensorflow as tf
from Hyper_parameters.hp_dqn_2015 import Hyperparameters


def build_network(rand):
    """ Build the network for RL algorithm.

    :param rand -- used to identify different network.
    :return:
        [[eval_net_input, target_net_input, q_target]: input
        [q_eval_net_out, loss, _train_op, q_target_net_out]: output
        [e_params, t_params]]: weights
    """
    # init Hp
    hp = Hyperparameters()

    # ------------------ all inputs --------------------------
    # input for target net
    eval_net_input = tf.placeholder(tf.float32, [None, hp.IMAGE_LENGTH, hp.IMAGE_LENGTH, hp.AGENT_HISTORY_LENGTH],
                                    name='eval_net_input_' + str(rand))
    # input for eval net
    target_net_input = tf.placeholder(tf.float32, [None, hp.IMAGE_LENGTH, hp.IMAGE_LENGTH, hp.AGENT_HISTORY_LENGTH],
                                      name='target_net_input_' + str(rand))
    # q_target for loss
    q_target = tf.placeholder(tf.float32, [None, hp.N_ACTIONS], name='q_target_' + str(rand))
    # initializer
    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0)

    # ------------------ build evaluate_net ------------------
    with tf.variable_scope('eval_net'):
        # [None, 96, 96, 4] --> [None, 48, 48, 32]
        eval_conv1 = tf.layers.conv2d(eval_net_input, kernel_size=3, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                      filters=32, strides=1, padding='SAME', name='eval_c1_' + str(rand))
        eval_pool1 = tf.layers.max_pooling2d(eval_conv1, pool_size=2, strides=2, padding='VALID',
                                             name='eval_p1_' + str(rand))
        # [None, 48, 48, 32] --> [None, 24, 24, 64]
        eval_conv2 = tf.layers.conv2d(eval_pool1, kernel_size=3, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                      filters=64, strides=1, padding='SAME', name='eval_c2_' + str(rand))
        eval_pool2 = tf.layers.max_pooling2d(eval_conv2, pool_size=2, strides=2, padding='VALID',
                                             name='eval_p2_' + str(rand))
        # [None, 24, 24, 64] --> [None, 12, 12, 64]
        eval_conv3 = tf.layers.conv2d(eval_pool2, kernel_size=3, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                      filters=64, strides=1, padding='SAME', name='eval_c3_' + str(rand))
        eval_pool3 = tf.layers.max_pooling2d(eval_conv3, pool_size=2, strides=2, padding='VALID',
                                             name='eval_p3_' + str(rand))

        length = eval_pool3.shape[1] * eval_pool3.shape[2] * eval_pool3.shape[3]
        eval_pool3_flat = tf.reshape(eval_pool3, [-1, length])
        f1 = tf.layers.dense(eval_pool3_flat, 512, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='eval_f1' + str(rand))
        # f2 = tf.layers.dense(f1, 128, tf.nn.relu, kernel_initializer=w_initializer,
        #                      bias_initializer=b_initializer, name='f2' + str(rand))
        eval_V = tf.layers.dense(f1, 1, None, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='eval_V')
        eval_A = tf.layers.dense(f1, hp.N_ACTIONS, None, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='eval_A')
        q_eval_net_out = eval_V + (eval_A - tf.reduce_mean(eval_A, axis=1, keep_dims=True))

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_net_out, name='TD_error_' + str(rand)))

    with tf.variable_scope('train'):
        _train_op = tf.train.RMSPropOptimizer(hp.LEARNING_RATE).minimize(loss)

    # ------------------ build target_net --------------------
    with tf.variable_scope('target_net'):
        # [None, 96, 96, 4] --> [None, 48, 48, 32]
        target_conv1 = tf.layers.conv2d(target_net_input, kernel_size=3, activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                        filters=32, strides=1, padding='SAME', name='target_c1_' + str(rand))
        target_pool1 = tf.layers.max_pooling2d(target_conv1, pool_size=2, strides=2, padding='VALID',
                                               name='target_p1_' + str(rand))
        # [None, 48, 48, 32] --> [None, 24, 24, 64]
        target_conv2 = tf.layers.conv2d(target_pool1, kernel_size=3, activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                        filters=64, strides=1, padding='SAME', name='target_c2_' + str(rand))
        target_pool2 = tf.layers.max_pooling2d(target_conv2, pool_size=2, strides=2, padding='VALID',
                                               name='target_p2_' + str(rand))
        # [None, 24, 24, 64] --> [None, 12, 12, 64]
        target_conv3 = tf.layers.conv2d(target_pool2, kernel_size=3, activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                        filters=64, strides=1, padding='SAME', name='target_c3_' + str(rand))
        target_pool3 = tf.layers.max_pooling2d(target_conv3, pool_size=2, strides=2, padding='VALID',
                                               name='target_p3_' + str(rand))

        length = target_pool3.shape[1] * target_pool3.shape[2] * target_pool3.shape[3]
        target_pool3_flat = tf.reshape(target_pool3, [-1, length])
        f1 = tf.layers.dense(target_pool3_flat, 512, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='target_f1' + str(rand))
        # f2 = tf.layers.dense(f1, 128, tf.nn.relu, kernel_initializer=w_initializer,
        #                      bias_initializer=b_initializer, name='f2' + str(rand))
        target_V = tf.layers.dense(f1, 1, None, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='target_V')
        target_A = tf.layers.dense(f1, hp.N_ACTIONS, None, kernel_initializer=w_initializer,
                                   bias_initializer=b_initializer, name='target_A')
        q_target_net_out = target_V + (target_A - tf.reduce_mean(target_A, axis=1, keep_dims=True))

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_' + str(rand))
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_' + str(rand))

    return [[eval_net_input, target_net_input, q_target], [q_eval_net_out, loss, _train_op, q_target_net_out],
            [e_params, t_params]]


if __name__ == '__main__':
    build_network(2)
