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

    s = tf.placeholder(tf.float32, [None, hp.IMAGE_LENGTH, hp.IMAGE_LENGTH, 1], 'S')

    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0)
    with tf.variable_scope('actor'):
        # [None, 96, 96, 4] --> [None, 48, 48, 32]
        actor_conv1 = tf.layers.conv2d(s, kernel_size=3, activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                       filters=32, strides=1, padding='SAME', name='actor_c1_')
        actor_pool1 = tf.layers.max_pooling2d(actor_conv1, pool_size=2, strides=2, padding='VALID',
                                              name='actor_p1_')
        # [None, 48, 48, 32] --> [None, 24, 24, 64]
        actor_conv2 = tf.layers.conv2d(actor_pool1, kernel_size=3, activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                       filters=64, strides=1, padding='SAME', name='actor_c2_')
        actor_pool2 = tf.layers.max_pooling2d(actor_conv2, pool_size=2, strides=2, padding='VALID',
                                              name='actor_p2_')
        # [None, 24, 24, 64] --> [None, 12, 12, 64]
        actor_conv3 = tf.layers.conv2d(actor_pool2, kernel_size=3, activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                       filters=64, strides=1, padding='SAME', name='actor_c3_')
        actor_pool3 = tf.layers.max_pooling2d(actor_conv3, pool_size=2, strides=2, padding='VALID',
                                              name='actor_p3_')

        length = actor_pool3.shape[1] * actor_pool3.shape[2] * actor_pool3.shape[3]
        actor_pool3_flat = tf.reshape(actor_pool3, [-1, length])
        f1 = tf.layers.dense(actor_pool3_flat, 512, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='actor_f1')
        actor_net_out = tf.layers.dense(f1, 2 ** hp.N_ACTIONS, tf.nn.softmax, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='actor_e_')

        # l_a = tf.layers.dense(s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
        # a_prob = tf.layers.dense(l_a, hp.N_ACTIONS, tf.nn.softmax, kernel_initializer=w_init, name='ap')
    with tf.variable_scope('critic'):
        # [None, 96, 96, 4] --> [None, 48, 48, 32]
        critic_conv1 = tf.layers.conv2d(s, kernel_size=3, activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                        filters=32, strides=1, padding='SAME', name='actor_c1_')
        critic_pool1 = tf.layers.max_pooling2d(critic_conv1, pool_size=2, strides=2, padding='VALID',
                                               name='actor_p1_')
        # [None, 48, 48, 32] --> [None, 24, 24, 64]
        critic_conv2 = tf.layers.conv2d(critic_pool1, kernel_size=3, activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                        filters=64, strides=1, padding='SAME', name='actor_c2_')
        critic_pool2 = tf.layers.max_pooling2d(critic_conv2, pool_size=2, strides=2, padding='VALID',
                                               name='actor_p2_')
        # [None, 24, 24, 64] --> [None, 12, 12, 64]
        critic_conv3 = tf.layers.conv2d(critic_pool2, kernel_size=3, activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.01),
                                        filters=64, strides=1, padding='SAME', name='actor_c3_')
        critic_pool3 = tf.layers.max_pooling2d(critic_conv3, pool_size=2, strides=2, padding='VALID',
                                               name='actor_p3_')

        length = critic_pool3.shape[1] * critic_pool3.shape[2] * critic_pool3.shape[3]
        critic_pool3_flat = tf.reshape(critic_pool3, [-1, length])
        f1 = tf.layers.dense(critic_pool3_flat, 512, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='actor_f1')
        critic_net_out = tf.layers.dense(f1, 2 ** hp.N_ACTIONS, tf.nn.softmax, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='actor_e_')

        # l_c = tf.layers.dense(s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
        # v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

    a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
    c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

    return [s, actor_net_out, critic_net_out, a_params, c_params]


if __name__ == '__main__':
    build_network(2)
