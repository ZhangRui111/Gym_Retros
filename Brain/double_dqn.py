import numpy as np
import random
import tensorflow as tf
from collections import deque

from Hyper_parameters.hp_double_dqn import Hyperparameters
from Utils.dtype_convert import binary_array_to_int


class DeepQNetwork:
    def __init__(self, network_build):
        self.hp = Hyperparameters()
        self.n_actions = self.hp.N_ACTIONS
        self.n_features = self.hp.N_FEATURES
        self.replay_start = self.hp.REPLY_START_SIZE
        self.lr = self.hp.LEARNING_RATE
        self.gamma = self.hp.DISCOUNT_FACTOR
        self.epsilon_max = self.hp.FINAL_EXPLOR
        self.replace_target_iter = self.hp.TARGET_NETWORK_UPDATE_FREQUENCY
        self.memory_size = self.hp.REPLY_MEMORY_SIZE
        self.batch_size = self.hp.MINIBATCH_SIZE
        self.epsilon_increment = (self.hp.FINAL_EXPLOR - self.hp.INITIAL_EXPLOR) / self.hp.FINAL_EXPLOR_FRAME
        self.epsilon = self.hp.INITIAL_EXPLOR
        self.flag = True  # output signal
        self.summary_flag = self.hp.OUTPUT_GRAPH  # tf.summary flag

        # network input/output
        self.eval_net_input = network_build[0][0]
        self.target_net_input = network_build[0][1]
        self.q_target = network_build[0][2]
        self.q_eval_net_out = network_build[1][0]
        self.loss = network_build[1][1]
        self.train_op = network_build[1][2]
        self.q_target_net_out = network_build[1][3]
        self.e_params = network_build[2][0]
        self.t_params = network_build[2][1]

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = deque()

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        # start a session
        self.sess = tf.Session(config=tf.ConfigProto(
            device_count={"CPU": 12},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))

        if self.summary_flag:
            self.writer = tf.summary.FileWriter("./logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def choose_action(self, observation):
        """ Choose action following epsilon-greedy policy.

        :param observation:
        :return:
        """
        # at the very beginning, only take actions randomly.
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
            action_index = np.argmax(actions_value)
            a = []
            [a.append(int(x)) for x in bin(action_index)[2:]]
            action = np.array(a)
        else:
            action = np.random.randint(2, size=8)
        return action

    def choose_action_greedy(self, observation):
        """ Choose action following greedy policy.

        :param observation:
        :return:
        """
        actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
        action_index = np.argmax(actions_value)
        a = []
        [a.append(int(x)) for x in bin(action_index)[2:]]
        action = np.array(a)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = random.sample(self.memory, self.batch_size)
        length = self.hp.IMAGE_LENGTH * self.hp.IMAGE_LENGTH * self.hp.AGENT_HISTORY_LENGTH
        batch_memory_arr = np.zeros((self.batch_size, length * 2 + 1 + self.n_actions))
        i = 0
        for d in batch_memory:
            batch_memory_arr[i, :] = d
            i += 1

        s_batch = batch_memory_arr[:, :length]
        eval_act_index_batch = batch_memory_arr[:, length:length + self.n_actions].astype(int)
        # convert the binary array to an int.
        eval_act_index_batch = binary_array_to_int(eval_act_index_batch, self.batch_size)
        reward_batch = batch_memory_arr[:, length + self.n_actions].astype(int)
        s_next_batch = batch_memory_arr[:, -length:]

        s_batch_input = np.reshape(s_batch, (self.hp.MINIBATCH_SIZE, self.hp.IMAGE_LENGTH,
                                             self.hp.IMAGE_LENGTH, self.hp.AGENT_HISTORY_LENGTH))
        s_next_batch_input = np.reshape(s_next_batch, (self.hp.MINIBATCH_SIZE, self.hp.IMAGE_LENGTH,
                                                       self.hp.IMAGE_LENGTH, self.hp.AGENT_HISTORY_LENGTH))

        q_target_select_a, q_target_out = \
            self.sess.run([self.q_eval_net_out, self.q_target_net_out],
                          feed_dict={self.eval_net_input: s_next_batch_input,
                                     self.target_net_input: s_next_batch_input})
        # real q_eval, input is the current observation
        q_eval = self.sess.run(self.q_eval_net_out,
                               {self.eval_net_input: s_batch_input})
        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # # Double DQN
        max_act4next = np.argmax(q_target_select_a, axis=1)
        selected_q_next = q_target_out[batch_index, max_act4next]

        # real q_target
        q_target[batch_index, eval_act_index_batch] = reward_batch + self.gamma * selected_q_next

        if self.summary_flag:
            tf.summary.histogram("q_target", q_target)

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.eval_net_input: s_batch_input,
                                                self.q_target: q_target})

        self.cost_his.append(self.cost)

        # epsilon-decay
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        if self.summary_flag:
            tf.summary.scalar("cost", self.cost)

        if self.summary_flag:
            # merge_all() must follow all tf.summary
            if self.flag:
                self.merge_op = tf.summary.merge_all()
                self.flag = False
        if self.summary_flag:
            merge_all = self.sess.run(self.merge_op, feed_dict={self.eval_net_input: s_batch_input,
                                                                self.q_target: q_target})
            self.writer.add_summary(merge_all, self.learn_step_counter)
