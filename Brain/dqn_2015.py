import numpy as np
import random
import tensorflow as tf
from collections import deque

from Hyper_parameters.hp_dqn_2015 import Hyperparameters


class DeepQNetwork:
    def __init__(
            self,
            network_build
    ):
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
        self.epsilon_increment = (self.hp.FINAL_EXPLOR-self.hp.INITIAL_EXPLOR) / self.hp.FINAL_EXPLOR_FRAME
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

        # consist of [target_net, evaluate_net]
        # build_network()

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        # start a session
        self.sess = tf.Session()

        if self.summary_flag:
            self.writer = tf.summary.FileWriter("./logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def choose_action(self, observation):
        """ Choose action following epsilon-greedy policy.

        :param observation:
        :param step:
        :return:
        """
        # at the very beginning, only take actions randomly.
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
            for i in range(actions_value.size):
                if actions_value[0][i] < 0.5:
                    actions_value[0][i] = 0
                else:
                    actions_value[0][i] = 1
            action = actions_value[0]
        else:
            action = np.random.randint(2, size=8)
        return action

    def choose_action_greedy(self, observation):
        """ Choose action following greedy policy.

        :param observation:
        :return:
        """
        actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
        for i in range(actions_value.size):
            if actions_value[0][i] < 0.5:
                actions_value[0][i] = 0
            else:
                actions_value[0][i] = 1
        action = actions_value[0]
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = random.sample(self.memory, self.batch_size)
        length = self.hp.IMAGE_LENGTH*self.hp.IMAGE_LENGTH*self.hp.AGENT_HISTORY_LENGTH
        batch_memory_arr = np.zeros((self.batch_size, length*2+1+self.n_actions))
        i = 0
        for d in batch_memory:
            batch_memory_arr[i, :] = d
            i += 1

        s_batch = batch_memory_arr[:, :length]
        eval_act_index_batch = batch_memory_arr[:, length:length+self.n_actions].astype(int)
        reward_batch = batch_memory_arr[:, length+self.n_actions].astype(int)
        s_next_batch = batch_memory_arr[:, -length:]

        s_batch_input = np.reshape(s_batch, (self.hp.MINIBATCH_SIZE, self.hp.IMAGE_LENGTH,
                                             self.hp.IMAGE_LENGTH, self.hp.AGENT_HISTORY_LENGTH))
        s_next_batch_input = np.reshape(s_next_batch, (self.hp.MINIBATCH_SIZE, self.hp.IMAGE_LENGTH,
                                                       self.hp.IMAGE_LENGTH, self.hp.AGENT_HISTORY_LENGTH))

        # input is all next observation
        q_target_out = self.sess.run(self.q_target_net_out,
                                     feed_dict={self.target_net_input: s_next_batch_input})

        # real q_eval, input is the current observation
        q_eval = self.sess.run(self.q_eval_net_out,
                               {self.eval_net_input: s_batch_input})
        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval)

        q_target = np.expand_dims(reward_batch, 1) + self.gamma * q_target_out

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
