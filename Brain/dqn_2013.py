import numpy as np
import tensorflow as tf


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            eval_net_input,
            q_target,
            q_eval_net_out,
            loss,
            train_op,
            replay_start_size=1000,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.replay_start = replay_start_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.flag = True  # output signal
        self.summary_flag = output_graph  # tf.summary flag

        # network input/output
        self.eval_net_input = eval_net_input
        self.q_target = q_target
        self.q_eval_net_out = q_eval_net_out
        self.loss = loss
        self.train_op = train_op

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # start a session
        self.sess = tf.Session()

        if self.summary_flag:
            self.writer = tf.summary.FileWriter("./logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s.flatten(), [a, r], s_.flatten()))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, step):
        """ Choose action following epsilon-greedy policy.

        :param observation:
        :param step:
        :return:
        """
        # at the very beginning, only take actions randomly.
        if step >= self.replay_start and np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval_net_out,
                                          feed_dict={self.eval_net_input: observation.reshape([1, 4])})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action_greedy(self, observation):
        """ Choose action following greedy policy.

        :param observation:
        :return:
        """
        actions_value = self.sess.run(self.q_eval_net_out,
                                      feed_dict={self.eval_net_input: observation.reshape([1, 4])})
        action = np.argmax(actions_value)
        return action

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # input is all next observation
        q_target_out = self.sess.run(self.q_eval_net_out,
                                     feed_dict={self.eval_net_input: batch_memory[:, -self.n_features:]})
        # real q_eval, input is the current observation
        q_eval = self.sess.run(self.q_eval_net_out,
                               feed_dict={self.eval_net_input: batch_memory[:, :self.n_features]})
        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval)

        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # # DQN
        selected_q_next = np.max(q_target_out, axis=1)

        # real q_target
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        if self.summary_flag:
            tf.summary.histogram("q_target", q_target)

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        self.cost_his.append(self.cost)
        if self.summary_flag:
            tf.summary.scalar("cost", self.cost)

        if self.summary_flag:
            # merge_all() must follow all tf.summary
            if self.flag:
                self.merge_op = tf.summary.merge_all()
                self.flag = False
        if self.summary_flag:
            merge_all = self.sess.run(self.merge_op, feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target})
            self.writer.add_summary(merge_all, self.learn_step_counter)
        # epsilon-decay
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
