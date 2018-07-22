""" This Prioritized DQN is based on DQN 2015.
"""
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """ This SumTree code is modified version and the original code is from:
        https://github.com/jaara/AI-blog/blob/master/SumTree.py
        Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, p)  # update tree_frame

        self.data[self.data_pointer] = data  # update data_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """ This SumTree code is modified version and the original code is from:
        https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, para):
        self.tree = SumTree(capacity)
        self._init_paras(para)

    def _init_paras(self, para):
        self.epsilon = para.epsilon  # small amount to avoid zero priority
        self.alpha = para.alpha  # [0~1] convert the importance of TD error to priority
        self.beta = para.beta  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = para.beta_increment_per_sampling
        self.abs_err_upper = para.abs_err_upper  # clipped abs error

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), \
                                     np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # beta_max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            # About ISWeights's calculation,
            # https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class MemoryParas(object):
    def __init__(self, m_epsilon, m_alpha, m_bata, m_beta_incre, m_abs_err_upper):
        self.epsilon = m_epsilon  # small amount to avoid zero priority
        self.alpha = m_alpha  # [0~1] convert the importance of TD error to priority
        self.beta = m_bata  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = m_beta_incre
        self.abs_err_upper = m_abs_err_upper  # clipped abs error


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            eval_net_input,
            target_net_input,
            q_target,
            ISWeights,
            q_eval_net_out,
            loss,
            train_op,
            q_target_net_out,
            abs_errors,
            e_params,
            t_params,
            memory_paras,
            replay_start_size=1000,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.memory_paras = memory_paras
        self.replay_start = replay_start_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # network input/output
        self.eval_net_input = eval_net_input
        self.target_net_input = target_net_input
        self.q_target = q_target
        self.ISWeights = ISWeights
        self.q_eval_net_out = q_eval_net_out
        self.loss = loss
        self.train_op = train_op
        self.q_target_net_out = q_target_net_out
        self.abs_errors = abs_errors
        self.e_params = e_params
        self.t_params = t_params

        self.learn_step_counter = 0

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        self.memory = Memory(capacity=memory_size, para=self.memory_paras)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.flatten(), [a, r], s_.flatten()))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def choose_action(self, observation, step):
        observation = observation[np.newaxis, :]
        if step >= self.replay_start and np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
            actions_value[actions_value > 0.5] = 1
            actions_value[actions_value <= 0.5] = 0
            action = actions_value
        else:
            action = np.random.randint(2, size=8)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        q_next, q_eval = self.sess.run(
            [self.q_target_net_out, self.q_eval_net_out],
            feed_dict={self.target_net_input: batch_memory[:, -self.n_features:],
                       self.eval_net_input: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, abs_errors, self.cost = self.sess.run([self.train_op, self.abs_errors, self.loss],
                                                 feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                            self.q_target: q_target,
                                                            self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
