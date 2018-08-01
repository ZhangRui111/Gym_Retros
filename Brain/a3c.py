"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
The Cartpole example.
"""
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt

from Hyper_parameters.hp_a3c import Hyperparameters
from Games.StarGunner_Atari2600.network_a3c import build_network

hp = Hyperparameters()
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


class Shared(object):
    def __init__(self, sess, opt_a, opt_c, coord):
        self.SESS = sess
        self.OPT_A = opt_a
        self.OPT_C = opt_c
        self.COORD = coord


class ACNet(object):
    def __init__(self, scope, shared, globalAC=None):
        self.shared = shared

        if scope == hp.GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s, non, non_, self.a_params, self.c_params = build_network(scope)
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # self.v_target is real v.

                self.s, self.a_prob, self.v, self.a_params, self.c_params = build_network(scope)  # self.v is eval v.

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, hp.N_ACTIONS, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = hp.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.shared.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.shared.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def update_global(self, feed_dict):  # run by a local
        self.shared.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.shared.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.shared.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, shared, globalAC):
        self.env = gym.make(hp.GAME).unwrapped  # every agent has its own copy of the environment.
        self.name = name
        self.AC = ACNet(name, shared, globalAC)
        self.shared = shared

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.shared.COORD.should_stop() and GLOBAL_EP < hp.MAX_GLOBAL_EPISODES:
            s = self.env.reset()
            ep_r = 0
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done:
                    r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % hp.UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.shared.SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + hp.DISCOUNT_FACTOR * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = \
                        np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    hp = Hyperparameters()
    SESS = tf.Session()
    COORD = tf.train.Coordinator()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(hp.LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(hp.LR_C, name='RMSPropC')
        shared = Shared(SESS, OPT_A, OPT_C, COORD)
        GLOBAL_AC = ACNet(hp.GLOBAL_NET_SCOPE, shared)  # we only need its params
        workers = []
        # Create worker
        for i in range(hp.N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, shared, GLOBAL_AC))

    SESS.run(tf.global_variables_initializer())

    if hp.OUTPUT_GRAPH:
        if os.path.exists(hp.LOGS_DATA_PATH):
            shutil.rmtree(hp.LOGS_DATA_PATH)
        tf.summary.FileWriter(hp.LOGS_DATA_PATH, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
