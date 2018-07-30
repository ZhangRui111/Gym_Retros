import retro
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time
import tensorflow as tf
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from Games.KungFuMaster_Atari2600.hyperparameters import Hyperparameters
from Utils.write_to_file import write_to_file_w
from Utils.write_to_file import write_to_file_a


def plot_results(his_natural, his_prio):
    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='normal_structure')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='rv_structure')
    plt.legend(loc='best')
    plt.ylabel('total training scores')
    plt.xlabel('episode')
    plt.grid()
    plt.savefig('./result.png')


def restore_parameters(sess, model):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(Hp.SAVED_NETWORK_PATH + model + '/')
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0
    return saver, step


def run_stargunner(env, RL, model, saver, load_step):
    total_steps = 0  # total steps after training begins.
    steps_total = []  # sum of steps until one episode.
    episodes = []  # episode's index.
    steps_episode = []  # steps for every single episode.

    rewards_episode = []

    for i_episode in range(Hp.MAX_EPISODES):
        print('episode:' + str(i_episode))
        observation = env.reset()
        episode_steps = 0

        rewards = 0

        while True:
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation.flatten(), total_steps)
            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            rewards += reward

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > RL.memory_size:
                if total_steps > Hp.REPLY_START_SIZE:
                    if total_steps % Hp.UPDATE_FREQUENCY == 0:
                        RL.learn()

                    if total_steps % Hp.WEIGHTS_SAVER_ITER == 0:
                        saver.save(RL.sess, Hp.SAVED_NETWORK_PATH + model + '/' + '-' + model + '-' +
                                   str(total_steps + load_step))
                        # print('-----save weights-----')

                    if total_steps % Hp.OUTPUT_SAVER_ITER == 0:
                        filename1 = Hp.LOGS_DATA_PATH + model + '/rewards_total.txt'
                        write_to_file_a(filename1, str(np.vstack((episodes, rewards_episode))))
                        # print('-----save outputs-----')

            observation = observation_
            episode_steps += 1
            total_steps += 1

            if done:
                print('episode ' +  str(i_episode) + ' | ' + str(rewards))
                steps_episode.append(episode_steps)
                steps_total.append(total_steps)
                rewards_episode.append(rewards)
                episodes.append(i_episode)
                break

    return np.vstack((episodes, rewards_episode))


def main(model):
    env = retro.make(game='KungFuMaster-Atari2600')
    # env.unwrapped can give us more information.
    env = env.unwrapped

    if model == 'pri_dqn':
        from Brain.pri_dqn import DeepQNetwork
        from Brain.pri_dqn import MemoryParas
        from Games.KungFuMaster_Atari2600.network_pri_dqn import build_network
        m_paras = MemoryParas(Hp.M_EPSILON, Hp.M_ALPHA, Hp.M_BETA, Hp.M_BETA_INCRE, Hp.M_ABS_ERROR_UPPER)
        # build network
        # build network.
        n_actions = env.action_space.n
        n_features = env.observation_space.high.size
        inputs, outputs, weights = build_network(n_features, n_actions, lr=Hp.LEARNING_RATE)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=sum([4, 4]),
            n_features=n_features,
            eval_net_input=inputs[0],
            target_net_input=inputs[1],
            q_target=inputs[2],
            ISWeights=inputs[3],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            q_target_net_out=outputs[3],
            abs_errors=outputs[4],
            e_params=weights[0],
            t_params=weights[1],
            memory_paras=m_paras,
            replace_target_iter=Hp.TARGET_REPLACE_ITER,
            learning_rate=Hp.LEARNING_RATE,
            reward_decay=Hp.REWARD_DECAY,
            e_greedy=Hp.E_GREEDY,
            replay_start_size=Hp.REPLY_START_SIZE,
            memory_size=Hp.MEMORY_SIZE,
            batch_size=Hp.BATCH_SIZE,
            e_greedy_increment=0.00005,
            output_graph=True,
        )
    else:  # pri_dqn_rv
        from Brain.pri_dqn import DeepQNetwork
        from Brain.pri_dqn import MemoryParas
        from Games.KungFuMaster_Atari2600.network_pri_dqn_rv import build_network
        # build network.
        n_actions = [4, 4]
        n_features = env.observation_space.high.size
        m_paras = MemoryParas(Hp.M_EPSILON, Hp.M_ALPHA, Hp.M_BETA, Hp.M_BETA_INCRE, Hp.M_ABS_ERROR_UPPER)
        # build network
        inputs, outputs, weights = build_network(n_features, n_actions, lr=Hp.LEARNING_RATE)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=[4, 4],
            n_features=n_features,
            eval_net_input=inputs[0],
            target_net_input=inputs[1],
            q_target=inputs[2],
            ISWeights=inputs[3],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            q_target_net_out=outputs[3],
            abs_errors=outputs[4],
            e_params=weights[0],
            t_params=weights[1],
            memory_paras=m_paras,
            replace_target_iter=Hp.TARGET_REPLACE_ITER,
            learning_rate=Hp.LEARNING_RATE,
            reward_decay=Hp.REWARD_DECAY,
            e_greedy=Hp.E_GREEDY,
            replay_start_size=Hp.REPLY_START_SIZE,
            memory_size=Hp.MEMORY_SIZE,
            batch_size=Hp.BATCH_SIZE,
            e_greedy_increment=0.00005,
            output_graph=True,
        )

    saver, load_step = restore_parameters(RL.sess, model)

    # Calculate running time
    start_time = time.time()

    results = run_stargunner(env, RL, model, saver, load_step)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    filename = Hp.LOGS_DATA_PATH + model + "/running_time.txt"
    write_to_file_w(filename, str(running_time))

    return results


if __name__ == '__main__':
    Hp = Hyperparameters()
    # # change different models here:
    # pri_dqn, double_dqn...
    # result1 = main(model='pri_dqn')
    result2 = main(model='pri_dqn_rv')
    # filename1 = Hp.LOGS_DATA_PATH + 'pri_dqn' + '/rewards_total.txt'
    # write_to_file_running_steps(filename1, str(result1))
    filename2 = Hp.LOGS_DATA_PATH + 'pri_dqn_rv' + '/rewards_total.txt'
    write_to_file_a(filename2, str(result2))
    #
    # plot_results(result1, result2)
