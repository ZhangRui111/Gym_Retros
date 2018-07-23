import retro
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import tensorflow as tf
import matplotlib as mlp
mlp.use('TkAgg')

from Games.KungFuMaster_Atari2600.hyperparameters import Hyperparameters
from Utils.write_to_file import write_to_file_running_steps


def run_stargunner(env, RL, update=2):
    total_steps = 0  # total steps after training begins.
    steps_total = []  # sum of steps until one episode.
    episodes = []  # episode's index.
    steps_episode = []  # steps for every single episode.

    rewards_episode = []

    for i_episode in range(2):
        print('episode:' + str(i_episode))
        observation = env.reset()
        episode_steps = 0

        rewards = 0

        while True:
            # env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation.flatten(), total_steps)
            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            rewards += reward

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > RL.memory_size:
                if total_steps > Hp.REPLY_START_SIZE:
                    if total_steps % update == 0:
                        RL.learn()

                    # if total_steps % Hp.WEIGHTS_SAVER_ITER == 0:
                    #     saver.save(RL.sess, Hp.SAVED_NETWORK_PATH + model + '/' + '-' + model + '-' +
                    #                str(total_steps + load_step))

            observation = observation_
            episode_steps += 1
            total_steps += 1

            if done:
                print('episode ', i_episode, ' finished')
                steps_episode.append(episode_steps)
                steps_total.append(total_steps)
                rewards_episode.append(rewards)
                episodes.append(i_episode)
                break

    return sum(rewards_episode)


def main(model, flag, para):
    env = retro.make(game='KungFuMaster-Atari2600')
    # env.unwrapped can give us more information.
    env = env.unwrapped

    if model == 'pri_dqn':
        from Brain.pri_dqn import DeepQNetwork
        from Brain.pri_dqn import MemoryParas
        from Games.KungFuMaster_Atari2600.test_network_pri_dqn import build_network
        m_paras = MemoryParas(Hp.M_EPSILON, Hp.M_ALPHA, Hp.M_BETA, Hp.M_BETA_INCRE, Hp.M_ABS_ERROR_UPPER)
        # build network
        # build network.
        n_actions = env.action_space.n
        n_features = env.observation_space.high.size
        rand = np.random.randint(1, 1000)
        inputs, outputs, weights = build_network(n_features, n_actions, 0.01, rand)
        # get the DeepQNetwork Agent
        if flag == 0:
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
        elif flag == 1:
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
                learning_rate=para,
                reward_decay=Hp.REWARD_DECAY,
                e_greedy=Hp.E_GREEDY,
                replay_start_size=Hp.REPLY_START_SIZE,
                memory_size=Hp.MEMORY_SIZE,
                batch_size=Hp.BATCH_SIZE,
                e_greedy_increment=0.00005,
                output_graph=True,
            )
        elif flag == 2:
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
                reward_decay=para,
                e_greedy=Hp.E_GREEDY,
                replay_start_size=Hp.REPLY_START_SIZE,
                memory_size=Hp.MEMORY_SIZE,
                batch_size=Hp.BATCH_SIZE,
                e_greedy_increment=0.00005,
                output_graph=True,
            )
        else:
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
                replace_target_iter=para,
                learning_rate=Hp.LEARNING_RATE,
                reward_decay=Hp.REWARD_DECAY,
                e_greedy=Hp.E_GREEDY,
                replay_start_size=Hp.REPLY_START_SIZE,
                memory_size=Hp.MEMORY_SIZE,
                batch_size=Hp.BATCH_SIZE,
                e_greedy_increment=0.00005,
                output_graph=True,
            )

        if flag == 0:
            results = run_stargunner(env, RL, update=para)
        else:
            results = run_stargunner(env, RL)

        return results
    else:
        print('Wrong model!')


if __name__ == '__main__':
    Hp = Hyperparameters()
    # # hyper-parameter list
    update_frequency = [1, 2, 4, 5]
    learning_rate = [0.0005, 0.005, 0.001, 0.05, 0.01, 0.1, 0.5]
    reward_decay = [0.5, 0.75, 0.9, 0.99]
    target_replace_iter = [500, 1000, 2000, 4000]
    for i in range(4):
        if i == 0:
            total_rewards = []
            for j in range(4):
                rewards = main(model='pri_dqn', flag=0, para=update_frequency[j])
                total_rewards.append(rewards)
            filename1 = Hp.LOGS_DATA_PATH + 'pri_dqn' + '/rewards_total_0.txt'
            write_to_file_running_steps(filename1, str(total_rewards))
        elif i == 1:
            total_rewards = []
            for j in range(7):
                rewards = main(model='pri_dqn', flag=1, para=learning_rate[j])
                total_rewards.append(rewards)
            filename1 = Hp.LOGS_DATA_PATH + 'pri_dqn' + '/rewards_total_1.txt'
            write_to_file_running_steps(filename1, str(total_rewards))
        elif i == 2:
            total_rewards = []
            for j in range(4):
                rewards = main(model='pri_dqn', flag=2, para=reward_decay[j])
                total_rewards.append(rewards)
            filename1 = Hp.LOGS_DATA_PATH + 'pri_dqn' + '/rewards_total_2.txt'
            write_to_file_running_steps(filename1, str(total_rewards))
        else:
            total_rewards = []
            for j in range(4):
                rewards = main(model='pri_dqn', flag=3, para=target_replace_iter[j])
                total_rewards.append(rewards)
            filename1 = Hp.LOGS_DATA_PATH + 'pri_dqn' + '/rewards_total_3.txt'
            write_to_file_running_steps(filename1, str(total_rewards))