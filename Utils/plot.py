"""
Plot training or testing results using matplotlib.pyplot
"""
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt


def plot_results(his_natural, his_prio):
    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='normal_structure')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='rv_structure')
    plt.legend(loc='best')
    plt.ylabel('total training scores')
    plt.xlabel('episode')
    plt.grid()
    plt.savefig('./result.png')