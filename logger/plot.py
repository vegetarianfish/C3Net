import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def history_ploter(history, path, x_lim=1000, labels=None):
    history = np.asarray(history)
    title = path.split('/')[-1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(history))
    if history.ndim == 1:
        y = history
        plt.xlim(0, x_lim)
        plt.ylim(0, 0.9)
        ax.plot(x[y != None], y[y != None])
    elif np.shape(history)[1] == 2:
        y = history[:, 0]
        ax.plot(x[y != None], y[y != None], label='train')
        y = history[:, 1]
        ax.plot(x[y != None], y[y != None], label='valid')
        ax.legend()
    elif np.shape(history)[1] == 4:
        y = history[:, 0]
        ax.plot(x[y != None], y[y != None], label=labels[0])
        y = history[:, 1]
        ax.plot(x[y != None], y[y != None], label=labels[1])
        y = history[:, 2]
        ax.plot(x[y != None], y[y != None], label=labels[2])
        y = history[:, 3]
        ax.plot(x[y != None], y[y != None], label=labels[3])
        ax.legend()
    ax.set_title(title)
    plt.savefig(str(path))
    plt.close()
