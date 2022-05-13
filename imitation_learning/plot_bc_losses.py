from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd


def plot_imitation_error_progress(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data[0], data[1], color='tab:blue')
    ax.plot(data[0], data[2], color='tab:red')
    ax.set_title('BC train vs test loss')
    plt.legend(["train losses", "test losses"])
    plt.xlabel("epoch")
    plt.ylabel("bc_loss")
    plt.savefig("../figs/bc_losses.png")

    plt.show()


df = pd.read_csv('losses.csv', delimiter=',').T
data = [list(row) for row in df.values]

plot_imitation_error_progress(data)

