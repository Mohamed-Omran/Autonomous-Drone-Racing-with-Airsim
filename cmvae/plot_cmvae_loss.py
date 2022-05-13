from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd

def plot_cmvae_error_progress(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data[0], data[1], color='tab:blue')
    ax.plot(data[0], data[4], color='tab:red')
    ax.set_title('image reconstruction loss')
    plt.legend(["train losses", "test losses"])
    plt.xlabel("epoch")
    plt.ylabel("img_rec_loss")
    plt.savefig("../figs/cmvae_img_rec_losses.png")

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(data[0], data[2], color='tab:blue')
    ax.plot(data[0], data[5], color='tab:red')
    ax.set_title('gate reconstruction loss')
    plt.legend(["train losses", "test losses"])
    plt.xlabel("epoch")
    plt.ylabel("gate_rec_loss")
    plt.savefig("../figs/cmvae_gate_rec_losses.png")

    fig3 = plt.figure()
    ax = fig3.add_subplot(1, 1, 1)
    ax.plot(data[0], data[3], color='tab:blue')
    ax.plot(data[0], data[6], color='tab:red')
    ax.set_title('K-L loss')
    plt.legend(["train losses", "test losses"])
    plt.xlabel("epoch")
    plt.ylabel("kl_loss")
    plt.savefig("../figs/cmvae_kl_losses.png")

    fig4 = plt.figure()
    ax = fig4.add_subplot(1, 1, 1)
    zipped1 = zip(data[1], data[2], data[3])
    Sum1 = [x + y + z for (x, y, z) in zipped1]
    zipped2 = zip(data[4], data[5], data[6])
    Sum2 = [x + y + z for (x, y, z) in zipped2]
    ax.plot(data[0], Sum1, color='tab:blue')
    ax.plot(data[0], Sum2, color='tab:red')
    ax.set_title('CMVAE Total Loss')
    plt.legend(["train losses", "test losses"])
    plt.xlabel("epoch")
    plt.ylabel("total_loss")
    plt.savefig("../figs/cmvae_total_losses.png")

    plt.show()


df = pd.read_csv('training_cmvae_losses.csv', delimiter=',').T
data = [list(row) for row in df.values]
plot_cmvae_error_progress(data)

