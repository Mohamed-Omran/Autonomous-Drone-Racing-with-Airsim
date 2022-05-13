from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd


def plot_cmvae_error_progress(epochs, errors, Type='Training'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, errors[0], color='tab:blue')
    ax.plot(epochs, errors[1], color='tab:orange')
    ax.plot(epochs, errors[2], color='tab:red')
    ax.set_title('cmvae ' + Type + ' losses')
    plt.legend(["Img recon", "Gate recon", "kl"])

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    zipped = zip(errors[0], errors[1], errors[2])
    Sum = [x + y + z for (x, y, z) in zipped]
    ax.plot(epochs, Sum, color='tab:blue')
    ax.set_title('CMVAE Total ' + Type + ' loss')
    plt.show()


df = pd.read_csv('training_cmvae_losses.csv', delimiter=',').T
# User list comprehension to create a list of lists from Dataframe rows
data = [list(row) for row in df.values]
plot_cmvae_error_progress(data[0], data[1:4])
plot_cmvae_error_progress(data[0], data[4:7], Type="Testing")
