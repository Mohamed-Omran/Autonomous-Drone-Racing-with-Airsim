from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd


def plot_imitation_error_progress(epochs, errors, Type='Training'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, errors, color='tab:blue')
    ax.set_title('bc ' + Type + ' Error')
    plt.show()


df = pd.read_csv('losses.csv', delimiter=',').T
data = [list(row) for row in df.values]
plot_imitation_error_progress(data[0], data[1], Type='Training')
plot_imitation_error_progress(data[0], data[2], Type='Test')
