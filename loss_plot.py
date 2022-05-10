import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rawData = pd.read_csv('log_files/densenet.log', sep=',')

    epoch = rawData['step'].to_numpy()
    trainingLoss = rawData['train_loss'].to_numpy()

    plt.plot(epoch, trainingLoss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.show()