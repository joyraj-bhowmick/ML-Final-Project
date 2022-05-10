import numpy as np
import pandas as pd

if __name__ == "__main__":
    predData = pd.read_csv('densenet_predictions.csv', header=None)
    
    rawPredictions = predData.iloc[:, 0].to_numpy()
    
    labels = np.load('datasets/testlabels.npy')

    uniqueDEC = np.load('unique_decs.npy')

    truePredictions = np.ndarray(labels.shape)
    tempArr = np.ndarray(uniqueDEC.shape)

    for i in range(truePredictions.shape[0]):
        for j in range(uniqueDEC.shape[0]):
            tempArr[j] = abs(rawPredictions[i]-uniqueDEC[j])
        index = np.argmin(tempArr)
        truePredictions[i] = uniqueDEC[index]

    matches = 0
    for i in range(truePredictions.shape[0]):
        if truePredictions[i] == labels[i]:
            matches += 1
    accuracy = matches/truePredictions.shape[0]
    
    print(f"Correct Matches: {matches}\tAccuracy {accuracy}")


