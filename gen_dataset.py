## Generate datasets for training and testing

import numpy as np
import pandas as pd
import json

if __name__ == "__main__":
    ## Training Part

    rawData = pd.read_csv('datasets/traindata.tsv', sep='\t')

    #features
    danceability = rawData['danceability'].to_numpy()
    energy = rawData['energy'].to_numpy()
    key = rawData['key'].to_numpy()
    loudness = rawData['loudness'].to_numpy()
    mode = rawData['mode'].to_numpy()
    speechiness = rawData['speechiness'].to_numpy()
    acousticness = rawData['acousticness'].to_numpy()
    instrumentalness = rawData['instrumentalness'].to_numpy()
    liveness = rawData['liveness'].to_numpy()
    valence = rawData['valence'].to_numpy()
    tempo = rawData['tempo'].to_numpy()
    duration_ms = rawData['duration_ms'].to_numpy()
    time_signature = rawData['time_signature'].to_numpy()

    features = np.stack((danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature), axis=1)

    #labels
    playlistNumbers = np.ndarray((features.shape[0],), dtype=np.uint8)
    playlist_pid = rawData['playlist_pid'].to_numpy()
    for i in range(playlist_pid.size):
        tempList = playlist_pid[i].split('_')
        playlistNumbers[i] = int(tempList[1])

    DEC = rawData['dance_energy_corr'].to_numpy()
    uniqueDEC = rawData['dance_energy_corr'].unique()
    np.save('unique_decs', uniqueDEC)

    #save features and labels
    np.save('datasets/traindata', features)
    #np.save('datasets/trainlabels', playlistNumbers)
    np.save('datasets/trainlabels', DEC)

    #create dictionary
    uniquePlaylists = rawData['playlist_pid'].unique()
    playlistDict = {}
    for i in range(100):
        playlistDict[i] = uniquePlaylists[i]

    #save dictionary
    '''
    json = json.dumps(playlistDict)
    f = open("playlistDict.json","w")
    f.write(json)
    f.close()
    '''



    ## Development/Testing Part

    rawData = pd.read_csv('datasets/devdata1.tsv', sep='\t')

    #features
    danceability = rawData['danceability'].to_numpy()
    energy = rawData['energy'].to_numpy()
    key = rawData['key'].to_numpy()
    loudness = rawData['loudness'].to_numpy()
    mode = rawData['mode'].to_numpy()
    speechiness = rawData['speechiness'].to_numpy()
    acousticness = rawData['acousticness'].to_numpy()
    instrumentalness = rawData['instrumentalness'].to_numpy()
    liveness = rawData['liveness'].to_numpy()
    valence = rawData['valence'].to_numpy()
    tempo = rawData['tempo'].to_numpy()
    duration_ms = rawData['duration_ms'].to_numpy()
    time_signature = rawData['time_signature'].to_numpy()

    features = np.stack((danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature), axis=1)

    #labels
    playlistNumbers = np.ndarray((features.shape[0],), dtype=np.uint8)
    playlist_pid = rawData['playlist_pid'].to_numpy()
    for i in range(playlist_pid.size):
        tempList = playlist_pid[i].split('_')
        playlistNumbers[i] = int(tempList[1])

    DEC = rawData['dance_energy_corr'].to_numpy()

    #save features and labels
    np.save('datasets/devdata', features)
    #np.save('datasets/devlabels', playlistNumbers)
    np.save('datasets/devlabels', DEC)



    ## Generate test dataset

    #df = pd.DataFrame()
    indexList = []
    rawData = pd.read_csv('datasets/traindata.tsv', sep='\t')
    #rawData = rawData.reset_index()
    #for index, row in rawData.iterrows():
    #    if index % 100 == 0:
    #        df.append(row)
    for i in range(100):
        randInt = np.random.randint(6100)
        indexList.append(randInt)
    #print(indexList)
    df = rawData.iloc[indexList,:]
    #df = df.reset_index()
    #print(df)
    df.to_csv('datasets/testdata.tsv', sep="\t")

    #features
    danceability = df['danceability'].to_numpy()
    energy = df['energy'].to_numpy()
    key = df['key'].to_numpy()
    loudness = df['loudness'].to_numpy()
    mode = df['mode'].to_numpy()
    speechiness = df['speechiness'].to_numpy()
    acousticness = df['acousticness'].to_numpy()
    instrumentalness = df['instrumentalness'].to_numpy()
    liveness = df['liveness'].to_numpy()
    valence = df['valence'].to_numpy()
    tempo = df['tempo'].to_numpy()
    duration_ms = df['duration_ms'].to_numpy()
    time_signature = df['time_signature'].to_numpy()

    features = np.stack((danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature), axis=1)

    #labels
    playlistNumbers = np.ndarray((features.shape[0],), dtype=np.uint8)
    playlist_pid = df['playlist_pid'].to_numpy()
    for i in range(playlist_pid.size):
        tempList = playlist_pid[i].split('_')
        playlistNumbers[i] = int(tempList[1])

    DEC = df['dance_energy_corr'].to_numpy()

    #save features and labels
    np.save('datasets/testdata', features)
    #np.save('datasets/testlabels', playlistNumbers)
    np.save('datasets/testlabels', DEC)
