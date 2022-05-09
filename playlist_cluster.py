import numpy as np
import pandas as pd
import time

rawData = pd.read_csv('dataset/traindata.tsv', sep='\t')
columns = rawData.columns
uniquePlaylists = rawData['playlist_pid'].unique()
uniqueDEC = rawData['dance_energy_corr'].unique()

print(uniquePlaylists.shape, uniqueDEC.shape)
print(columns)
#print(uniqueDEC.shape)

playlistCluster = {}
tempDict = {}
nestedTempDict = {}

for i in uniquePlaylists:
    playlistCluster[i] = {}

## Not able to create this dictionary
rawData = rawData.reset_index()
for index, row in rawData.iterrows():
    nestedTempDict = row.to_dict()
    #print(nestedTempDict, "\n")
    tempDict[row['uri']] = nestedTempDict
    #print(tempDict, "\n")
    #playlistCluster[row['playlist_pid']] = tempDict
    playlistCluster[row['playlist_pid']].append(tempDict)
    #print(playlistCluster, "\n")
    #time.sleep(5)
#print(tempDict)
#print(playlistCluster['Throwbacks_0'])
for key, value in playlistCluster['Throwbacks_0'].items():
    print(key, ':', value)

#print(playlistCluster.keys())