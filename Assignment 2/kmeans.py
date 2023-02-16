from itertools import filterfalse
from matplotlib import pyplot as plt
import pandas as pd
import random
from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
import numpy as np


def euclidean(list1, list2):
    ed = 0
    for i in range(len(list1)):
        ed += (list1[i] - list2[i])**2
    ed = ed**.5
    return ed


def assign_cluster(list1, list2, diction):
    for index in range(len(list1)):
        distance0 = euclidean(list1[index], list2[0])
        distance1 = euclidean(list1[index], list2[1])
        distance2 = euclidean(list1[index], list2[2])
        cluster_distance = min(distance0, distance1, distance2)
        if cluster_distance == distance0:
            diction['0'].append(index)
        elif cluster_distance == distance1:
            diction['1'].append(index)
        else:
            diction['2'].append(index)


def recluster(list1, diction):
    clusters_points = []
    for key, value in diction.items():
        p0 = 0
        p1 = 0
        p2 = 0
        p3 = 0
        for val in value:
            p0 += list1[val][0]
            p1 += list1[val][1]
            p2 += list1[val][2]
            p3 += list1[val][3]
        p0 /= len(value)
        p1 /= len(value)
        p2 /= len(value)
        p3 /= len(value)
        clusters_points.append([p0, p1, p2, p3])
    return clusters_points


def compare_centroids(list1, list2):
    if list1[0][0] == list2[0][0] and list1[0][1] == list2[0][1] and list1[0][2] == list2[0][2] and list1[1][0] == list2[1][0] and list1[1][1] == list2[1][1] and list1[1][2] == list2[1][2] and list1[2][0] == list2[2][0] and list1[2][1] == list2[2][1] and list1[2][2] == list2[2][2]:
        return True
    return False


k = 3
df = pd.read_csv('K_means_train.csv')
all_rows = []
for index, rows in df.iterrows():
    row_list = [rows.SepalLengthCm, rows.SepalWidthCm, rows.PetalLengthCm, rows.PetalWidthCm]
    all_rows.append(row_list)
#print(all_rows)
"""
cluster_indices = random.sample(range(len(all_rows)), k)
cluster_points = []
for x in range(len(cluster_indices)):
    cluster_points.append(all_rows[cluster_indices[x]])
"""
cluster_points = [[4.9, 2.5, 4.5, 1.7],[5.6, 2.5, 3.9, 1.1],[6.3, 2.9, 5.6, 1.8]]
print(cluster_points)


clusters = {'0': [], '1': [], '2': []}
assign_cluster(all_rows, cluster_points, clusters)
#print(clusters)
#print(clusters)
centroids = recluster(all_rows, clusters)
#print(centroids)
old_centroids = cluster_points

while (compare_centroids(old_centroids, centroids) == False):
    clusters = {'0': [], '1': [], '2': []}
    #print(old_centroids)
    #print(centroids)
    #print('not clustered yet')
    assign_cluster(all_rows, centroids, clusters)

    old_centroids = centroids
    centroids = recluster(all_rows, clusters)
    #print(clusters)
    #print(old_centroids)
    #print(centroids)
    #print('reclustered')

#print("final")
print(centroids)
print(clusters)
for key, val in clusters.items():
    df.loc[val, "Labels"] = int(key)
del df['Id']
print(df)

tsne = TSNE(random_state=0)
tsne_results = tsne.fit_transform(df)
tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
plt.figure(figsize=(8,5))
plot = plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=df.Labels)
handles, _ = plot.legend_elements(prop='colors')
plt.legend(handles, ['Cluster 1', 'Cluster 2', 'Cluster 3'], loc="upper center")
plt.show()

df2 = pd.read_csv('K_means_test.csv')
all_valid_rows = []
for index, rows in df2.iterrows():
    valid_row_list = [rows.SepalLengthCm, rows.SepalWidthCm, rows.PetalLengthCm, rows.PetalWidthCm]
    all_valid_rows.append(valid_row_list)

cluster_points = centroids
test_clusters = {'0': [], '1': [], '2': []}
assign_cluster(all_valid_rows, cluster_points, test_clusters)
for key, val in test_clusters.items():
    df2.loc[val, "labels"] = int(key)
del df2['Id']
print(df2)
compression_options = dict(method='zip', archive_name='K_Means_Test.csv')
df2.to_csv('K_Means_Test.zip', compression=compression_options)
