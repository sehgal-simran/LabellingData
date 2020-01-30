from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import math

def computeDistance(a, b):
    dist = 0
    for i in range(len(a)):
        dist+=  (a[i] - b[i])*(a[i] - b[i])
    return math.sqrt(dist)

no_of_clusters = 20
no_of_sectors = 3   #no of sectors in each cluster
test_selection = 15 # no of datapoints to select in innermost sector
decay = math.floor(test_selection/(no_of_sectors-1)) # decay of the points selected in successive sectors of each cluster
queried_points = 0

labelled_data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
unlabelled_data = labelled_data[:,:-1]

size_of_data = unlabelled_data.shape[0]
labels = labelled_data[:,-1:].reshape(size_of_data).astype('int64')
no_of_labels = len(np.unique(labels))
pred_label = [None for _ in range(size_of_data)] # List to store the predicted labels
# Here we run K means on the unlabelled data.
kmeans = KMeans(n_clusters = no_of_clusters).fit(unlabelled_data)
clus_labels = kmeans.labels_
centers = kmeans.cluster_centers_

clusters = [[] for i in range(no_of_clusters)] # 2D List to store the clusters

# Sorting the data into clusters
for i in range(size_of_data):
    clusters[clus_labels[i]].append(i)

for i in range(no_of_clusters):
    section = [[] for i in range(no_of_sectors)] # 2D List to store sectors
    sec_sel = test_selection  # resetting the section selector variable
    testpoints = [] # List to store the test points to determine cluster labels
    dist = [] # List to store tuples of (distance, index), to sort points into sectors

    # Here we compute the distance of every point in the cluster from the
    # cluster centre and sort them on the distance.
    for j in clusters[i]:
        dist.append((computeDistance(centers[i], unlabelled_data[j]), j))
    dist.sort()

    radius=dist[-1][0]-dist[0][0] # Difference between the closest and farthest
    # We sort each point into a sector, based on how far it is from the centre.
    for x in range(len(dist)):
        sec = math.floor(no_of_sectors*(dist[x][0]-dist[0][0])/radius)
        if sec == no_of_sectors:
            sec -= 1
        section[sec].append(dist[x][1])
    # Now, from each sector we select some random points for labelling.
    # Points closer to the center are chosen preferentially.
    for x in range(no_of_sectors):
        if(len(section[x])>=sec_sel):
            testpoints += list(np.random.choice(section[x], sec_sel, replace=False))
            sec_sel -= decay
    hist = [0 for _ in range(no_of_labels)] # Histogram for labels.
    for j in testpoints:
        hist[labels[j]] += 1
    predicted_label = np.argmax(hist)
    # Assigning the most common label to the entire group.
    for j in clusters[i]:
        pred_label[j] = predicted_label
    queried_points += len(testpoints)
# Calculate Accuracy
classif_rate = np.mean(np.array(pred_label).ravel() == np.array(labels).ravel()) * 100
print("Accuracy: ", classif_rate, "% No of points queried ", queried_points)
