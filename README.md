Machine Learning (BITS F464)

Group No. 2
Aditya Vasudevan (2017A7PS0175P)
Simran Sehgal (2017A8PS0405P)
Ishani Rajput (2017A8PS0515P)

Title: Assignment 2 (Active Learning)

The code package contains two Python scripts, 'activeLearning.py' and 'cluster.py'.

activeLearning.py contains the code implementing Pool-based and Stream-based learning techniques. cluster.py contains the code for the cluster based labelling strategy.
The code has been commented for readability and explains what is happening at each step.
For the purpose of the assignments, we required some Classifiers and Clustering methods, hence, we have used the Python Library sklearn for these. The implementation for the active learning query selection however, is coded.

How to Run:
Both scripts can be run like normal Python scripts with no command line arguments.
The parameters to be changes have been described in the scripts itself and can be modified from there.

Datasets used for Testing:
We used two datasets for testing purposes. One is the HTRU_2 dataset and the other is a banknote authentication dataset, both from the UCI ML Repository. Both are binary classification problems, however, the code will work for a classification problem with an arbitrary number of classes.

Potential Error while Running:
To begin with, the code chooses a small random set to label as the initial training set.
The problem with this is that it is entirely possible for all data points to belong to the same class. If that happens to be the case, the code will throw a Value Error, the best course of action in that case is to simply run the program again. After a few tries, you will definitely reach a point where the code will run properly.
This problem is likely to occur for multi-class problems,

Cluster-based labelling Approach
First we cluster the data into a large number of clusters (by large, we mean that the number of clusters is greater than the number of expected labels)
In our implementation, we have used the K-Means provided in the sklearn library for Python.
Once we have clustered our data, we select some points to be labelled. Before we choose however, we divide each cluster into sectors on the basis of the distance of points from the centroid value. Now, we choose points to be labelled from this set, with more preference to points that lie closer to the center of the cluster. This also ensures that we get some points that lie away from the cluster center. We then do a majority polling for the label of the cluster based in this selection. And once we have decided the label, we label the entire cluster using this predicted label. Choosing the right number of clusters and the number of sections within a cluster is critical for the accuracy to be high. After some tweaking of parameters, we got an accuracy as high as 98%, on the banknote authentication dataset.
