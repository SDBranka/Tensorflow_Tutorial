# Clustering
# first unsupervised learning method attempt
# Clustering is a Machine Learning technique that involves the grouping of data points
# Clustering is used when you have a bunch of input information, but no labels or output information
# what clustering essentially does is find clusters of like data points and give you the location

# Basic Algorithm for K-Means.
# Step 1: Randomly pick K points to place K centroids
# Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
# Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
# Step 4: Reassign every point once again to the closest centroid.
# Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.

# a centroid (K) is where our current cluster is currently defined
