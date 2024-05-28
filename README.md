# K_Means-MNIST
Machine Learning Model for the hand-written digit classification.
1.	The algorithm chosen is k-means clustering. It is a type of unsupervised learning that does not require training. Firstly, we arbitrarily choose k means and assign each object to the most similar cluster center. Then after we update the cluster means and re-assign the objects. We repeat this step until the cluster center no longer changes. The input for k-means is the x-values of the dataset, which are pixels  0 to 783. We run the input through the vector assembler to create the features vector column. We then fit the transformed data with the K-means algorithm to get the K-means model. Then we can get our predictions with model.transform(data). The output for predictions is which cluster the algorithm assigns each object to out of the k clusters. 

2.	With k=10, 
Training Accuracy: 0.5021
Test Accuracy: 0.5051

3.	With k-100,
Training Accuracy = 0.8727
Test Accuracy = 0.8877
