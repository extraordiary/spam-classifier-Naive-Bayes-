# spam-classifier-Naive-Bayes-
Machine learning-using matlab complish spam classifier
there are 5 braches, each braches means a kind of classifier(except the master)
Please put spamData.mat to the right path before running the code.
1. bi  means that the data in the code is preprocessed by binarization
   log means that the data in the code is preprocessed by log-transform
   Z   means that the data in the code is preprocessed by Z-normalization
2. for Gaussian Naive Bayes classifier,
   if you want to get the error rate of both the data being Z-normalized and log-transformed
   you can run Gaussian_Naive_Bayes_classifier.m
3. for Logistic regression classifier and K-Nearest Neighbors classifier,
   if you want to get plots of all the 3 methods of data preprocessing in a figure and error rates corresponding to particular parameters
   you can run Logistic_regression_classifier.m and K_Nearest_Neighbors_classifier.m
