 100 Days Challenge - Day 7.
Anomaly Detection (Credit Card Fraud Project)

Dec 9

Completed the Credit Card Fraud Detection Project using Anomaly Detection algorithm.

Project: Found out features which have the highest covariances with the fraud/not fraud column among a total of 30 columns. Trained the model using normal probability distribution assumptions on 60% of rows classified as non-fraud, evaluated the algorithm and tuned the decision boundary parameter for the probability (epsilon) based on highest F1-score on the cross validation set (20% of rows classified as non-fraud and 50% of rows classified as fraud), used this model to predict on the test (remaining 20% of rows classified as non-fraud and 50% of rows classified as fraud)

Results: Out of 57109 transactions (99.57% non-fraud and 0.43% fraud) in the test set, 246  were fraud transactions and the algorithm detected 124 of them along with 41 false positives.

Accuracy is over 99.71% (but it's not a good measure of performance for skewed data like in this case. Note that even predicting all non-fraud will give an accuracy of 99.57%)

Jupyter Notebook file: https://github.com/hithesh111/Hith100/blob/master/creditcardfraud.ipynb


Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud




