 100 Days Challenge - Day 7.
Anomaly Detection (Credit Card Fraud Project)

Dec 9

Completed the Credit Card Fraud Detection Project using Anomaly Detection algorithm.

Project: Found out features which have the highest covariances with the fraud/not fraud column among a total of 30 columns. Trained the model using normal probability distribution assumptions on 60% of rows classified as non-fraud, evaluated the algorithm and tuned the decision boundary parameter for the probability (epsilon) based on highest F1-score on the cross validation set (20% of rows classified as non-fraud and 50% of rows classified as fraud), used this model to predict on the test (remaining 20% of rows classified as non-fraud and 50% of rows classified as fraud)

Results: Out of 57109 transactions (99.57% non-fraud and 0.43% fraud) in the test set, 246  were fraud transactions and the algorithm detected 124 of them restricting false positives to 41.

In other words, 50.4% of fraud transactions were detected and only 0.07% of non-fraud transactions were flagged wrong.

Jupyter Notebook file: https://github.com/hithesh111/Hith100/blob/master/creditcardfraud.ipynb


Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud




