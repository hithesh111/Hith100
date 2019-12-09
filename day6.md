 100 Days Challenge - Day 6.
Anomaly Detection (Credit Card Fraud Project)

Dec 8


Revised Anomaly Detection  from Andrew Ng's Machine Learning course on Coursera which I had already completed a few months ago and started working on Credit Card Transaction Dataset to detect fraudulent transactions.



Topics covered include:


Anomaly Detection

    Normal Assumption
    Algorithm
    Evaluation
    Choosing Features
    Anomaly Detection vs Supervised Learning
    Anomaly Detection using Multivariate Normal

Project: Found out features which have the highest covariances with the fraud/not fraud column among a total of 30 columns.
Then ran anomaly detection algorithm on training data (60%) using product of normal distributions,
used Precision, Recall and F1-score to find a reasonable threshold value (or decision boundary) for the probability value,
used this model to predict on a 50-50 subdataset from the test set (40%).

Note: There were some minor issues due to which the predictions came out erroneously.
Will fix it soon and upload the python notebook file along with tomorrow's post.

Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
