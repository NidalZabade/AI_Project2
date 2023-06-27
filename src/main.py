import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

TEST_SIZE = 0.3
K = 3


class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def distance(self, x, y):
        """
        Given two vectors x and y, return the Euclidean distance between them
        """
        return np.linalg.norm(x - y)

    def getMajority(self, neighbors):
        """
        Given a list of neighbors, return the majority label
        """
        labels = [neighbor[0] for neighbor in neighbors]
        return max(set(labels), key=labels.count)

    def predict(self, features, k):
        """
        Given a list of features vectors of testing examples
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors
        """
        predictions = []
        for feature in features:
            distances = []
            for i in range(len(self.trainingFeatures)):
                distances.append(
                    (
                        self.trainingLabels[i],
                        self.distance(feature, self.trainingFeatures[i]),
                    )
                )
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k]
            predictions.append(self.getMajority(neighbors))
        return predictions


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert into a list of
    features vectors and a list of target labels. Return a tuple (features, labels).

    features vectors should be a list of lists, where each list contains the
    57 features vectors

    labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    print("Loading data...")
    df = pd.read_csv(filename, header=None)  # read csv file
    features = df.iloc[:, :-1].values  # get features
    labels = df.iloc[:, -1].values  # get labels
    return features, labels


def preprocess(features):
    """
    normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation
    """

    for i in range(len(features[0])):
        mean = np.mean(features[:, i])
        std = np.std(features[:, i])
        features[:, i] = (features[:, i] - mean) / std
    return features


def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    model = MLPClassifier(random_state=1, max_iter=300).fit(features, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(labels)):
        if labels[i] == 1 and predictions[i] == 1:
            tp += 1
        elif labels[i] == 1 and predictions[i] == 0:
            fn += 1
        elif labels[i] == 0 and predictions[i] == 1:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


def main():
    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data("spambase.csv")
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE
    )

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)


if __name__ == "__main__":
    print("Running main")
    main()
