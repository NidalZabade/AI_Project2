import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import sys


TEST_SIZE = 0.3
K = 3


class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        """
        Given a list of features vectors of testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors.
        """
        predictions = []
        for feature in features:
            distances = []
            for i in range(len(self.trainingFeatures)):
                # calculate distance between feature and trainingFeatures using np.linalg.norm (euclidean distance = sqrt(sum((x - y)^2)))
                distances.append(
                    (
                        self.trainingLabels[i],
                        np.linalg.norm(
                            feature - self.trainingFeatures[i]
                        ),  # euclidean distance = sqrt(sum((x - y)^2))
                    )
                )
            distances.sort(key=lambda x: x[1])  # sort by distance
            neighbors = distances[:k]  # get k nearest neighbors
            labels = [neighbor[0] for neighbor in neighbors]  # get labels of neighbors
            predictions.append(max(set(labels), key=labels.count))  # get majority label
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
    features = []
    labels = []
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            features.append([float(i) for i in row[:-1]])
            labels.append(int(row[-1]))
    print("Data loaded.")
    return features, labels


def preprocess(features):
    """
    normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation
    """
    return (features - np.mean(features, axis=0)) / np.std(
        features, axis=0
    )  # normalize (x - mean) / std


def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000).fit(
        features, labels
    )  # train MLP model
    return model


# confusion matrix
def confusion_matrix(labels, predictions):
    tp = np.sum(np.logical_and(labels, predictions))  # true positive
    fp = np.sum(np.logical_and(np.logical_not(labels), predictions))  # false positive
    tn = np.sum(
        np.logical_and(np.logical_not(labels), np.logical_not(predictions))
    )  # true negative
    fn = np.sum(np.logical_and(labels, np.logical_not(predictions)))  # false negative
    return tp, fp, tn, fn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    tp, fp, tn, fn = confusion_matrix(labels, predictions)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")
    
    file_name = sys.argv[1]
    # Load data from spreadsheet and split into train and test sets if no arguments are given to the program
    features, labels = load_data(file_name)

    features = preprocess(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE
    )

    print("Preprocessing data...")

    # Train a k-NN model and make predictions
    model_knn = NN(X_train, y_train)
    knn_predictions = model_knn.predict(X_test, K)
    knn_tp, knn_fp, knn_tn, knn_fn = confusion_matrix(y_test, knn_predictions)
    knn_accuracy, knn_precision, knn_recall, knn_f1 = evaluate(y_test, knn_predictions)

    # Train an MLP model and make predictions
    model_mlp = train_mlp_model(X_train, y_train)
    mlp_predictions = model_mlp.predict(X_test)
    mlp_tp, mlp_fp, mlp_tn, mlp_fn = confusion_matrix(y_test, mlp_predictions)
    mlp_accuracy, mlp_precision, mlp_recall, mlp_f1 = evaluate(y_test, mlp_predictions)
    print("Data preprocessed.")

    print("Plotting results...")


    # Plot confusion matrix
    confusion_matrix_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    confusion_matrix_fig.suptitle("Confusion Matrix")
    ax1.matshow([[mlp_tp, mlp_fp], [mlp_fn, mlp_tn]], cmap=plt.cm.Blues)
    ax1.set_title("MLP")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Spam", "Not Spam"])
    ax1.set_yticklabels(["Spam", "Not Spam"])
    ax1.text(0, 0, mlp_tp, ha="center", va="center", color="black")
    ax1.text(0, 1, mlp_fn, ha="center", va="center", color="black")
    ax1.text(1, 0, mlp_fp, ha="center", va="center", color="black")
    ax1.text(1, 1, mlp_tn, ha="center", va="center", color="black")
    ax2.matshow([[knn_tp, knn_fp], [knn_fn, knn_tn]], cmap=plt.cm.Blues)
    ax2.set_title("k-NN")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Spam", "Not Spam"])
    ax2.set_yticklabels(["Spam", "Not Spam"])
    ax2.text(0, 0, knn_tp, ha="center", va="center", color="black")
    ax2.text(0, 1, knn_fn, ha="center", va="center", color="black")
    ax2.text(1, 0, knn_fp, ha="center", va="center", color="black")
    ax2.text(1, 1, knn_tn, ha="center", va="center", color="black")

    results_fig, ax3 = plt.subplots(1, 1, figsize=(15, 5))

    # Plot results
    results_fig.suptitle("Results")
    ax3.bar(
        np.arange(4),
        [mlp_accuracy, mlp_precision, mlp_recall, mlp_f1],
        width=0.3,
        label="MLP",
    )
    ax3.bar(
        np.arange(4) + 0.3,
        [knn_accuracy, knn_precision, knn_recall, knn_f1],
        width=0.3,
        label="k-NN",
    )
    ax3.set_ylabel("Score")
    ax3.set_xticks(np.arange(4) + 0.3 / 2)
    ax3.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax3.set_ylim([0, 1])
    ax3.set_title("MLP vs k-NN")
    ax3.text(0, mlp_accuracy, round(mlp_accuracy, 3), ha="center", va="bottom")
    ax3.text(1, mlp_precision, round(mlp_precision, 3), ha="center", va="bottom")
    ax3.text(2, mlp_recall, round(mlp_recall, 3), ha="center", va="bottom")
    ax3.text(3, mlp_f1, round(mlp_f1, 3), ha="center", va="bottom")
    ax3.text(0.3, knn_accuracy, round(knn_accuracy, 3), ha="center", va="bottom")
    ax3.text(1.3, knn_precision, round(knn_precision, 3), ha="center", va="bottom")
    ax3.text(2.3, knn_recall, round(knn_recall, 3), ha="center", va="bottom")
    ax3.text(3.3, knn_f1, round(knn_f1, 3), ha="center", va="bottom")
    ax3.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running main")
    main()
