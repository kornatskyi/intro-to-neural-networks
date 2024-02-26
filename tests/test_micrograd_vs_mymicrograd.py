import sys

from micrograd.nn import MLP
import numpy as np
from sklearn.datasets import load_iris

sys.path.append("./")
from src.NeuralNetwork import MLP as MyMLP
import random


def get_iris_data():
    random.seed(42)
    iris = load_iris()

    def minmax_norm(data: list):
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
        return (data - min_values) / (max_values - min_values)

    def zscore_norm(data):
        return (data - np.mean(data)) / np.std(data)

    combined_data = list(
        zip(zscore_norm(iris.data), minmax_norm(iris.target))
    )  # i score normalization

    random.shuffle(combined_data)
    shx, shy = zip(*combined_data)
    X, y = list([list(x) for x in shx]), list(shy)

    training_data_percent = 0.80
    training_size = int(len(X) * training_data_percent)
    testing_size = len(X) - training_size
    training_data_points, training_targets = X[:training_size], y[:training_size]
    testing_data_points, testing_targets = X[-testing_size:], y[-testing_size:]
    return training_data_points, training_targets, testing_data_points, testing_targets


def train_mlp(mlp, xs, ys):
    """
    Trains model returns loss
    """
    number_of_epochs = 10
    for epoch in range(number_of_epochs):
        print(f"Training: {(epoch / number_of_epochs) * 100}%")
        # Forward pass
        actual_ys = [mlp(x) for x in xs]
        loss = sum(
            (expected_y - actual_y) ** 2 for expected_y, actual_y in zip(ys, actual_ys)
        )
        # zero grad
        for p in mlp.parameters():
            p.grad = 0.0
        # Backward pass
        loss.backward()

        # Correcting parameters
        for p in mlp.parameters():
            p.data += -0.001 * p.grad

    loss = sum(
        (expected_y - actual_y) ** 2 for expected_y, actual_y in zip(ys, actual_ys)
    )

    # Backward pass
    loss.backward()

    for p in mlp.parameters():
        p.data += -0.01 * p.grad
    return loss


def test_models_equivalence_on_simplest_data():
    xs = [[2.0, 3.0, 1.0], [-3.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-2.0, -1.0, 7.0]]
    ys = [-1.0, -1.0, 1.0, -1.0]
    number_of_outputs_for_each_layer = [4, 1]
    random.seed(42)
    my_MLP = MyMLP(len(xs[0]), number_of_outputs_for_each_layer)

    random.seed(42)
    micro_mlp = MLP(len(xs[0]), number_of_outputs_for_each_layer)

    train_mlp(my_MLP, xs, ys)
    train_mlp(micro_mlp, xs, ys)

    for x in xs:
        pred1 = my_MLP(x)
        pred2 = micro_mlp(x)
        assert round(pred1.data, 4) == round(pred2.data, 4)


def test_models_equivalence_on_iris():
    train_xs, train_ys, test_xs, test_ys = get_iris_data()
    number_of_outputs_for_each_layer = [4, 4, 1]
    random.seed(42)
    my_MLP = MyMLP(len(train_xs[0]), number_of_outputs_for_each_layer)

    random.seed(42)
    micro_mlp = MLP(len(train_xs[0]), number_of_outputs_for_each_layer)

    train_mlp(my_MLP, train_xs, train_ys)
    train_mlp(micro_mlp, train_xs, train_ys)

    for x in test_xs:
        pred1 = my_MLP(x)
        pred2 = micro_mlp(x)
        assert round(pred1.data, 4) == round(pred2.data, 4)
