import random
import math
import statistics
from typing import Tuple, List

import matplotlib.pyplot as plt

from linear_model import LinearModel


class TargetFunction():
    """
    Class representing the target function that separates the data points into two classes.
    """

    def __init__(self, p: Tuple[float], q: Tuple[float]):
        """
        Initializes the TargetFunction with two points p and q.

        Args:
        - p: A tuple representing a point in 2D space.
        - q: A tuple representing a point in 2D space.
        """
        self.__m, self.__b = self.initialize(p, q)

    def initialize(self, p: Tuple[float], q: Tuple[float]) -> Tuple[float]:
        """
        Calculates the slope and the y-intercept of the line connecting points p and q.

        Args:
        - p: A tuple representing a point in 2D space.
        - q: A tuple representing a point in 2D space.

        Returns:
        - A tuple representing the slope and y-intercept of the line connecting points p and q.
        """
        m = (q[1] - p[1]) / (q[0] - p[0])
        b = m * q[0] - q[1]
        return m, b

    def classify(self, point: Tuple[float]) -> float:
        """
        Classifies a point based on its location relative to the target function.

        Args:
        - point: A tuple representing a point in 2D space.

        Returns:
        - 1 if the point is above the target function, -1 otherwise.
        """
        if point[1] > self.m * point[0] + self.b:
            return 1
        return -1

    @property
    def m(self) -> float:
        """
        Returns the slope of the target function.
        """
        return self.__m

    @property
    def b(self) -> float:
        """
        Returns the y-intercept of the target function.
        """
        return self.__b


class Perceptron():
    """
    Class representing a Perceptron model.
    """

    def __init__(self):
        """
        Initializes the Perceptron model with zero weights and bias.
        """
        self.__weights = (0, 0)
        self.__bias = 0

    def predict(self, inputs: List[float]) -> int:
        """
        Predicts the class of a data point.

        Args:
        - inputs: A list representing the data point.

        Returns:
        - 1 if the point is above the target function, -1 otherwise..
        """
        wtx = sum([w * x for w, x in zip(self.weights, inputs)]) + self.bias
        return int(math.copysign(1, wtx))

    def train(self, inputs: List[Tuple[float]], outputs: List[int], max_iterations: int = 10) -> Tuple[int, List[float]]:
        """
        Trains the Perceptron model on a set of labeled data points using the perceptron learning algorithm.

        Args:
        - inputs: A list of tuples representing the features of each data point.
        - outputs: A list of integers representing the labels of each data point.

        Returns:
        - The number of iterations needed to converge.
        """
        best = Perceptron()
        errors = [1.0]
        min_error = 1.0

        for _ in range(max_iterations):
            predictions = [self.predict(x) for x in inputs]
            missclassified_points = self.__missclassified_points(
                inputs, outputs, predictions
            )
            current_error = len(missclassified_points)/len(outputs)
            if current_error < min_error:
                min_error = current_error
                best.bias = self.bias
                best.weights = self.weights

            # Random select a missclassfied point to update the weights
            if missclassified_points:
                random_missclassfied_point = random.choice(
                    missclassified_points
                )
                x = random_missclassfied_point[0]
                y = random_missclassfied_point[1]
                self.__update_weights(x, y)
            errors.append(min_error)

        return (best, errors)

    def train_lm(self, inputs: List[Tuple[float]], outputs: List[int], max_iterations: int = 100) -> Tuple[int, List[float]]:
        """
        Trains the Perceptron model on a set of labeled data points using the perceptron learning algorithm.

        Args:
        - inputs: A list of tuples representing the features of each data point.
        - outputs: A list of integers representing the labels of each data point.

        Returns:
        - The number of iterations needed to converge.
        """
        best = Perceptron()
        errors = [1.0]
        min_error = 1.0

        linear_model = LinearModel()
        linear_model.train(inputs, outputs)
        self.__bias = linear_model.weights[0]
        self.__weights = linear_model.weights[1:]

        for _ in range(max_iterations):
            predictions = [self.predict(x) for x in inputs]
            missclassified_points = self.__missclassified_points(
                inputs, outputs, predictions
            )
            current_error = len(missclassified_points)/len(outputs)
            if current_error < min_error:
                min_error = current_error
                best.bias = self.bias
                best.weights = self.weights

            # Random select a missclassfied point to update the weights
            random_missclassfied_point = random.choice(
                missclassified_points
            )
            x = random_missclassfied_point[0]
            y = random_missclassfied_point[1]
            self.__update_weights(x, y)

            errors.append(min_error)

        return (best, errors)

    def test(self, inputs: List[Tuple[float]], outputs: List[int]) -> float:
        """
        Tests the accuracy of the Perceptron on the given input and output data.

        Args:
        - inputs: A list of input points, where each point is a tuple of floats.
        - outputs: A list of expected output classifications for each input point.

        Returns:
        - The ratio of misclassified points to the total number of input points.
        """
        predictions = [self.predict(x) for x in inputs]
        missclassified_points = self.__missclassified_points(
            inputs, outputs, predictions
        )

        return len(missclassified_points) / len(outputs)

    def __missclassified_points(self, inputs: List[Tuple[float]], outputs: List[int], predictions: List[int]):
        """
        Returns a list of misclassified points.

        Args:
        - inputs: A list of input points, where each point is a tuple of floats.
        - outputs: A list of expected output classifications for each input point.
        - predictions: A list of predicted output classifications for each input point.

        Returns:
        - A list of misclassified input points and their corresponding output and predicted classifications.
        """
        return list(
            filter(lambda x: x[1] != x[2], zip(inputs, outputs, predictions))
        )

    def __update_weights(self, x: Tuple[float], y: float) -> None:
        """
        Updates the weights and bias of the Perceptron based on a misclassified point.

        Args:
        - x: The misclassified input point as a tuple of floats.
        - y: The expected output classification of the misclassified point.

        """
        self.__bias += y
        self.__weights = [w + y * xi for w, xi in zip(self.weights, x)]

    @property
    def weights(self) -> float:
        """
        Returns the current weights of the Perceptron.
        """
        return self.__weights

    @weights.setter
    def weights(self, weights):
        """
        Sets the current weights of the Perceptron.
        """
        self.__weights = weights

    @property
    def bias(self) -> float:
        """
        Returns the current bias of the Perceptron.
        """
        return self.__bias

    @bias.setter
    def bias(self, bias):
        """
        Sets the current bias of the Perceptron.
        """
        self.__bias = bias


class Experiment():
    """
    A class representing an experiment that generates random points and trains a perceptron to classify them.
    """

    def __init__(self, input_size: int, max_iterations: int = 10, experiment_size: int = 1000, test_size: int = 1000):
        self.__input_size = input_size
        self.__max_iterations = max_iterations
        self.__experiment_size = experiment_size
        self.__test_size = test_size
        self.__target_function = None
        self.__perceptron = None
        self.__pocket_errors = []
        self.__errors = []

        self.__inputs = None
        self.__outputs = None
        self.__test_inputs = None
        self.__test_outputs = None

    def start(self) -> None:
        for _ in range(self.__experiment_size):
            self.__initialize()
            self.__inputs = self.__generate_points(self.__input_size)
            self.__outputs = self.__classify_points(self.__inputs)
            # Generate noise
            for i in range(len(self.__outputs)):
                if random.random() < 0.1:
                    self.__outputs[i] *= -1
            model, errors = self.perceptron.train(
                self.__inputs, self.__outputs, max_iterations=self.__max_iterations
            )

            self.perceptron = model
            for error in errors:
                self.pocket_errors.append(error)

            self.__test_inputs = self.__generate_points(self.__test_size)
            self.__test_outputs = self.__classify_points(self.__test_inputs)
            # Generate noise
            for i in range(len(self.__test_outputs)):
                if random.random() < 0.1:
                    self.__test_outputs[i] *= -1
            accuracy = self.perceptron.test(
                self.__test_inputs, self.__test_outputs
            )

            self.__errors.append(accuracy)

    def plot(self) -> None:
        intercept = -self.perceptron.bias/self.perceptron.weights[1]
        slope = -self.perceptron.weights[0]/self.perceptron.weights[1]

        x = self.__linespace(-1, 1, n=1000)
        y_pred = list(map(lambda x: slope * x + intercept, x))
        y = list(map(lambda x: self.target_function.m *
                 x + self.target_function.b, x))

        plt.plot(x, y, 'k--', label="f(x)")
        plt.plot(x, y_pred, 'b--', label="g(x)")
        plt.fill_between(x, 1, y_pred, color='green', alpha=0.05)
        plt.fill_between(x, y_pred, -1, color='red', alpha=0.05)

        x = [x[0] for x in self.__inputs]
        y = [x[1] for x in self.__inputs]
        c = ['g' if y == 1 else 'r' for y in self.__outputs]
        plt.scatter(x, y, c=c, marker="o", label="Train")

        x = [x[0] for x in self.__test_inputs]
        y_pred = [x[1] for x in self.__test_inputs]
        c = ['g' if y_pred == 1 else 'r' for y_pred in self.__test_outputs]
        plt.scatter(x, y_pred, c=c, marker="x", label="Test")

        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.legend()

    def __initialize(self) -> None:
        p, q = self.__generate_points(2)
        self.__target_function = TargetFunction(p, q)
        self.__perceptron = Perceptron()

    def __generate_points(self, size: int = 10) -> List[Tuple[float]]:
        return [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(size)]

    def __classify_points(self, inputs: List[Tuple[float]]) -> List[float]:
        return [self.target_function.classify(i) for i in inputs]

    def __linespace(self, lower: float, upper: float, n: int = 100) -> List[float]:
        return [lower + x*(upper - lower)/n for x in range(n)]

    @property
    def pocket_errors(self) -> List[float]:
        return self.__pocket_errors

    @property
    def perceptron(self) -> Perceptron:
        return self.__perceptron

    @perceptron.setter
    def perceptron(self, perceptron):
        self.__perceptron = perceptron

    @property
    def target_function(self) -> TargetFunction:
        return self.__target_function

    @property
    def mean_error(self) -> float:
        return statistics.mean(self.__errors)


def run_experiments():
    experiment = Experiment(
        input_size=100, max_iterations=10, experiment_size=1)
    experiment.start()
    experiment.plot()
    plt.savefig("N100_10.png")
    print(
        f"Approx P(f(x)≠g(x)): {experiment.mean_error}"
    )
    # plt.show()
    plt.clf()

    plt.plot(list(range(len(experiment.pocket_errors))),
             experiment.pocket_errors, 'k--', label="f(x)")
    plt.savefig("N100_10_steps.png")
    # plt.show()
    plt.clf()

    experiment = Experiment(
        input_size=100, max_iterations=50, experiment_size=1)
    experiment.start()
    experiment.plot()
    plt.savefig("N100_50.png")
    print(
        f"Approx P(f(x)≠g(x)): {experiment.mean_error}"
    )
    # plt.show()
    plt.clf()

    plt.plot(list(range(len(experiment.pocket_errors))),
             experiment.pocket_errors, 'k--', label="f(x)")
    plt.savefig("N100_50_steps.png")
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    run_experiments()
