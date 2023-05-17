import random
import math
import statistics
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


class TargetFunction():
    """
    Class representing the target function that separates the data points into two classes.
    """

    def classify(self, point: Tuple[float]) -> float:
        """
        Classifies a point based on its location relative to the target function.

        Args:
        - point: A tuple representing a point in 2D space.

        Returns:
        - 1 if the point is above the target function, -1 otherwise.
        """
        return int(math.copysign(1, point[0]**2 + point[1]**2 - 0.6))


class LinearModel():
    """
    Class representing a Linear model.
    """

    def __init__(self):
        """
        Initializes the Linear model.
        """
        self.__weights = None

    def predict(self, inputs: List[float]) -> int:
        """
        Predicts the class of a data point.

        Args:
        - inputs: A list representing the data point.

        Returns:
        - 1 if the point is above the target function, -1 otherwise..
        """
        wtx = np.matmul((1, inputs[0], inputs[1]), self.__weights)
        return int(math.copysign(1, wtx))

    def train(self, inputs: List[Tuple[float]], outputs: List[int]) -> None:
        """
        Trains the linear_model model on a set of labeled data points using the linear_model learning algorithm.

        Args:
        - inputs: A list of tuples representing the features of each data point.
        - outputs: A list of integers representing the labels of each data point.

        Returns:
        - The number of iterations needed to converge.
        """
        x = [(1, x[0], x[1]) for x in inputs]
        x_pinv = np.linalg.pinv(x)
        self.__weights = np.matmul(x_pinv, outputs)

    def test(self, inputs: List[Tuple[float]], outputs: List[int]) -> float:
        """
        Tests the error of the linear_model on the given input and output data.

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

    @property
    def weights(self) -> float:
        """
        Returns the current weights of the linear_model.
        """
        return self.__weights


class Experiment():
    """
    A class representing an experiment that generates random points and trains a linear_model to classify them.
    """

    def __init__(self, input_size: int, experiment_size: int = 1000, test_size: int = 1000):
        self.__input_size = input_size
        self.__experiment_size = experiment_size
        self.__test_size = test_size
        self.__target_function = None
        self.__linear_model = None
        self.__out_of_sample_error = []
        self.__in_sample_error = []

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

            self.linear_model.train(self.__inputs, self.__outputs)
            in_sample_error = self.linear_model.test(
                self.__inputs, self.__outputs
            )
            self.__in_sample_error.append(in_sample_error)

            self.__test_inputs = self.__generate_points(self.__test_size)
            self.__test_outputs = self.__classify_points(self.__test_inputs)

            out_of_sample_error = self.linear_model.test(
                self.__test_inputs, self.__test_outputs
            )
            self.__out_of_sample_error.append(out_of_sample_error)

    def plot(self) -> None:
        intercept = -self.linear_model.weights[0]/self.linear_model.weights[2]
        slope = -self.linear_model.weights[1]/self.linear_model.weights[2]

        x = (self.__linespace(-1, 1, n=1000), self.__linespace(-1, 1, n=1000))
        y_pred = list(map(lambda x: slope * x + intercept, x[0]))

        plt.plot(x[0], y_pred, 'b--', label="g(x)")
        plt.fill_between(x[0], 1, y_pred, color='green', alpha=0.05)
        plt.fill_between(x[0], y_pred, -1, color='red', alpha=0.05)

        x = [x[0] for x in self.__inputs]
        y = [x[1] for x in self.__inputs]
        c = ['g' if y == 1 else 'r' for y in self.__outputs]
        plt.scatter(x, y, c=c, marker="o", label="Train")

        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.legend()

    def __initialize(self) -> None:
        self.__target_function = TargetFunction()
        self.__linear_model = LinearModel()

    def __generate_points(self, size: int = 10) -> List[Tuple[float]]:
        return [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(size)]

    def __classify_points(self, inputs: List[Tuple[float]]) -> List[float]:
        return [self.target_function.classify(i) for i in inputs]

    def __linespace(self, lower: float, upper: float, n: int = 100) -> List[float]:
        return [lower + x*(upper - lower)/n for x in range(n)]

    @property
    def linear_model(self) -> LinearModel:
        return self.__linear_model

    @property
    def target_function(self) -> TargetFunction:
        return self.__target_function

    @property
    def mean_out_of_sample_error(self) -> float:
        return statistics.mean(self.__out_of_sample_error)

    @property
    def mean_in_sample_error(self) -> float:
        return statistics.mean(self.__in_sample_error)


def run_experiments():
    experiment = Experiment(input_size=100)
    experiment.start()
    experiment.plot()
    plt.savefig("N100.png")
    print(
        f"Approx P(f(x)≠g(x)) (in sample): {experiment.mean_in_sample_error}"
    )
    # plt.show()
    plt.clf()

    experiment = Experiment(input_size=1000)
    experiment.start()
    experiment.plot()
    plt.savefig("N1000.png")
    print(
        f"Approx P(f(x)≠g(x)) (out of sample): {experiment.mean_out_of_sample_error}"
    )
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    run_experiments()
