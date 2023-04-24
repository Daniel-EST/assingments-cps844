import random
import math
import statistics
from typing import Tuple, List

import matplotlib.pyplot as plt


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

    def train(self, inputs: List[Tuple[float]], outputs: List[int]) -> int:
        """
        Trains the Perceptron model on a set of labeled data points using the perceptron learning algorithm.
        
        Args:
        - inputs: A list of tuples representing the features of each data point.
        - outputs: A list of integers representing the labels of each data point.
        
        Returns:
        - The number of iterations needed to converge.
        """
        predictions = [self.predict(i) for i in inputs]
        missclassified_points = self.__missclassified_points(
            inputs, outputs, predictions
        )

        iterations = 0
        while len(missclassified_points) > 0:
            iterations += 1
            predictions = [self.predict(x) for x in inputs]
            missclassified_points = self.__missclassified_points(
                inputs, outputs, predictions
            )
            if len(missclassified_points) > 0:
                # Random select a missclassfied point to update the weights
                random_missclassfied_point = random.choice(
                    missclassified_points
                )
                x = random_missclassfied_point[0]
                y = random_missclassfied_point[1]
                self.__update_weights(x, y)

        return iterations

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

    @property
    def bias(self) -> float:
        """
        Returns the current bias of the Perceptron.
        """
        return self.__bias


class Experiment(): 
    """
    A class representing an experiment that generates random points and trains a perceptron to classify them.
    """
    def __init__(self, input_size: int, experiment_size: int = 1000, test_size: int = 1000):
        self.__input_size = input_size
        self.__experiment_size = experiment_size
        self.__test_size = test_size
        self.__target_function = None
        self.__perceptron = None
        self.__iterations = []
        self.__errors = []

    def start(self) -> None:
        for _ in range(self.__experiment_size):
            self.__initialize()
            inputs = self.__generate_points(self.__input_size)
            outputs = self.__classify_points(inputs)
            n_iterations = self.perceptron.train(inputs, outputs)

            test_inputs = self.__generate_points(self.__test_size)
            test_outputs = self.__classify_points(test_inputs)
            accuracy = self.perceptron.test(test_inputs, test_outputs)

            self.__errors.append(accuracy)
            self.__iterations.append(n_iterations)

    def plot(self) -> None:
        print(
            f"Mean iteration: {self.mean_iterations}, Mean error {self.mean_error}"
        )
        print("Ploting Graph: ") 

    def __initialize(self) -> None:
            p, q = self.__generate_points(2)
            self.__target_function = TargetFunction(p, q)
            self.__perceptron = Perceptron()

    def __generate_points(self, size: int = 10) -> List[Tuple[float]]:
        return [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(size)]

    def __classify_points(self, inputs: List[Tuple[float]]) -> List[float]:
        return [self.target_function.classify(i) for i in inputs]

    @property
    def perceptron(self) -> Perceptron:
        return self.__perceptron

    @property
    def target_function(self) -> TargetFunction:
        return self.__target_function

    @property
    def mean_error(self) -> float:
        return statistics.mean(self.__errors)

    @property
    def mean_iterations(self) -> float:
        return statistics.mean(self.__iterations)


def run_experiments():
    experiment = Experiment(input_size=10)
    experiment.start()
    experiment.plot()

    experiment = Experiment(input_size=100)
    experiment.start()
    experiment.plot()


if __name__ == "__main__":
    run_experiments()
