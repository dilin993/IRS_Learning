import numpy as np
from abc import ABC, abstractmethod


class Simulation(ABC):

    @abstractmethod
    def simulate(self):
        pass


class Result:

    def __init__(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text


class SimulationResult:

    def __init__(self, title,  xlabel, ylabel):
        self.results = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def add_result(self, x, y, text):
        self.results.append(Result(x, y, text))
