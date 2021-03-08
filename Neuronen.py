import numpy as np


class Neuron:
    def __init__(self, weights: list, bias: float, name: str):
        self.weights = weights
        self.name = name
        self.bias = bias

    def calculate_output(self, inputs):
        """In deze definitie kijken we bij een perceptron welke inputs hij krijgt en hoe de weights en de bias dit
        beïnvloeden. Vervolgens kijken we of het antwoord de treshold haalt."""
        inputs_met_weight = [self.weights[i] * inputs[i] for i in range(0, len(inputs))]  # Inputs * weights
        update = sum(inputs_met_weight) + self.bias
        return 1 / (1 + np.exp(-update))

    def __str__(self):
        """Informatie van de perceptron"""
        return 'Mijn naam is {} en ik heb {} input variabelen. Mijn bias is {}.'.format(self.name, str(len(self.weights)), str(self.bias))



