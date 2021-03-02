import numpy as np


class Neutron:
    def __init__(self, weights: list, bias: float, name: str):
        self.weights = weights
        self.name = name
        self.bias = bias

    def calculate_output(self, inputs):
        """In deze definitie kijken we bij een perceptron welke inputs hij krijgt en hoe de weights en de bias dit
        beïnvloeden. Vervolgens kijken we of het antwoord de treshold haalt."""
        inputs_met_weight = [1/(1 + np.exp(-self.weights[i] * inputs[i])) for i in range(0, len(inputs))]  # Inputs * weights
        inputs_met_weight.append(1/(1 + np.exp(-self.bias)))
        return np.mean(inputs_met_weight)

    def __str__(self):
        """Informatie van de perceptron"""
        return 'Mijn naam is {} en ik heb {} input variabelen. Mijn bias is {}.'.format(self.name, str(len(self.weights)), str(self.bias))



