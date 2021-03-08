import math
import unittest

from ML.Neutron.Neuronen import Neuron
from ML.Neutron.Neuronen_laag import NeuronLaag
from ML.Neutron.Neuronen_netwerk import NeuronNetwork


class MyTestCase(unittest.TestCase):
    def test_AND(self):
        """test de AND gate met de Perceptron inputs."""
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        verwachte_outputs = [0, 0, 0, 1]

        AND = Neuron([100, 100], -200, 'AND gate')

        for i in inputs:
            output = AND.calculate_output(i)

            if (output % 1) == 0.5:
                antwoorden.append(int(math.ceil(output)))
            else:
                antwoorden.append(int(round(output)))

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

    def test_INVERT(self):
        """test de INVERT gate met de Perceptron inputs."""
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        inputs = [[0], [1]]
        verwachte_outputs = [1, 0]

        AND = Neuron([-1], 0, 'INVERT gate')

        for i in inputs:
            output = AND.calculate_output(i)

            if (output % 1) == 0.5:
                antwoorden.append(int(math.ceil(output)))
            else:
                antwoorden.append(int(round(output)))

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

    def test_OR(self):
        """test de OR gate met de Perceptron inputs."""
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        verwachte_outputs = [0, 1, 1, 1]

        AND = Neuron([1, 1], -1, 'OR gate')

        for i in inputs:
            output = AND.calculate_output(i)

            if (output % 1) == 0.5:
                antwoorden.append(int(math.ceil(output)))
            else:
                antwoorden.append(int(round(output)))

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

    def test_NOR(self):
        """test de NOR gate met de Perceptron inputs."""
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        verwachte_outputs = [1, 0, 0, 0, 0, 0, 0, 0]

        AND = Neuron([-1, -1, -1], 0, 'NOR gate')

        for i in inputs:
            output = AND.calculate_output(i)

            if (output % 1) == 0.5:
                antwoorden.append(int(math.ceil(output)))
            else:
                antwoorden.append(int(round(output)))

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

    """
    Hierboven blijkt dat de invoerparameters van de Perceptron blijkbaar ook werken als invoerparameters bij de 
    Neutron. Dit komt omdat, wanneer een antwoord goed is, deze minstens een 0 meegeven aan de sigmoid functie. En 
    sigmoid(0) = 0.5 wat afrond naar een 1. En alles wat niet goed is, komt daar net onder en wordt afgerond naar een 0.
    """


    def test_half_adder(self):
        # Maak het network
        x1_1 = Neuron([1, 1], -1, 'x1_1')
        x1_2 = Neuron([-1, -1], 1.5, 'x1_1')
        x1_3 = Neuron([1, 1], -2, 'x1_3')

        laag_een = NeuronLaag([x1_1, x1_2, x1_3], 'L1')

        x2_1 = Neuron([1, 1, 0], -2, 'x2_1')
        x2_2 = Neuron([0, 0, 1], -1, 'x2_2')

        laag_twee = NeuronLaag([x2_1, x2_2], 'L2')

        Half_adder = NeuronNetwork([laag_een, laag_twee])

        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for i in inputs:
            output = Half_adder.feed_forward(i)
            print(output)
            tussen_antwoorden = []
            for j in output:
                if (j % 1) == 0.5:
                    tussen_antwoorden.append(int(math.ceil(j)))
                else:
                    tussen_antwoorden.append(int(round(j)))

            antwoorden.append(tussen_antwoorden)

        print(antwoorden)

        self.assertEqual(antwoorden, [[0, 0], [1, 0], [1, 0], [0, 1]])  # Kijk of de outputs goed zijn


if __name__ == '__main__':
    unittest.main()
