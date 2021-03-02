import math
import unittest

from ML.Neutron.Neutronen import Neutron


class MyTestCase(unittest.TestCase):
    def test_AND(self):
        """test de AND gate met de Perceptron inputs."""
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                AND = Neutron([1, 1], -2, 'AND gate')
                output = AND.calculate_output([i, j])
                print(output)
                if (output % 1) == 0.5:
                    antwoorden.append(int(math.ceil(output)))
                else:
                    antwoorden.append(int(round(output)))

        self.assertEqual(antwoorden, [0, 0, 0, 1])  # Kijk of de outputs goed zijn

    def test_NOT_GATE_P(self):
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            NOT = Neutron([1], 0, 'NOT gate')
            output = NOT.calculate_output([i])
            if (output % 1) == 0.5:
                antwoorden.append(math.ceil(output))
            else:
                antwoorden.append(round(output))

        self.assertEqual(antwoorden, [1, 0])  # Kijk of de outputs goed zijn

    def test_OR_beter(self):
        """test de OR gate met de Perceptron inputs."""
        print('jow')
        antwoorden = []  # Kijk per input wat het antwoord zou zijn.
        for i in range(0, 2):
            for j in range(0, 2):
                OR = Neutron([1, 1], -1, 'OR gate')
                output = OR.calculate_output([i, j])
                if (output % 1) == 0.5:
                    antwoorden.append(math.ceil(output))
                else:
                    antwoorden.append(round(output))

        self.assertEqual(antwoorden, [0, 1, 1, 1])  # Kijk of de outputs goed zijn


if __name__ == '__main__':
    unittest.main()
