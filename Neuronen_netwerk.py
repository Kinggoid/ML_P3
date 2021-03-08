from typing import List

from ML.Neutron.Neuronen_laag import NeuronLaag


class NeuronNetwork:
    def __init__(self, lagen: List[NeuronLaag]):
        self.lagen = lagen

    def feed_forward(self, inputs: List[int]):
        """In deze definitie geven we de inputs van een perceptronlaag door naar de volgende laag tot we uiteindelijk
        de outputlaag bereiken."""

        alle_outputs = [inputs]
        for laag in self.lagen:  # Per laag
            x = laag.laag_forward(alle_outputs[-1])
            alle_outputs.append(x)  # Pak de output van de laatste laag en calculeer de output van deze laag voor alle lagen
        return alle_outputs[-1]  # Pak de output van de laatste laag

    def __str__(self):
        """Informatie van het perceptron network"""
        perceptronen = ''
        for i in range(0, len(self.lagen) - 1):
            perceptronen += str(self.lagen[i].name) + ', '
        perceptronen += self.lagen[-1].name

        return 'In dit network zitten de volgende lagen: {}'.format(perceptronen)