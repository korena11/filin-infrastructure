import random


class LetThereBeRandomColors:
    def __init__(self):
        self.randomsGenerated = []

    def __get_random_color(self, pastel_factor=0.5):
        return tuple([int(255 * (x + pastel_factor) / (1.0 + pastel_factor)) for x in
                      [random.uniform(0, 1.0) for _ in [1, 2, 3]]])

    def __color_distance(self, c1, c2):
        return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])

    def GenerateNewColor(self, pastel_factor=0.5, numberOfColorsToGenerate=1):

        tempColorsList = []

        for ind in range(numberOfColorsToGenerate):
            max_distance = None
            best_color = None

            for i in range(0, 100):
                color = self.get_random_color(pastel_factor=pastel_factor)
                if not self.randomsGenerated:
                    best_color = color
                    break
                best_distance = min([self.color_distance(color, c) for c in self.randomsGenerated])
                if not max_distance or best_distance > max_distance:
                    max_distance = best_distance
                    best_color = color

            tempColorsList.append((best_color[0] / 255., best_color[1] / 255., best_color[2] / 255.))
            self.randomsGenerated.append(best_color)

        if numberOfColorsToGenerate == 1:
            return tempColorsList[0]
        return tempColorsList
