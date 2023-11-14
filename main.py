# Bayg29.tsp
import codecs
import csv
import math
import random
import sys
import time

import numpy as np
from matplotlib import pyplot as plt


class TSPGA:

    def __init__(self, config):
        # Максимальное количество хромосом в популяции
        self.max_population = config['max_population']
        # Максимальное количество эпох
        self.max_epochs = config['max_epochs']
        # Актуальная эпоха
        self.current_epoch = 0
        # Вероятность кроссовера
        self.crossover_chance = config['crossover_chance']
        # Вероятность мутации
        self.mutation_chance = config['mutation_chance']
        # Актуальная популяция хромосом
        # Потомки актуальной популяции
        self.children = []
        # Лучшее решение актуальной популяции
        self.current_best_solution = sys.maxsize - 1

    def read_file(self):
        self.locations = []
        with open("bayg29.txt") as f:
            lines = f.readlines()
            index = lines.index("DISPLAY_DATA_SECTION\n")
            for row in lines[index+1:]:
                if row != 'EOF':
                    row = row.rstrip()
                    elements = row.split(" ")
                    elements = [el for el in elements if el != ""]
                    self.locations.append(np.asarray([elements[1], elements[2]], dtype=np.float32))
                else:
                    break
            self.tspSize = len(self.locations)
            # initialize distance matrix by filling it with 0's:
            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]
            # populate the distance matrix with calculated distances:
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    # calculate euclidean distance between two ndarrays:
                    distance = np.linalg.norm(self.locations[j] - self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
        self.population = [np.arange(1, self.tspSize+1, 1) for i in range(self.max_population)]
        for tur in self.population:
            np.random.shuffle(tur)

    def getTotalDistance(self, indices):
        # distance between th elast and first city:
        distance = self.distances[indices[-1]][indices[0]]
        # add the distance between each pair of consequtive cities:
        for i in range(0, len(indices)-1):
            distance += self.distances[indices[i]-1][indices[i + 1]-1]
        return distance

    def create_neighbour_tour(self, tur):
        result = [0 for i in range(self.tspSize)]
        for i in range(0, self.tspSize):
            if result[i] != 0:
                result[i] = tur[i+1]
                tur.index[tur[i]]


    def plotData(self, indices):
        # plot the dots representing the cities:
        plt.scatter(*zip(*self.locations), marker='.', color='red')
        # create a list of the corresponding city locations:
        locs = [self.locations[i-1] for i in indices]
        locs.append(locs[0])
        # plot a line between each pair of consequtive cities:
        plt.plot(*zip(*locs), linestyle='-', color='blue')
        return plt
    def selection(self):
        reselected_population = []
        f = [self.getTotalDistance(tur) for tur in self.population]
        sum_f = sum(f)
        fitness_function = [self.getTotalDistance(tur)/sum_f for tur in self.population]
        for i in range(0, self.max_population):
           for ff in range(0, int(abs(fitness_function[i]))):
               reselected_population.append(self.population[i])
           if random.uniform(0, 1) <= int(abs(fitness_function[i]) % 1 * 1000):
               reselected_population.append(self.population[i])
        self.population = reselected_population

    def crossover(self):
        for i in range(0, len(self.population)):
            if random.uniform(0, 1) <= self.crossover_chance:
                chrom_a = self.population[i]

                chrom_b = self.population[random.randint(0, len(self.population)-1)]

                k = random.randint(0, self.chromosome_length-1)

                chrom_a_ = chrom_a[:k] + chrom_b[k:]
                chrom_b_ = chrom_b[:k] + chrom_a[k:]

                self.children.append(chrom_a_)
                self.children.append(chrom_b_)

    def mutation(self):
        for i in range(0, len(self.children)):
            if round(random.uniform(0, 1), 3) <= self.mutation_chance:
                a_place = random.randint(0, len(self.children)-1)
                chrom_a = self.children[a_place]
                k = random.randint(0, self.chromosome_length-1)
                temp = list(chrom_a)
                if chrom_a[k] == '0':
                    temp[k] = '1'
                else:
                    temp[k] = '0'
                chrom_a_ = ''.join(temp)
                self.children[a_place] = chrom_a_

    def sort(self):
        return lambda x: self.function(self.from_binary_to_number(x))

    def reduction(self):
        self.population = self.population + self.children
        self.population.sort(key=self.sort(), reverse=True)
        self.population = self.population[:self.max_population]

    def run(self):
        epochs_without_changes = 0
        delta = 0.001
        #self.plot_graph()
        execution_time = 0
        while self.current_epoch != self.max_epochs and epochs_without_changes != 10:
            self.current_epoch += 1
            start = time.time()
            self.selection()
            self.crossover()
            self.mutation()
            self.reduction()
            end = time.time()
            execution_time+=(end-start)
            best_solution = max([self.function(self.from_binary_to_number(x)) for x in self.population])
            print(
f"""
++++++++++
Эпоха: {self.current_epoch}
Лучшее решение эпохи: {best_solution}
Лучшее решение за все эпохи: {self.current_best_solution}
Количество эпох без изменения результата: {epochs_without_changes}
Время вычисления эпохи в секундах: {end- start}
Время вычисления для всех эпох в секундах: {execution_time}
++++++++++
"""
            )
            dif = abs(best_solution - self.current_best_solution)
            if best_solution > self.current_best_solution \
                    and dif > delta:
                self.current_best_solution = best_solution
                epochs_without_changes = 0
            elif dif <= delta:
                epochs_without_changes += 1
            else:
                epochs_without_changes = 0
            #self.plot_graph()

        with open("result.txt", "a") as file:
            file.write(f"\n{self.mutation_chance},{self.crossover_chance},{self.max_population},{execution_time},{self.current_epoch},{self.current_best_solution}")


def main():
    indicise = [1, 28, 6, 12, 9, 26, 3, 29, 5, 21, 2, 20, 10, 4, 15, 18, 14, 17, 22, 11, 19, 25, 7, 23, 8, 27,
                16, 13, 24]
    config = {
        "crossover_chance" : 0.7,
        "mutation_chance" : 0.2,
        "max_population" : 100,
        "max_epochs" : 100,
        "lower_bound" : -20,
        "upper_bound" : -3.1,
    }
    GA = TSPGA(config)
    GA.read_file()
    print(GA.getTotalDistance(indicise))
    GA.plotData(indicise)

if __name__ == '__main__':
    random.seed(round(time.time()))
    main()
