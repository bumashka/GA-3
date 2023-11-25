# Bayg29.tsp
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
        # Тип кроссовера
        self.crossover_type = config['crossover_type']
        # Тип мутации
        self.mutation_type = config['mutation_type']
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
        self.population = []
        self.locations = []
        self.tspSize = None
        self.distances = []
        self.read_file()

    def read_file(self):
        with open("bayg29.txt") as f:
            lines = f.readlines()
            index = lines.index("DISPLAY_DATA_SECTION\n")
            for row in lines[index + 1:]:
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
        self.population = [np.arange(1, self.tspSize + 1, 1) for i in range(self.max_population)]
        for i in range(self.max_population):
            np.random.shuffle(self.population[i])
            self.population[i] = self.create_neighbour_tour(self.population[i])
    def getTotalDistance(self, indices):
        # distance between th elast and first city:
        distance = self.distances[indices[-1]-1][indices[0]-1]
        # add the distance between each pair of consequtive cities:
        for i in range(0, len(indices) - 1):
            distance += self.distances[indices[i] - 1][indices[i + 1] - 1]
        return distance

    def create_neighbour_tour(self, tur):
        result = [0 for i in range(self.tspSize)]
        for i in range(0, self.tspSize - 1):
            result[tur[i] - 1] = tur[i + 1]
        result[tur[-1] - 1] = tur[0]
        return result

    def from_neighbour_route(self, neighbour):
        result = [0 for i in range(len(neighbour))]
        result[0] = 1
        point = result[0]
        for i in range(1, len(neighbour)):
            result[i] = neighbour[point - 1]
            point = neighbour[point - 1]
        return result

    def isCycle(self, neighbour):
        tur = self.from_neighbour_route(neighbour)
        for point in tur:
            idx = abs(point)
            if tur[idx] < 0:
                return True
            tur[idx] = -tur[idx]
        return False

    def returnNotVisited(self, visited):
        not_visited = []
        for i in range(0, len(visited)):
            if visited[i]:
                not_visited.append(i + 1)
        return random.choice(not_visited)

    def plotData(self, indices):
        # plot the dots representing the cities:
        plt.clf()
        plt.scatter(*zip(*self.locations), marker='.', color='red')
        # create a list of the corresponding city locations:
        locs = [self.locations[i - 1] for i in indices]
        locs.append(locs[0])
        # plot a line between each pair of consequtive cities:
        plt.plot(*zip(*locs), linestyle='-', color='blue')
        return plt

    def alternating_edges(self):
        for i in range(0, len(self.population) // 2):
            if random.uniform(0, 1) <= self.crossover_chance:
                # chrom_a = self.population[i]
                # chrom_b = self.population[random.randint(0, len(self.population) - 1)]
                # chrom_a_b_, chrom_b_a_ = [0 for i in range(0, self.tspSize)], [0 for i in range(0, self.tspSize)]
                # chrom_a_b_visited, chrom_b_a_visited = [1 for i in range(0, self.tspSize)], [1 for i in range(0, self.tspSize)]
                # for i in range(self.tspSize):
                #     idx = i % 2
                #     if idx == 0:
                #         if chrom_a_b_visited[chrom_b[i] - 1]:
                #             chrom_a_b_[i] = chrom_b[i]
                #             chrom_a_b_visited[chrom_b[i] - 1] = 0
                #         else:
                #             chrom_a_b_[i] = self.returnNotVisited(chrom_a_b_visited)
                #             chrom_a_b_visited[chrom_a_b_[i] - 1] = 0
                #
                #         if chrom_b_a_visited[chrom_a[i] - 1]:
                #             chrom_b_a_[i] = chrom_a[i]
                #             chrom_b_a_visited[chrom_a[i] - 1] = 0
                #         else:
                #             chrom_b_a_[i] = self.returnNotVisited(chrom_b_a_visited)
                #             chrom_b_a_visited[chrom_b_a_[i] - 1] = 0
                #     else:
                #         if chrom_a_b_visited[chrom_a[i] - 1]:
                #             chrom_a_b_[i] = chrom_a[i]
                #             chrom_a_b_visited[chrom_a[i] - 1] = 0
                #         else:
                #             chrom_a_b_[i] = self.returnNotVisited(chrom_a_b_visited)
                #             chrom_a_b_visited[chrom_a_b_[i] - 1] = 0
                #
                #         if chrom_b_a_visited[chrom_b[i] - 1]:
                #             chrom_b_a_[i] = chrom_b[i]
                #             chrom_b_a_visited[chrom_b[i] - 1] = 0
                #         else:
                #             chrom_b_a_[i] = self.returnNotVisited(chrom_b_a_visited)
                #             chrom_b_a_visited[chrom_b_a_[i] - 1] = 0
                # self.children.append(chrom_a_b_)
                # self.children.append(chrom_b_a_)
                route_a = self.population[2 * i]
                route_b = self.population[2 * i + 1]

                new_route = [0] * self.tspSize
                not_visited = [i for i in range(1, self.tspSize + 1)]
                first = random.randint(1, self.tspSize - 1)
                not_visited.pop(first-1)

                current = first
                while len(not_visited) != 0:
                    a_dist = math.dist(
                        self.locations[current-1], self.locations[route_a[current-1]-1]
                    )
                    b_dist = math.dist(
                        self.locations[current-1], self.locations[route_b[current-1]-1]
                    )

                    next = route_a[current-1] if a_dist < b_dist else route_b[current-1]

                    if next in not_visited:
                        new_route[current-1] = next
                        not_visited.pop(not_visited.index(next))
                    else:
                        new_route[current-1] = not_visited.pop(
                            random.randint(0, len(not_visited)-1)
                        )

                    current = new_route[current-1]

                new_route[current-1] = first
                self.children.append(new_route)
    def selection(self):
        reselected_population = []
        f = [self.getTotalDistance(self.from_neighbour_route(tur)) for tur in self.population]
        sum_f = sum(f)
        fitness_function = [self.getTotalDistance(self.from_neighbour_route(tur)) / sum_f for tur in self.population]
        for i in range(0, self.max_population):
            for ff in range(0, int(abs(fitness_function[i]))):
                reselected_population.append(self.population[i])
            if random.uniform(0, 1) <= int(abs(fitness_function[i]) % 1 * 1000):
                reselected_population.append(self.population[i])
        self.population = reselected_population

    def crossover(self, type):
        if type == 1:
            self.alternating_edges()

    def swap_mutation(self):
        pass

    def mutation(self, type):
        for i in range(0, len(self.children)):
            if round(random.uniform(0, 1), 3) <= self.mutation_chance:
                a_place = random.randint(0, len(self.children) - 1)
                chrom_a = self.children[a_place]
                if type == 1:
                    old = chrom_a.copy()
                    el = random.randint(1, self.tspSize)
                    el_ = random.randint(1, self.tspSize)
                    child, child_ = chrom_a[el - 1], chrom_a[el_ - 1]
                    parent, parent_ = chrom_a.index(el), chrom_a.index(el_)
                    if chrom_a[el_ - 1] == el:
                        chrom_a[parent_] = el
                        chrom_a[el_ - 1] = child
                        chrom_a[el - 1] = el_
                    elif chrom_a[el - 1] == el_:
                        chrom_a[parent] = el_
                        chrom_a[el - 1] = child_
                        chrom_a[el_ - 1] = el
                    else:
                        chrom_a[el_ - 1] = child
                        chrom_a[parent] = el_
                        chrom_a[el - 1] = child_
                        chrom_a[parent_] = el
                self.children[a_place] = chrom_a

    def sort(self):
        return lambda x: self.getTotalDistance(self.from_neighbour_route(x))

    def reduction(self):
        self.population = self.population + self.children
        self.population.sort(key=self.sort(), reverse=False)
        self.population = self.population[:self.max_population]

    def run(self):
        epochs_without_changes = 0
        # self.plot_graph()
        execution_time = 0
        while self.current_epoch != self.max_epochs and epochs_without_changes != 15:
            self.current_epoch += 1
            start = time.time()
            self.selection()
            self.crossover(self.crossover_type)
            self.mutation(self.mutation_type)
            self.reduction()
            end = time.time()
            execution_time += (end - start)
            solutions = [self.getTotalDistance(self.from_neighbour_route(x)) for x in self.population]
            best_solution = min(solutions)
            print(
                f"""
++++++++++
Эпоха: {self.current_epoch}
Лучшее решение эпохи: {best_solution}
Лучшее решение за все эпохи: {self.current_best_solution}
Количество эпох без изменения результата: {epochs_without_changes}
Время вычисления эпохи в секундах: {end - start}
Время вычисления для всех эпох в секундах: {execution_time}
++++++++++
"""
            )
            if best_solution < self.current_best_solution:
                self.current_best_solution = best_solution
                epochs_without_changes = 0
            elif best_solution == self.current_best_solution:
                epochs_without_changes += 1
            else:
                epochs_without_changes = 0
            self.plotData(self.from_neighbour_route(self.population[solutions.index(best_solution)]))

        with open("result.txt", "a") as file:
            file.write(
                f"\n{self.mutation_chance},{self.crossover_chance},{self.max_population},{execution_time},{self.current_epoch},{self.current_best_solution}")


def main():
    indicise = [1, 28, 6, 12, 9, 26, 3, 29, 5, 21, 2, 20, 10, 4, 15, 18, 14, 17, 22, 11, 19, 25, 7, 23, 8, 27,
                16, 13, 24]
    config = {
        "crossover_chance": 0.7,
        "mutation_chance": 0.2,
        "max_population": 100,
        "max_epochs": 1000,
        "crossover_type": 1,
        "mutation_type": 1
    }
    GA = TSPGA(config)
    #print(GA.getTotalDistance(indicise))
    #GA.plotData(indicise)
    GA.run()


if __name__ == '__main__':
    random.seed(round(time.time()))
    main()
