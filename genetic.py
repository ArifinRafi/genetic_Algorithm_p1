import numpy as np
import random

class GeneticAlgorithm:
    start_population_size = 10
    mutation_threshold = 0.3

    def __init__(self, path):
        self.limit = 100
        self.error = -1
        self.Data = []
        file = open(path, 'r')
        data = file.readline()
        file.close()

        for i, d in enumerate(data):
            temp = d[:-1].split()
            if i == 0
                self.error = int(temp[0])
            elif temp[0] == 'l':
                self.Data.append(int(temp[1]) * 1)
            elif temp[0] == 'd':
                self.Data.append(int(temp[1]))

            print(self.error)
            print(self.Data)
            self.best_fit_population = self.calc_goal_state()


    def get_best_fit(self):
        if np.any(self.best_fit_population == -1):
            return "-1"
        string_rep = np.array2string(self.best_fit_population, separator = "")
        return string_rep[1:-1]

    def calc_goal_state(self):
        self.population = np.random.randint(0, 2, (GeneticAlgorithm.start_population_size, self.error))
        for i in range(self.itr_limit):
            print("GEN:", i+1)
            new_population = []
            fitness_scores = self.fitness()
            max_index = np.where(fitness_scores == max(fitness_scores))[0]
            if 0 in fitness_scores:
                index = np.where(fitness_scores == 0)[0][0]
                print("Fittest candidate index", index)
                print("Best fit Candidate", self.population[index])
                if np.all(self.population[index] == 0):
                    self.population[index] = self.population[max_index]
                    print("Not Acceptable")

                else:
                    return self.population[index]
            for j in range(GeneticAlgorithm.start_population_size):
                x, y = self.select(fitness_scores)
                child = self.crossover(x, y)
                child = self.mutate(child)
                if np.all(child == 0):
                    continue

                new_population.append(child)
            self.population = new_population
        return  np.array([-1])

    def fitness(self):
        print("Population", self.population)
        score = []
        for people in self.population:
            total = 0
            for i in range(len(people)):
                if people[i] == 1:
                    total += self.Data[i]
            score.append(total)
        score = np.absolute(np.array(score))
        print("Fitness Score ", score)
        return score

    def select(self, scores):
        total = np.sum(scores)
        prob = total - scores
        prob = prob / np.sum(prob)
        print("Probability: ", prob)

        ind = []
        for i in range(len(self.population)):
            ind.append(i)
        choice = np.random.choice(ind, 2 True, prob)
        print(choice)
        return self.population[choice[0]], self.population[choice[1]]

    def crossingover(self, x, y):
        crossover_id = np.random.randint(0, self.error)
        child = np.append(x[:crossover_id], y[crossover_id:])
        print("X", x)
        print("Y", y)
        print("Crossover Index", crossover_id)
        print("child", child)
        return child

    def mutation(self, child):
        prob = GeneticAlgorithm.mutation_threshold
        if np.random.random(1)[0] > prob:
            mutation_id = np.random.randint(0, self.error)
            mutation_val = np.random.randint(0, 2)
            print("Before Mutation: ", child, mutation_id, mutation_val)
            child[mutation_id] = mutation_val
            print("After Mutation: ", child)
        return child

def tester():
    bank1 = GeneticAlgorithm('input0.txt')
    print(bank1.get_best_fit())

if __name__ == '__main__':
    tester()

