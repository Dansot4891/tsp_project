import random

def create(length):
    return [random.choice([0, 1]) for _ in range(length)]

def initialize_population(pop_size, individual_length):
    return [create(individual_length) for _ in range(pop_size)]

def fitness(individual):
    return sum(individual)

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def select(population, fitness_values, num_to_select):
    selected = []
    total_fitness = sum(fitness_values)
    for _ in range(num_to_select):
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, ind in enumerate(population):
            current += fitness_values[i]
            if current > pick:
                selected.append(ind)
                break
    return selected

def replace(population, children):
    return children

def genetic(pop_size, individual_length, generations, mutation_rate):
    population = initialize_population(pop_size, individual_length)
    for generation in range(generations):
        fitness_values = [fitness(ind) for ind in population]
        
        print(f"Generation {generation + 1}:")
        for i, ind in enumerate(population):
            print(f"{i + 1}: {ind}, (f : {fitness_values[i]})")

        best_fitness = max(fitness_values)
        best = population[fitness_values.index(best_fitness)]
        print(f"Best : {best} (f : {best_fitness})")

        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select(population, fitness_values, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = replace(population, new_population)

    return best

if __name__ == "__main__":
    pop_size = 30
    individual_length = 20
    generations = 300
    mutation_rate = 0.01

    best_solution = genetic(pop_size, individual_length, generations, mutation_rate)
    #print("Best Solution:", best_solution)