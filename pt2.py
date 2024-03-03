import random

def binary_list_to_string(binary_list):
    return ''.join(str(bit) for bit in binary_list)

def one_point_crossover(parent1, parent2, crossover_point):
    if crossover_point < 0 or crossover_point >= len(parent1):
        raise ValueError("Invalid crossover point")

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

parent1_input = input("parent1 : ")
parent2_input = input("parent2 : ")

parent1 = [int(bit) for bit in parent1_input]
parent2 = [int(bit) for bit in parent2_input]

crosspoint = random.randint(1, 9)

child1, child2 = one_point_crossover(parent1, parent2, crosspoint)

print("Cut Point: before index ", crosspoint)
print("Child 1:", binary_list_to_string(child1))
print("Child 2:", binary_list_to_string(child2))