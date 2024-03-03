import random

arrays = []

for i in range(30):
    array = [[random.randint(0, 1) for _ in range(10)]]
    arrays.append(array)

for i, array in enumerate(arrays):
    for row in array:
        count = 0
        for k in range(10):
            if(row[k] == 1):
                count = count + 1
                f = count 
        print(f"{i} : {row} (f:{f})")

random_index1 = random.randint(0,29)
random_index2 = random.randint(0,29)
random_index3 = random.randint(0,29)
random_index4 = random.randint(0,29)

p1 = arrays[random_index1]
p2 = arrays[random_index2]
p3 = arrays[random_index3]
p4 = arrays[random_index4]

p1 = p1[0]
p2 = p2[0]
p3 = p3[0]
p4 = p4[0]

f1 = 0
f2 = 0
f3 = 0
f4 = 0

for j in range(10):
    if(p1[j] == 1):
        f1 = f1 + 1
    if(p2[j] == 1):
        f2 = f2 + 1
    if(p3[j] == 1):
        f3 = f3 + 1
    if(p4[j] == 1):
        f4 = f4 + 1

if (f1>=f2):
    pr1 = p1
    f1 = f1
elif (f2>f1):
    pr1 = p2
    f1 = f2

if (f3>=f4):
    pr2 = p3
    f2 = f3
elif (f4>f3):
    pr2 = p4
    f2 = f4

print("Tournament selection")
print(f"Parent 1 : {pr1} (f:{f1})")
print(f"Parent 2 : {pr2} (f:{f2})")

def uniform_crossover(parent1, parent2):
    crossover_prob=0.5
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < crossover_prob:
            
            child.append(gene2)
        else:
            child.append(gene1)

    return child

offspring = uniform_crossover(pr1,pr2)

f3 = 0
for h in range(10):
    if(offspring[h] == 1):
        f3 = f3 + 1

print("Uniform crossover")
print(f"Offspring : {offspring} (f:{f3})")