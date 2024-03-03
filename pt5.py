import random

def crossover(parent1, parent2):
    crossover_prob=0.5
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < crossover_prob:
            
            child.append(gene2)
        else:
            child.append(gene1)

    return child

chromosome_length = 20
population_max = 29
generation_max = 500

bbest = []
bbf = 0

for k in range(1):
    best = []
    bf = 0
    print(f"- Generation {k+1}")
    arrays = []
    for i in range(population_max):
        count = 0
        array = [random.randint(0, 1) for _ in range(chromosome_length)]
        for k in range(20):
            if(array[k] == 1):
                count = count + 1
                f = count
        if(bbf<f):
            bbf = f
            bbest = array         
        arrays.append(array)
        print(f"{i+1} : {array} (f:{f})")
    print(f"Best : {bbest} (f:{bbf})")

best = []
bf = 0
for k in range(generation_max):
    
    print(f"- Generation {k+2}")
    arrays = []
    for i in range(population_max):
        count = 0
        array = [random.randint(0, 1) for _ in range(chromosome_length)]
        for k in range(20):
            if(array[k] == 1):
                count = count + 1
                f = count
        if(bf<f):
            bf = f
            best = array         
        arrays.append(array)
        print(f"{i+1} : {array} (f:{f})")
    best = crossover(best,bbest)
    crosscount = 0
    for j in range(20):
            if(best[j] == 1):
                crosscount = crosscount + 1    
    print(f"Best : {best} (f:{crosscount})")
    if(bbf<bf):
        bbf=bf
        bbest = best
    
