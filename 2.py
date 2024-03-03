from Gene_module import *

gen = []  #Generation
for i in range(30):
    gen.append(Make_Chromosome(20))
for i in range(50):
    print(f"- Generation {i}")
    for k in range(len(gen)):
        print(f"{k}: {gen[k][0]} (f:{gen[k][1]})")
    p1,p2 = Roulette(gen), Roulette(gen)

    offspring1= Uniform_crossover(p1,p2)
    mutation(offspring1)
    Replacement(gen,offspring1)
    print(f"Best: {ReturnBest(gen)}")


