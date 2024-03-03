from Gene_module import *
lst = []
gen = []  #Generation
for i in range(30):
    gen.append(Make_Chromosome(20))
for i in range(50):
    print(f"- Generation {i}")
    for k in range(len(gen)):
        print(f"{k}: {gen[k][0]} (f:{gen[k][1]})")
    c1,c2,c3,c4 = random.sample(range(30),4)
    p1,p2 = tournament(gen[c1],gen[c2]), tournament(gen[c3],gen[c4])
    offspring1,offspring2 = OnePoint_Crossover(p1,p2)
    mutation(offspring1,0.05)
    mutation(offspring2,0.05)
    Replacement(gen,offspring1)
    Replacement(gen,offspring2)
    print(ReturnBest(gen)[1])
