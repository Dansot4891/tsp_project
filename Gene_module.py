import random

# x개의 유전자를 가진 [크로모좀, 적합도] 반환, flag는 적합도를 반환할지 결정
def Make_Chromosome(x, flag=True):
    chromosome = ''.join(str(random.randint(0, 1)) for _ in range(x))
    if flag:
        fit = Fitness(chromosome)
        return [chromosome, fit]
    return chromosome

# chromosome 중 1의 개수 반환
def Fitness(chromosome):
    return chromosome.count('1')

# x, y 각각의 크로모좀 중 랜덤 확률이 보다 크면 적합도가 큰 것 반환, 아니면 작은 것 반환
def tournament(x, y,p=0.5):
    if x[1] >= y[1]:
        winner, loser = x, y
    else:
        winner, loser = y, x
    tmp = random.random()
    return winner if tmp > p else loser

# 룰렛 선택 적합도 비중이 더 큰것이 선택될 확률이 크다. 
def Roulette(gen):
    sum_fit = sum([chromo[1] for chromo in gen])
    point = random.uniform(0,sum_fit)
    current_fit = 0
    for chromo in gen:
        current_fit += chromo[1]
        if point<current_fit:
            return chromo
# 가중치 준 룰렛 
# def Roulette_Grad(gen):
#     #(Cw-Ci) + (Cw-Cb)/(k-1)
#     fit_list = [chromo[1] for chromo in gen]
#     C_b = max(fit_list)
#     C_w = min(fit_list)
#     k = 3
#     G = abs((C_w-C_b)/k)
#     sum_fit = sum([abs(C_w-fit)+G for fit in fit_list])
#     point = random.uniform(0,sum_fit)
#     current_fit = 0
    
#     for fit in fit_list:
#         current_fit += abs(C_w-fit)+G
#         if point<current_fit:
#             return gen[fit_list.index(fit)]
        
# 룰렛 선택 적합도 비중이 더 큰것이 선택될 확률이 크다. , 적합도를 랭크에 가중치를 준것으로 한다 
def Roulette_rank(gen, Gradient = -1, min = 0 ):
    gen = sorted(gen, key=lambda item: item[1], reverse=True)
    k = len(gen)
    sum_fit = k*min + k*(k-1)*abs(Gradient)/2
    point = random.uniform(0,sum_fit)
    current_fit = 0
    for i in range(len(gen)-1,-1,-1):
        current_fit += min+i*abs(Gradient)
        if point<current_fit:
            return gen[i]


    
# 중간에 지점 선택해서 서로 바꾸기 
def OnePoint_Crossover(x,y,cut_point=-1): 
    if cut_point==-1:
        cut_point = random.randint(0,len(x[0])-1)
        # print(f"cut_point {cut_point}")
    x = list(x[0]) 
    y = list(y[0])
    for i in range(cut_point):
        x[i],y[i] = y[i],x[i]
    x[0] = ''.join(x)
    y[0] = ''.join(y)

    fit_1 = Fitness(x[0]) 
    fit_2 = Fitness(y[0])
    return [x[0],fit_1] , [y[0],fit_2]

# 2개의 크로노좀 중 각각의 유전자에서 랜덤 확률이 0.5보다 크면 앞에 것의 유전자
def Uniform_crossover(x, y):
    chromosome = ''.join(a if random.random() >= 0.5 else b for a, b in zip(x[0], y[0]))
    fit = Fitness(chromosome)
    return [chromosome, fit]


# 돌연변이 발생
def mutation(offspring, p=0.5):
    chromosome = ''.join(random.choice(['0','1']) if random.random() < p else (i) for i in offspring[0])

    fit = Fitness(chromosome)
    return [chromosome,fit]

#제일 작은 적합도 가진거보다 적합도가 크면 서로 교체 해버리기
def Replacement(Generation, x):
    min_value = min(Generation, key=lambda item: item[1])
    min_index = Generation.index(min_value)
    if min_value[1]<x[1]:
        Generation[min_index] = x

#제일 큰거 반환
def ReturnBest(Generation):
    return max(Generation, key=lambda item: item[1])