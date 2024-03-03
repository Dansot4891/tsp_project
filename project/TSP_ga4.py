import random
import numpy
import itertools 
import math
import time



# 도시 개수
CITIES = 48
# population에 포함된 chromosome의 개수
SIZE = 40
# 세대
generation = 1


# 도시 간 거리
table = [[]]
# population
p = [[]]
# fitness
f = [0 for i in range(SIZE)]


def mix_initialize():
    p = []
    start = list(range(SIZE))
    
    random.shuffle(start)
        
    for i in range(SIZE):
        current = start[i]
        unvisited = list(range(SIZE))
        unvisited.remove(current)
        individual = [current]

        while unvisited:
            # 다음 도시 선택
            next_node = min(unvisited, key=lambda node: table[current][node])
            unvisited.remove(next_node)
            individual.append(next_node)
            current = next_node

        p.append(individual)
    
    return p

# 파일로부터 도시 간 거리 읽어와서 반환
def getTable(filename):
    f = open(filename, 'r')
    table = [[0 for col in range(CITIES)] for row in range(CITIES)]
    
    for i in range(CITIES):
        table[i] = list(map(float, f.readline().split()))

    return table
# ^getTable^

    
# 초기 population 반환 (order-based)
def getPopulation():
    p = [[col for col in range(CITIES)] for row in range(SIZE)]

    for i in range(SIZE):
        random.shuffle(p[i])
    
    return p
# ^getPopulation^


# chromosome의 fitness(도시 순회 시 총 거리) 반환
def getDistance(c):
    dist = 0
    
    for i in range(CITIES-1):
        dist += table[c[i]][c[i+1]]
    dist += table[c[CITIES-1]][c[0]]

    return dist
# ^getDistance^


# 가장 좋은 fitenss인(짧은 거리인) chromosome의 index 반환
def getBestIdx():
    minDist = numpy.finfo(numpy.float64).max
    idx = -1

    for i in range(SIZE):
        if(f[i] <= minDist):
            minDist = f[i]
            idx = i

    return idx
# ^getBestIdx^


# 가장 나쁜 fitness인(긴 거리인) chromosome의 index 반환
def getWorstIdx():
    maxDist = 0
    idx = -1

    for i in range(SIZE):
        if(f[i] >= maxDist):
            maxDist = f[i]
            idx = i

    return idx
# ^getWorstIdx^

#룰렛
def rouletteWheel():
    #(Cw-Ci) + (Cw-Cb)/(k-1) , k = 3
    Cw = max(f)
    Cb = min(f)
    sum_fit = sum([(Cw-fit)+(Cw-Cb)/2 for fit in f])
    point = random.uniform(0,sum_fit)
    current_fit = 0 

    for i in range(SIZE):
        current_fit += (Cw-f[i])+(Cw-Cb)/2
        if point<=current_fit:
            return i

# tournamentSelection
def tournamentSelection(i1, i2):
    if(random.random() > 0.5):
        return i1 if f[i1] <= f[i2] else i2
    else:
        return i2 if f[i1] <= f[i2] else i1
# ^tournamentSelection^


# orderCrossOver
def orderCrossover(c1, c2, l, r):
    c = [0 for i in range(CITIES)]
    isVisited = [False for i in range(CITIES)]

    for i in range(l, r+1, 1):
        c[i] = c1[i]
        isVisited[c1[i]] = True

    for i in c2:
        if(isVisited[i] == False):
            r += 1
            c[r % CITIES] = i
    return c
# ^orderCrossover^

# cycleCrossover(chromosome1, chromosome2)
def cycleCrossover(c1, c2):
    # 반환할 offspring
    offspring = [-1 for i in range(CITIES)]
    
    # c1의 index 저장
    index = [-1 for i in range(CITIES)]
    for i in range(CITIES):
        index[c1[i]] = i

    # chromosome1에서 떼어올지 chromosome2에서 떼어올지 여부
    flag = True

    # cycle 되었는지 여부 저장
    isVisited = [False for i in range(CITIES)]

    for i in range(CITIES):
        if(offspring[i] == -1):
            temp = i
            
            while True:
                if(flag == True):
                    offspring[temp] = c1[temp] ### chromosome1
                else:
                    offspring[temp] = c2[temp] ### chromosome2
                    
                isVisited[temp] = True
                temp = index[c2[temp]]
                
                if(isVisited[temp] == True):
                    break
                
            flag = ~flag
                
    return offspring
# ^cycleCrossover^

# pmx(chromosome1, chromosome2, index1, index2)
def pmx(c1, c2, i1, i2):
    # 반환할 offspring
    offspring = [-1 for i in range(CITIES)]
    
    # c1의 index 저장
    index = [-1 for i in range(CITIES)]
    for i in range(CITIES):
        index[c1[i]] = i

    # chromosome1의 부분을 떼어다가 그대로 붙여넣기
    isVisited = [False for i in range(CITIES)]
    for i in range(i1, i2+1):
        offspring[i] = c1[i]
        isVisited[c1[i]] = True

    # chromosome2 중복 처리
    for i in range(0, CITIES):
        if not i1 <= i <= i2:  # chromosome1에서 붙여넣은 부분이 아니라면
            temp = c2[i]
            while isVisited[temp]:  # 중복이 아니게 될 때까지
                temp = c2[index[temp]]
            offspring[i] = temp
                
    return offspring
# ^pmx^

# edgeRecombinationCrossover(chromosome1, chromosome2)
def edgeRecombinationCrossover(c1, c2):
    offspring = [-1 for i in range(CITIES)]

    table = [set() for i in range(CITIES)] ### 인접 도시 저장
    c1.append(c1[0])
    c2.append(c2[0])
    for i in range(CITIES):
        table[c1[i]].add(c1[i+1])
        table[c1[i+1]].add(c1[i])
        table[c2[i]].add(c2[i+1])
        table[c2[i+1]].add(c2[i])

    size = [0 for i in range(CITIES)] ### 각 도시의 이웃 개수 저장
    for i in range(CITIES):
        size[i] = len(table[i])
    
    isVisited = [False for i in range(CITIES)] ### 방문여부를 True, False로 저장
    unvisited_list = [i for i in range(CITIES)] ### 미방문여부를 list로 저장
    

    x = c1[0] if random.random() >= 0.5 else c2[0] ### 부모 중 랜덤하게 골라 맨 처음 노드 저장
    for i in range(CITIES):
        isVisited[x] = True
        unvisited_list.remove(x)
        offspring[i] = x

        # offspring이 완성되었으면
        if(i == CITIES-1):
            return offspring

        # 인접 도시 리스트에서 x 삭제
        for i in range(CITIES):
            if(isVisited[i] == False):
                table[i].discard(x)
                size[i] -= 1

        # x의 이웃한 도시가 남아있지 않다면
        if(len(table[x]) == 0):
            # 방문하지 않은 도시들 중 랜덤으로 선택해 x에 저장
            x = random.choice(unvisited_list)
        # 아니라면
        else:
            # x의 이웃한 도시 중 가장 적은 인접 도시를 가진 도시를 뽑아 x에 저장 (2개 이상이라면 랜덤)
            neighbors = [[] for i in range(5)]
            for city in table[x]:
                neighbors[size[city]].append(city)
            for city in range(5):
                if(len(neighbors[city]) > 0):
                    x = random.choice(neighbors[city])
                    break
# ^edgeRecombinationCrossover^

def two_opt(c1):
    fit = getDistance(c1)
    c3 = c1.copy()
    for i in itertools.combinations((c1), 2):
        c2 = c1.copy()
        start, end = i[0], i[1]
        c2[start:end] = reversed(c1[start:end])
        if getDistance(c2) < fit:
            fit = getDistance(c2)
            c3 = c2.copy()
    return c3             

def three_opt(c):
    c3 = c
    dist = getDistance(c)

    for i, j, k in itertools.combinations(range(1, CITIES - 2), 3):
        new_c = c[:i] + c[i:j][::-1] + c[j:k][::-1] + c[k:]
        new_dist = getDistance(new_c)
        if new_dist < dist:
            dist = new_dist
            c3 = new_c

    return c3

def twohalf_opt(c):
    c1 = two_opt(c)
    c2 = three_opt(c)
    if(getDistance(c1) < getDistance(c)):
        return c1
    elif(getDistance(c2)< getDistance(c)):
        return c2
    else:
        return c

def mutate(individual):
    mutation_rate = 0.05
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

def inversion_mutation(individual, mutation_rate):
    # 뮤테이션 확률을 체크
    if random.random() < mutation_rate:
        # 랜덤하게 두 위치 선택
        index1, index2 = random.sample(range(len(individual)), 2)
        index1, index2 = min(index1, index2), max(index1, index2)

        # 선택한 범위 내의 유전자를 역전시킴
        individual[index1:index2+1] = reversed(individual[index1:index2+1])

    return individual    

# 이하 main
for i in range(5):
    start_itme = 0
    start_time = time.time()
    table = [[]]
    table = getTable(r'C:\Users\myung\generic\project\input48.txt')
    p = [[]]
    # fitness
    generation = 1
    p = getPopulation()
    f = [0 for i in range(SIZE)]

    for i in range(SIZE):
        f[i] = getDistance(p[i])

    # for i in range(SIZE):
    #     print(i," -> ",f[i])
    # print()


    while generation <=200:
        isVisited = []
        isVisited = [False for i in range(SIZE)]

        # crossover 3회
        for i in range(0, 3, 1):
            
            #RouletteWheel 사용
            i1 = rouletteWheel()
            i2 = rouletteWheel()

            #Tournament 사용
            # c1,c2,c3,c4= random.sample(range(SIZE),4)
            # i1,i2 = tournamentSelection(c1,c2), tournamentSelection(c3,c4)

            #offspring = edgeRecombinationCrossover(p[i1], p[i2])
            #offspring = orderCrossover(p[i1], p[i2], 5, 12)
            #offspring = cycleCrossover(p[i1], p[i2])
            offspring = pmx(p[i1], p[i2], 5, 12)
            
            
            #mutation
            #mutate(offspring)
            inversion_mutation(offspring, 0.05)
            
            #local search
            #offspring = two_opt(offspring)
            #offspring = twohalf_opt(offspring)
            #offspring = three_opt(offspring)

            worstIdx = getWorstIdx()
            p[worstIdx] = offspring
            f[worstIdx] = getDistance(p[worstIdx])
            
        generation += 1

    # 코드 실행 종료 시간 기록
    end_time = time.time()

    # 실행 시간 계산
    execution_time = end_time - start_time

    # 실행 시간 출력
    print(f"코드 실행 시간: {execution_time} 초")
    #print("최단 경로 : ", p[getBestIdx()])
    print("최단 거리 : ", f[getBestIdx()])
    # main 종료







