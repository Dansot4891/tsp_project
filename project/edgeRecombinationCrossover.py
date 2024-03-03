import random

# 테스트용 상수
CITIES = 17

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

# 테스트용 출력 (실행 시마다 결과가 달라짐)
print(edgeRecombinationCrossover([4, 9, 10, 11, 8, 15, 6, 3, 14, 2, 16, 7, 5, 12, 0, 13, 1],[13, 14, 15, 11, 6, 7, 3, 0, 12, 5, 16, 2, 4, 1, 10, 9, 8]))
