graph = {
    '국어': {'A'},
    '영어': {'A', 'B'},
    '수학': {'A', 'D', 'E'},
    '과학': {'B', 'E'},
    '컴퓨': {'C', 'D'},
}

def greedy_coloring(graph):
    colors = {}  # 정점에 할당된 색을 저장하는 딕셔너리
    classroom_count = 0  # 사용된 교실의 수

    for vertex in graph:
        # 인접한 정점들의 색을 확인
        neighbor_colors = {colors[neighbor] for neighbor in graph[vertex] if neighbor in colors}

        if neighbor_colors:
            # 사용된 색 중에서 가장 낮은 색을 할당
            assigned_color = min(set(range(1, max(neighbor_colors)+2)) - neighbor_colors)
        else:
            # 아직 사용되지 않은 색 중에서 가장 낮은 색을 할당
            assigned_color = 1

        colors[vertex] = assigned_color
        classroom_count = max(classroom_count, assigned_color)

    return classroom_count, colors

classroom_count, coloring = greedy_coloring(graph)
print(f"최소 교실 수: {classroom_count}")
print("과목 색칠 결과:", coloring)