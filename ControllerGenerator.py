## 总体目标， 判断是否存在不动点，判断是否每个初始状态都能到达不动点，为每一个初始化状态计算一个最优SCF, 使其能以最快的速度稳定

"""
    1.
    功能:根据列表信息输出有向图的邻接矩阵
    输入：形如 lists=[[2,1,3,4],[3,2,1,4]]的列表，
         列表的每一个元素表示一种网络通路，
         lists[i][j]=z 表示有一条从节点j指向z的通路
    输出：二维列表输出邻接矩阵adjacency_matrix

"""


"""
    2.
    功能:
        1.根据邻接矩阵判断是否存在仅指向自己，没有其他出度的节点(不动点) X1,X2...   
        2.若存在，判断是否每一个节点都能到达这些节点中的至少一个  3.若能，给出所有到达长度中最短的路径
    输入：1 中计算所得的邻接矩阵
    输出： 1.列表fix_points，元素是一个表示一个不动点 
          2.列表 path，每一个元素path[i]包括离初始节点i最近的path[i][0]不动点，以及该初始节点到不动点的最短路径path[i][1]
          3.所有最短路径中最长路径的长度

"""

"""
    3.
    功能: 根据2中给出的最短路径和1中的lists,对于每一个节点，对于最短路径上的一跳，遍历lists中的元素list[i]，记录能实现这一跳的所有网络通路i作为一个集合
    输入：1. 1中的lists 2. 2中保存的最短路径path
    输出：1.一个列表choice_sets，列表中每一个元素choice_sets[i]包含距离节点i最近的里不动点choice_sets[i][0]，
        能实现最短路径上每一条对应的网络通路在lists中的最小下标，比如choice[i][1][0]表示能实现节点i到达最近不动点的路径的第一跳的网络通路在lists中的最小下标

"""

"""
    4.
    功能：利用3中算出的每一个节点通往不动点的网络通路序列，即从其每一条的集合中选择下表最小的那个


"""

"""
    5.
    功能：通过1中的adjacency_matrix和


"""
"""
    6.
    根据1中的邻接矩阵画出网络图要求
    把不动点放置在靠近图像中心的位置，
    然后把整个图按照等距圆环的形式放置，圆环的数量和最大长度max_path_len(这会作为参数输入)，
    节点按照各自到不动点最短距离的放置到对应圆环上，圆环由内到外表示最短距离的长度由小到大，
    此外，要求将下一条开始路径相同的node放得更密集
    
    节点间的连线只画出到达不动点的最短路径，并使用颜色淡一些的灰色虚线，
    不动点使用橙色表示，其他节点使用蓝色，要求蓝色和橙色有高对比度，所有节点的数字都省略，
    橙色节点的node_size=400,蓝色节点的node_size=200
    最后要求图像效果整体对称，协调美观

"""

### 对比试验
"""
    随机生成多个(100)SCF序列，计算所有初始状态到达稳定的平均最短时间
    与3中所有最短路径中最长路径的长度进行比较

"""


import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# 1. 根据列表信息输出邻接矩阵
def lists_to_adjacency_matrix(lists):
    n = len(lists[0])
    adjacency_matrix = [[0]*n for _ in range(n)]
    for path in lists:
        for i, node in enumerate(path):
            adjacency_matrix[i][node-1] = 1
    return adjacency_matrix

# 2. 找到不动点并寻找每个节点到不动点的最短路径
def find_fix_points_and_paths(adjacency_matrix):
    n = len(adjacency_matrix)
    fix_points = []
    for i in range(n):
        if adjacency_matrix[i][i] == 1 and sum(adjacency_matrix[i]) == 1:
            fix_points.append(i)

    paths = []
    max_path_len = 0
    total_path_len = 0
    reachable_count = 0

    # 广度优先搜索算法 BFS
    for start in range(n):
        visited = [False] * n
        queue = deque([(start, [start])])
        visited[start] = True
        found = False
        while queue and not found:
            current, path = queue.popleft()
            if current in fix_points:
                paths.append([current, path])
                path_length = len(path) - 1
                max_path_len = max(max_path_len, path_length)
                total_path_len += path_length
                reachable_count += 1
                found = True
                break
            for neighbor, connected in enumerate(adjacency_matrix[current]):
                if connected and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, path + [neighbor]))
        if not found:
            paths.append([None, []])

    average_path_len = total_path_len / reachable_count if reachable_count > 0 else None

    return fix_points, paths, max_path_len, average_path_len



# 3. 根据路径信息和网络通路信息找到最小下标通路集合
def find_choice_sets(lists, paths):
    choice_sets = []
    for fix_point, path in paths:
        if fix_point is None:
            choice_sets.append([None, []])
            continue
        choice = []
        for i in range(len(path)-1):
            current, next_node = path[i], path[i+1]
            min_index = float('inf')
            for idx, lst in enumerate(lists):
                if lst[current] == next_node + 1:  # lists中节点从1开始
                    min_index = min(min_index, idx)
            choice.append(min_index)
        choice_sets.append([fix_point, choice])
    return choice_sets



# 根据路径长度分类并绘制网络图
def draw_network_by_path_length(adjacency_matrix, paths, max_path_len):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    G = nx.DiGraph()
    n = len(adjacency_matrix)

    clusters = defaultdict(list)
    fix_points_set = set()

    # 找到不动点和对应路径
    node_to_fix_point_dist = {}
    next_hop_group = defaultdict(list)

    for idx, (fix_point, path) in enumerate(paths):
        if fix_point is not None and path:
            fix_points_set.add(fix_point)
            dist = len(path) - 1
            node_to_fix_point_dist[idx] = (fix_point, dist, path[1] if len(path) > 1 else fix_point)
            next_hop_group[(fix_point, path[1] if len(path) > 1 else fix_point)].append(idx)

    # 布局节点
    pos = {}
    center = np.array([0, 0])
    radius_step = 1

    # 放置不动点在中心
    angle_offset = 0
    num_fix_points = len(fix_points_set)
    for i, fix_point in enumerate(sorted(fix_points_set)):
        angle = 2 * np.pi * i / num_fix_points + angle_offset
        pos[fix_point + 1] = center + np.array([0.1 * np.cos(angle), 0.1 * np.sin(angle)])  # 微小偏移保证可视化更美观

    # 放置其他节点在对应圆环上
    for dist in range(1, max_path_len + 1):
        nodes_at_dist = [idx for idx, (_, d, _) in node_to_fix_point_dist.items() if d == dist]
        num_nodes = len(nodes_at_dist)
        for idx, node in enumerate(sorted(nodes_at_dist, key=lambda x: (node_to_fix_point_dist[x][0], node_to_fix_point_dist[x][2]))):
            angle = 2 * np.pi * idx / num_nodes
            r = dist * radius_step
            pos[node + 1] = center + np.array([r * np.cos(angle), r * np.sin(angle)])

    # 添加边
    for node, (_, _, next_hop) in node_to_fix_point_dist.items():
        G.add_edge(node + 1, next_hop + 1)

    plt.figure(figsize=(10, 10))

    # 画边
    nx.draw_networkx_edges(G, pos, style='dashed', edge_color='#CCCCCC', arrows=True, arrowsize=12)

    # 节点颜色和大小
    node_colors = ['#FF8000' if node - 1 in fix_points_set else '#0066CC' for node in G.nodes]
    node_sizes = [400 if color == 'orange' else 200 for color in node_colors]

    # 画节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    plt.axis('off')
    plt.show()


# def draw_network_by_path_length(adjacency_matrix, paths):
#     G = nx.DiGraph()
#     pos = {}
#     n = len(adjacency_matrix)

#     clusters = defaultdict(list)
#     fix_points_set = set()

#     for idx, (fix_point, path) in enumerate(paths):
#         if fix_point is not None:
#             clusters[fix_point].append(idx)
#             fix_points_set.add(fix_point)

#     # 布局各个群落
#     offset = 0
#     for fix_point, nodes in clusters.items():
#         subgraph = nx.DiGraph()
#         for node in nodes:
#             if paths[node][1]:
#                 path = paths[node][1]
#                 for i in range(len(path)-1):
#                     subgraph.add_edge(path[i]+1, path[i+1]+1)

#         sub_pos = nx.spring_layout(subgraph, center=(offset, 0))
#         pos.update(sub_pos)
#         offset += 2
#         G.add_edges_from(subgraph.edges)

#     plt.figure(figsize=(14, 8))

#     # 画边
#     nx.draw_networkx_edges(G, pos, style='dashed', edge_color='lightgray', arrows=True, arrowsize=15)

#     # 分别设置节点大小
#     node_colors = ['orange' if node-1 in fix_points_set else 'blue' for node in G.nodes]
#     node_sizes = [400 if color == 'orange' else 200 for color in node_colors]

#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

#     # plt.title('Network Clusters by Paths to Fix Points')
#     plt.axis('off')
#     plt.show()




# 示例
if __name__ == '__main__':
    # lists = [[41, 47, 41, 47, 53, 59, 53, 59, 65, 71, 65, 71, 41, 47, 41, 47, 54, 60, 54, 60, 65, 71, 65, 71, 41, 47, 41, 47, 53, 59, 53, 59, 65, 71, 65, 71, 109, 111, 109, 111, 122, 124, 122, 124, 133, 135, 133, 135, 110, 112, 110, 112, 122, 124, 122, 124, 133, 135, 133, 135, 109, 111, 109, 111, 121, 123, 121, 123, 133, 135, 133, 135, 41, 47, 41, 47, 54, 60, 54, 60, 65, 71, 65, 71, 42, 48, 42, 48, 54, 60, 54, 60, 65, 71, 65, 71, 41, 47, 41, 47, 53, 59, 53, 59, 65, 71, 65, 71, 109, 111, 109, 111, 122, 124, 122, 124, 133, 135, 133, 135, 110, 112, 110, 112, 122, 124, 122, 124, 133, 135, 133, 135, 109, 111, 109, 111, 121, 123, 121, 123, 133, 135, 133, 135], [41, 48, 41, 48, 53, 60, 53, 60, 65, 72, 65, 72, 41, 48, 41, 48, 53, 60, 53, 60, 65, 72, 65, 72, 41, 48, 41, 48, 53, 60, 53, 60, 65, 72, 65, 72, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 41, 48, 41, 48, 53, 60, 53, 60, 65, 72, 65, 72, 41, 48, 41, 48, 53, 60, 53, 60, 65, 72, 65, 72, 41, 48, 41, 48, 53, 60, 53, 60, 65, 72, 65, 72, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136], [37, 39, 37, 39, 53, 55, 53, 55, 69, 71, 69, 71, 37, 39, 37, 39, 54, 56, 54, 56, 69, 71, 69, 71, 37, 39, 37, 39, 53, 55, 53, 55, 69, 71, 69, 71, 109, 111, 109, 111, 126, 128, 126, 128, 141, 143, 141, 143, 110, 112, 110, 112, 126, 128, 126, 128, 141, 143, 141, 143, 109, 111, 109, 111, 125, 127, 125, 127, 141, 143, 141, 143, 37, 39, 37, 39, 54, 56, 54, 56, 69, 71, 69, 71, 38, 40, 38, 40, 54, 56, 54, 56, 69, 71, 69, 71, 37, 39, 37, 39, 53, 55, 53, 55, 69, 71, 69, 71, 109, 111, 109, 111, 126, 128, 126, 128, 141, 143, 141, 143, 110, 112, 110, 112, 126, 128, 126, 128, 141, 143, 141, 143, 109, 111, 109, 111, 125, 127, 125, 127, 141, 143, 141, 143], [37, 40, 37, 40, 53, 56, 53, 56, 69, 72, 69, 72, 37, 40, 37, 40, 53, 56, 53, 56, 69, 72, 69, 72, 37, 40, 37, 40, 53, 56, 53, 56, 69, 72, 69, 72, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 37, 40, 37, 40, 53, 56, 53, 56, 69, 72, 69, 72, 37, 40, 37, 40, 53, 56, 53, 56, 69, 72, 69, 72, 37, 40, 37, 40, 53, 56, 53, 56, 69, 72, 69, 72, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144], [5, 11, 5, 11, 17, 23, 17, 23, 29, 35, 29, 35, 5, 11, 5, 11, 18, 24, 18, 24, 29, 35, 29, 35, 5, 11, 5, 11, 17, 23, 17, 23, 29, 35, 29, 35, 109, 111, 109, 111, 122, 124, 122, 124, 133, 135, 133, 135, 110, 112, 110, 112, 122, 124, 122, 124, 133, 135, 133, 135, 109, 111, 109, 111, 121, 123, 121, 123, 133, 135, 133, 135, 5, 11, 5, 11, 18, 24, 18, 24, 29, 35, 29, 35, 6, 12, 6, 12, 18, 24, 18, 24, 29, 35, 29, 35, 5, 11, 5, 11, 17, 23, 17, 23, 29, 35, 29, 35, 109, 111, 109, 111, 122, 124, 122, 124, 133, 135, 133, 135, 110, 112, 110, 112, 122, 124, 122, 124, 133, 135, 133, 135, 109, 111, 109, 111, 121, 123, 121, 123, 133, 135, 133, 135], [5, 12, 5, 12, 17, 24, 17, 24, 29, 36, 29, 36, 5, 12, 5, 12, 17, 24, 17, 24, 29, 36, 29, 36, 5, 12, 5, 12, 17, 24, 17, 24, 29, 36, 29, 36, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 5, 12, 5, 12, 17, 24, 17, 24, 29, 36, 29, 36, 5, 12, 5, 12, 17, 24, 17, 24, 29, 36, 29, 36, 5, 12, 5, 12, 17, 24, 17, 24, 29, 36, 29, 36, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136, 109, 112, 109, 112, 121, 124, 121, 124, 133, 136, 133, 136], [1, 3, 1, 3, 17, 19, 17, 19, 33, 35, 33, 35, 1, 3, 1, 3, 18, 20, 18, 20, 33, 35, 33, 35, 1, 3, 1, 3, 17, 19, 17, 19, 33, 35, 33, 35, 109, 111, 109, 111, 126, 128, 126, 128, 141, 143, 141, 143, 110, 112, 110, 112, 126, 128, 126, 128, 141, 143, 141, 143, 109, 111, 109, 111, 125, 127, 125, 127, 141, 143, 141, 143, 1, 3, 1, 3, 18, 20, 18, 20, 33, 35, 33, 35, 2, 4, 2, 4, 18, 20, 18, 20, 33, 35, 33, 35, 1, 3, 1, 3, 17, 19, 17, 19, 33, 35, 33, 35, 109, 111, 109, 111, 126, 128, 126, 128, 141, 143, 141, 143, 110, 112, 110, 112, 126, 128, 126, 128, 141, 143, 141, 143, 109, 111, 109, 111, 125, 127, 125, 127, 141, 143, 141, 143], [1, 4, 1, 4, 17, 20, 17, 20, 33, 36, 33, 36, 1, 4, 1, 4, 17, 20, 17, 20, 33, 36, 33, 36, 1, 4, 1, 4, 17, 20, 17, 20, 33, 36, 33, 36, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 1, 4, 1, 4, 17, 20, 17, 20, 33, 36, 33, 36, 1, 4, 1, 4, 17, 20, 17, 20, 33, 36, 33, 36, 1, 4, 1, 4, 17, 20, 17, 20, 33, 36, 33, 36, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144, 109, 112, 109, 112, 125, 128, 125, 128, 141, 144, 141, 144]]
    lists = [
            [1, 23, 1, 23, 25, 31, 25, 31, 1, 23, 1, 23, 26, 32, 26, 32, 37, 55, 37, 55, 62, 64, 62, 64, 38, 56, 38, 56, 62, 64, 62, 64, 1, 23, 1, 23, 26, 32, 26, 32, 2, 24, 2, 24, 26, 32, 26, 32, 38, 56, 38, 56, 62, 64, 62, 64, 38, 56, 38, 56, 62, 64, 62, 64],
            [1, 24, 1, 24, 25, 32, 25, 32, 1, 24, 1, 24, 25, 32, 25, 32, 37, 56, 37, 56, 61, 64, 61, 64, 37, 56, 37, 56, 61, 64, 61, 64, 1, 24, 1, 24, 25, 32, 25, 32, 1, 24, 1, 24, 25, 32, 25, 32, 37, 56, 37, 56, 61, 64, 61, 64, 37, 56, 37, 56, 61, 64, 61, 64],
            [1, 19, 1, 19, 29, 31, 29, 31, 1, 19, 1, 19, 30, 32, 30, 32, 33, 51, 33, 51, 62, 64, 62, 64, 34, 52, 34, 52, 62, 64, 62, 64, 1, 19, 1, 19, 30, 32, 30, 32, 2, 20, 2, 20, 30, 32, 30, 32, 34, 52, 34, 52, 62, 64, 62, 64, 34, 52, 34, 52, 62, 64, 62, 64],
            [1, 20, 1, 20, 29, 32, 29, 32, 1, 20, 1, 20, 29, 32, 29, 32, 33, 52, 33, 52, 61, 64, 61, 64, 33, 52, 33, 52, 61, 64, 61, 64, 1, 20, 1, 20, 29, 32, 29, 32, 1, 20, 1, 20, 29, 32, 29, 32, 33, 52, 33, 52, 61, 64, 61, 64, 33, 52, 33, 52, 61, 64, 61, 64],
            [1, 7, 1, 7, 9, 15, 9, 15, 1, 7, 1, 7, 10, 16, 10, 16, 53, 55, 53, 55, 62, 64, 62, 64, 54, 56, 54, 56, 62, 64, 62, 64, 1, 7, 1, 7, 10, 16, 10, 16, 2, 8, 2, 8, 10, 16, 10, 16, 54, 56, 54, 56, 62, 64, 62, 64, 54, 56, 54, 56, 62, 64, 62, 64],
            [1, 8, 1, 8, 9, 16, 9, 16, 1, 8, 1, 8, 9, 16, 9, 16, 53, 56, 53, 56, 61, 64, 61, 64, 53, 56, 53, 56, 61, 64, 61, 64, 1, 8, 1, 8, 9, 16, 9, 16, 1, 8, 1, 8, 9, 16, 9, 16, 53, 56, 53, 56, 61, 64, 61, 64, 53, 56, 53, 56, 61, 64, 61, 64],
            [1, 3, 1, 3, 13, 15, 13, 15, 1, 3, 1, 3, 14, 16, 14, 16, 49, 51, 49, 51, 62, 64, 62, 64, 50, 52, 50, 52, 62, 64, 62, 64, 1, 3, 1, 3, 14, 16, 14, 16, 2, 4, 2, 4, 14, 16, 14, 16, 50, 52, 50, 52, 62, 64, 62, 64, 50, 52, 50, 52, 62, 64, 62, 64],
            [1, 4, 1, 4, 13, 16, 13, 16, 1, 4, 1, 4, 13, 16, 13, 16, 49, 52, 49, 52, 61, 64, 61, 64, 49, 52, 49, 52, 61, 64, 61, 64, 1, 4, 1, 4, 13, 16, 13, 16, 1, 4, 1, 4, 13, 16, 13, 16, 49, 52, 49, 52, 61, 64, 61, 64, 49, 52, 49, 52, 61, 64, 61, 64]
            ]
    #lists = [[2,1,3,4],[3,2,1,4]]
    # lists = [[2,3,1,4]]
    adjacency_matrix = lists_to_adjacency_matrix(lists)
    print("Adjacency Matrix:", adjacency_matrix)

    """
    0 1 1 0
    1 1 0 0
    1 0 1 0
    0 0 0 1
    """

    # 通过邻接矩阵绘制网络图
    # draw_network_from_adjacency_matrix(adjacency_matrix)

    fix_points, paths, max_path_len, average_path_len = find_fix_points_and_paths(adjacency_matrix)
    print("Fix Points:", fix_points)
    print("Paths:", paths)
    print("Max Path Length:", max_path_len)
    print("Average Path Length", average_path_len)
    # 144状态：max 4 ave 2.3958333333333335  64状态：max 2 ave 1.71875

    # 根据路径长度分类并绘制网络图
    # draw_network_by_path_length(adjacency_matrix, paths, max_path_len)

    # print(paths[108])
    choice_sets = find_choice_sets(lists, paths)
    print("Choice Sets:", choice_sets)
    # print(choice_sets[2],choice_sets[17],choice_sets[46],choice_sets[66],choice_sets[108],choice_sets[134]) # 实际值-1
    print(choice_sets[0],choice_sets[2],choice_sets[8],choice_sets[13],choice_sets[26],choice_sets[63])

    # 3,18,47,67,109,135的最优控制路径
    # [108, [2, 0]] [108, [0, 0, 0]] [108, [0, 0]] [108, [0, 1]] [108, []] [108, [0]]

    # 1,3,9,14,27,64的最优控制路径
    # [0, []] [0, [0]] [0, [0]] [63, [0, 0]] [0, [3, 0]] [63, []]
