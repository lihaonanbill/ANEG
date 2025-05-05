from itertools import product
import numpy as np

def generate_game_states(strategy_counts):
    # 生成所有可能的策略组合
    strategies = product(*(range(count) for count in strategy_counts))
    
    # 格式化为字符串，并按字典序排列
    sorted_strategies = sorted("".join(map(str, strategy)) for strategy in strategies)
    
    return sorted_strategies

def calculate_payoffs(adj_matrix, payoff_matrices, strategy_counts):
    # 获取玩家数量
    num_players = len(adj_matrix)
    
    # 生成所有可能的局势
    game_states = generate_game_states(strategy_counts)
    
    # 初始化收益数组，形状为 (玩家数, 局势数)
    payoffs = np.zeros((num_players, len(game_states)))
    
    # 遍历每一种局势
    for index, state in enumerate(game_states):
        state = list(map(int, state))  # 将策略组合转换为整数列表
        
        # 计算每个玩家的收益
        for i in range(num_players):
            total_payoff = 0  # 记录玩家 i 在该局势下的总收益
            
            # 遍历所有其他玩家，检查连接并累加收益
            for j in range(num_players):
                if adj_matrix[i][j] == 1 and i != j:  # 只有在有连接时计算收益
                    total_payoff += payoff_matrices[f'C({i+1})({j+1})'][state[i]][state[j]]
            
            # 记录玩家 i 在该局势下的收益
            payoffs[i][index] = total_payoff
    
    return payoffs

# def generate_time_evolution_states(strategy_counts, Tmax):
#     # 生成所有可能的Tmax时间段内的局势
#     single_time_states = generate_game_states(strategy_counts)  # 单个时间点的所有局势
    
#     # 生成Tmax长度的所有时间序列组合
#     time_series_states = product(single_time_states, repeat=Tmax)
    
#     # 将每个时间序列连接成字符串并按字典序排列
#     sorted_time_series_states = sorted("".join(state) for state in time_series_states)
    
#     return sorted_time_series_states

def generate_time_evolution_states(strategy_counts, Tmax):
    # 生成所有可能的Tmax时间段内的局势，每位玩家的策略存储在自己的块中
    num_players = len(strategy_counts)
    
    # 为每个玩家生成可能的策略演化序列
    player_sequences = [list(product(map(str, range(strategy_counts[i])), repeat=Tmax)) for i in range(num_players)]

    # 生成所有玩家的组合
    time_series_states = list(product(*player_sequences))

    # 直接拼接各个玩家的策略演化字符串
    sorted_time_series_states = sorted("".join("".join(seq) for seq in state) for state in time_series_states)
    
    return sorted_time_series_states

def split_and_concat(s, n, k):
    # s是字符串，n是分块数，k是每个分块中要提取的数量
    
    # 计算每一份的长度
    part_length = len(s) // n
    result = []

    # 遍历每一份
    for i in range(n):
        if k==0:
            break
        # 获取当前份的起始和结束位置
        start = i * part_length
        end = start + part_length if i != n - 1 else len(s)  # 处理最后一份
        part = s[start:end]
        
        # 获取当前部分的后k个字符并加入结果
        result.append(part[-k:] if len(part) >= k else part)  # 如果该部分长度小于k，取整部分

    # 拼接结果并返回
    return ''.join(result)
  
# 把一个字符串分n块,取每一块的从右往左数第k个字符，然后拼接起来
def split_and_extract(s, n, k):
    # 计算每一份的长度
    part_length = len(s) // n
    result = []

    # 遍历每一份
    for i in range(n):
        # 获取当前份的起始和结束位置
        start = i * part_length
        end = start + part_length if i != n - 1 else len(s)  # 处理最后一份
        part = s[start:end]

        # 如果该部分的长度足够k，取第k个字符
        if len(part) >= k:
            result.append(part[-1-k])  # k-1 因为索引从0开始
        else:
            result.append('')  # 如果该部分的长度小于k，添加空字符串

    # 拼接结果并返回
    return ''.join(result)

def find_optimal_strategies(adj_matrix, payoff_matrices, strategy_counts, memory_lengths, Tmax, payoffs, game_states, indicator):
    # 生成所有可能的Tmax时间段内的局势
    time_evolution_states = generate_time_evolution_states(strategy_counts, Tmax)
    num_players = len(strategy_counts)
    
    optimal_strategies = []
    

    for state in time_evolution_states:
        next_strategies = []
        for i in range(num_players):
            # 检查是否更新 0表示更新，1表示不更新
            if indicator[i] == "1" :
                # 此时取该玩家上以时刻的信息
                best_strategy = state[Tmax*(i+1)-1]
            else :
                Ti = memory_lengths[i]  # 获取玩家 i 的记忆长度
            
                # 提取该玩家需要的历史信息（所有玩家在 Ti 轮内的策略）
                past_state = split_and_concat(state,num_players,Ti)
                
                best_strategy = None
                best_payoff = float('-inf')
                # 遍历所有可能的策略，找到最优策略
                for strategy in range(strategy_counts[i]):
                    total_payoff=0
                    # 生成重复字符串，替代Xi在Ti时间内的历史信息
                    modified_past_state = past_state[:i*Ti] + str(strategy)*Ti + past_state[(i+1)*Ti:]
                    print
                    #计算Xi过去每个时刻的的收益，求和
                    for k in range(Ti):
                        total_payoff += payoffs[i][game_states.index(split_and_extract(modified_past_state,num_players,k))]
                
                    # 选择收益最大的策略，如果相同，选择策略编号较大的
                    if total_payoff > best_payoff or (total_payoff == best_payoff and strategy > best_strategy):
                        best_payoff = total_payoff
                        best_strategy = strategy
            
            next_strategies.append(str(best_strategy))
        
        optimal_strategies.append("".join(next_strategies))
    
    return optimal_strategies

def insert_substring(s1, s2):
    n = len(s2)  # 获取n的值
    k = len(s1) // n  # 计算每份的长度
    result = []
    
    # 将 s1 分成 n 份，并将 s2 中的字符插入到每一份的最后
    for i in range(n):
        start = i * k
        end = start + k
        part = s1[start:end]  # 取出s1的第i份
        part += s2[i]  # 将s2的第i个字符插入到这一份的最后
        result.append(part)
    
    return ''.join(result)

def generate_optimal_augmented_state(time_evolution_states, optimal_strategies, memory_lengths, Tmax):
    optimal_augmented_state = []
    # 将每一个augmented state后Tmax-1 与optimal strategy拼接起来 
    for i in range(len(time_evolution_states)):
        # optimal_augmented_state.append(insert_substring(split_and_concat(time_evolution_states[i], len(memory_lengths), Tmax-1), optimal_strategies[i]))
        optimal_augmented_state.append(time_evolution_states.index(insert_substring(split_and_concat(time_evolution_states[i], len(memory_lengths), Tmax-1), optimal_strategies[i]))+1)
    
    return optimal_augmented_state

# 生成不同的矩阵分块, 0表示更新
def generate_blocks(strategy_counts,adj_matrix, payoff_matrices, memory_lengths, Tmax, payoffs, game_states, time_evolution_states, optimal_strategies):
    update_indicators = generate_game_states([2]*len(strategy_counts))
    print("update_indicators:"+str(update_indicators))
    Blocks = []
    for indicator in update_indicators:
        #print(indicator)
        optimal_strategies = find_optimal_strategies(adj_matrix, payoff_matrices, strategy_counts, memory_lengths, Tmax, payoffs, game_states, indicator)
        #print(optimal_strategies)
        Blocks.append(generate_optimal_augmented_state(time_evolution_states, optimal_strategies, memory_lengths, Tmax))
       
    return Blocks

# # 邻接矩阵
# adj_matrix = np.array([
#     [0, 1],
#     [1, 0]
# ])

# # 收益矩阵
# payoff_matrices = {
#     'C(1)(2)': np.array([[4, 3], [6, 4]]),
#     'C(2)(1)': np.array([[3, 2], [5, 1]])
# }

# # 策略数（每个玩家的可选策略数）
# strategy_counts = [2, 2]

# # 记忆长度（每个玩家的记忆长度）
# memory_lengths = [1, 2]

# 邻接矩阵
adj_matrix = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]

])



'''
[0, 1, 1],
[1, 0, 1],
[1, 1, 0]

'''


# C(x)(y)矩阵的第i行第j列表示玩家x与玩家j博弈时，玩家x使用策略i,玩家y使用策略j时玩家x的收益
# 收益矩阵
payoff_matrices = {
    # 'C(1)(2)': np.array([[4, 3], [6, 4]]),
    # 'C(2)(1)': np.array([[3, 2], [5, 1]]),
    # 'C(1)(3)': np.array([[5, 4], [7, 5]]),
    # 'C(3)(1)': np.array([[4, 3], [6, 2]])
    
    # 'C(1)(2)': np.array([[ 0, 6, 0], [9,  7,  5]]),
    # 'C(1)(3)': np.array([[4, 3], [6, 8]]),
    # 'C(2)(1)': np.array([[  1,  9], [10,   8], [  4,  3]]),
    # 'C(2)(3)': np.array([[5, 7], [1,  3], [ 6, 9]]),
    # 'C(3)(1)': np.array([[ 8, 2], [ 3,  0]]),
    # 'C(3)(2)': np.array([[  7,  1, 10], [  6,   9,   1]])

    # 两两之间都是囚徒困境的支付双矩阵
    # 'C(1)(2)': np.array([[ -1, -9], [0, -6]]),
    # 'C(1)(3)': np.array([[ -1, -9], [0, -6]]),
    # 'C(2)(1)': np.array([[ -1, -9], [0, -6]]),
    # 'C(2)(3)': np.array([[ -1, -9], [0, -6]]),
    # 'C(3)(1)': np.array([[ -1, -9], [0, -6]]),
    # 'C(3)(2)': np.array([[ -1, -9], [0, -6]])

    'C(1)(2)': np.array([[ 4, 4], [0, 10]]),
    'C(1)(3)': np.array([[ 4, 4], [0, 10]]),
    'C(2)(1)': np.array([[ 4, 4], [0, 10]]),
    'C(2)(3)': np.array([[ 4, 4], [0, 10]]),
    'C(3)(1)': np.array([[ 4, 4], [0, 10]]),
    'C(3)(2)': np.array([[ 4, 4], [0, 10]])

}

# 策略数（每个玩家的可选策略数）
strategy_counts = [2, 2, 2]

# 记忆长度（每个玩家的记忆长度）
memory_lengths = [1, 1, 2]


# 计算收益
game_states = generate_game_states(strategy_counts)
payoffs = calculate_payoffs(adj_matrix, payoff_matrices, strategy_counts)
print("Game States:", game_states)
print("Payoffs:", payoffs)

# 生成 Tmax 时间段内的所有局势
Tmax = max(memory_lengths)
time_evolution_states = generate_time_evolution_states(strategy_counts, Tmax)
print("Time Evolution States:", time_evolution_states)

# 计算每个玩家的最优策略, 每个玩家都更新
optimal_strategies = find_optimal_strategies(adj_matrix, payoff_matrices, strategy_counts, memory_lengths, Tmax, payoffs, game_states, '0'*len(strategy_counts))
print("Optimal Strategies for next step:", optimal_strategies)

# 计算最优增广后的state
optimal_augmented_state = generate_optimal_augmented_state(time_evolution_states, optimal_strategies, memory_lengths, Tmax)
print("Optimal Augmented State:", optimal_augmented_state)


# 计算分块矩阵
blocks = generate_blocks(strategy_counts,adj_matrix, payoff_matrices, memory_lengths, Tmax, payoffs, game_states, time_evolution_states, optimal_strategies)

# 这上面部分都是计算分块矩阵
#-------------------



#print("Blocks:", blocks, len(blocks))
print(len(blocks),"Blocks:")
for block in blocks:
    print(block)

# # 至少存在一个玩家更新
# indices=[1,6]
# new_blocks = [blocks[i] for i in indices]



def list_to_logical_matrix(lst):
    """将形如 [2,3,1,4] 的列表转换为 n×n 逻辑矩阵"""
    n = len(lst)
    matrix = np.zeros((n, n), dtype=int)
    for col, row in enumerate(lst):
        matrix[row - 1, col] = 1  # 由于索引从1开始，需要-1调整
    return matrix

def lists_to_logical_matrices(lst):
    """将多个逻辑矩阵的列表[[1,2,3,4],[1,2,1,1]] """
    maxtrices=[]
    for list in lst:
        matrices.append(list_to_logical_matrix(list))
    return matrices


def boolean_operations(*lists):
    """计算 k 个逻辑矩阵的布尔和，并计算幂直到收敛"""
    
    # 将所有列表转换为逻辑矩阵
    matrices = np.array([list_to_logical_matrix(lst) for lst in lists])
    print(matrices)

    # 计算布尔和（按位或）
    s = np.any(matrices, axis=0).astype(int)
    print("Boolean Sum (S^1):\n", s)

    # 计算布尔幂，直到结果不再变化
    prev_s = np.copy(s)
    power = 1
    while True:
        power += 1
        new_s = np.dot(prev_s, s).clip(0, 1)  # 进行布尔乘法 (0-1 矩阵运算)
        print(f"S^{power}:\n", new_s)

        # 如果结果不再改变，退出循环
        if np.array_equal(new_s, prev_s):
            break
        prev_s = new_s

    # 判断最终结果是否为逻辑矩阵（每列仅有一个 1，其余为 0）
    is_logical_matrix = np.all(np.sum(prev_s, axis=0) == 1)
    print(prev_s[63])
    print("Is final matrix a logical matrix?:", is_logical_matrix)

    return prev_s, is_logical_matrix

def weighted_logical_sum(lists, probabilities):
    """计算加权逻辑矩阵的和。"""
    if len(lists) != len(probabilities):
        raise ValueError("逻辑矩阵数量与概率数量不匹配")
    
    # 生成逻辑矩阵
    matrices = np.array([list_to_logical_matrix(lst) for lst in lists])
    probabilities = np.array(probabilities).reshape(-1, 1, 1)  # 调整形状用于广播
    
    # 计算加权和
    weighted_sum = np.sum(matrices * probabilities, axis=0)
    
    return weighted_sum

# 函数作用：判断一个矩阵中是否的行列相同的所有位置中是否存在值为1的元素，如果有，输出对应行号
def find_rows_with_ones_on_diagonal(matrix):
    """
    检查矩阵中主对角线上是否存在值为1的元素，并返回对应的行号。
    :param matrix: 2D list or numpy array
    :return: List of row indices where the diagonal element is 1
    """
    matrix = np.array(matrix)  # 转换为numpy数组，方便操作
    rows_with_ones = [i for i in range(min(matrix.shape)) if matrix[i, i] == 1]
    return rows_with_ones

# # 示例使用
# matrix = [
#     [1, 0, 0],
#     [0, 0, 1],
#     [0, 1, 1]
# ]

# result = find_rows_with_ones_on_diagonal(matrix)
# print("对角线值为1的行号:", result)




# 函数作用：判断一个矩阵中某一行的所有元素是否都大于0




# 示例输入
lists = [
    [22, 24, 22, 24, 30, 32, 30, 32, 22, 24, 22, 24, 30, 32, 30, 32, 50, 52, 50, 52, 58, 60, 58, 60, 50, 52, 50, 52, 58, 60, 58, 60, 22, 24, 22, 24, 30, 32, 30, 32, 22, 24, 22, 24, 30, 32, 30, 32, 49, 51, 49, 51, 57, 59, 57, 59, 49, 51, 49, 51, 57, 59, 57, 59]
# , [21, 24, 21, 24, 29, 32, 29, 32, 21, 24, 21, 24, 29, 32, 29, 32, 49, 52, 49, 52, 57, 60, 57, 60, 49, 52, 49, 52, 57, 60, 57, 60, 21, 24, 21, 24, 29, 32, 29, 32, 21, 24, 21, 24, 29, 32, 29, 32, 49, 52, 49, 52, 57, 60, 57, 60, 49, 52, 49, 52, 57, 60, 57, 60]
# , [18, 20, 18, 20, 30, 32, 30, 32, 18, 20, 18, 20, 30, 32, 30, 32, 50, 52, 50, 52, 62, 64, 62, 64, 50, 52, 50, 52, 62, 64, 62, 64, 18, 20, 18, 20, 30, 32, 30, 32, 18, 20, 18, 20, 30, 32, 30, 32, 49, 51, 49, 51, 61, 63, 61, 63, 49, 51, 49, 51, 61, 63, 61, 63]
# , [17, 20, 17, 20, 29, 32, 29, 32, 17, 20, 17, 20, 29, 32, 29, 32, 49, 52, 49, 52, 61, 64, 61, 64, 49, 52, 49, 52, 61, 64, 61, 64, 17, 20, 17, 20, 29, 32, 29, 32, 17, 20, 17, 20, 29, 32, 29, 32, 49, 52, 49, 52, 61, 64, 61, 64, 49, 52, 49, 52, 61, 64, 61, 64]
# , [6, 8, 6, 8, 14, 16, 14, 16, 6, 8, 6, 8, 14, 16, 14, 16, 50, 52, 50, 52, 58, 60, 58, 60, 50, 52, 50, 52, 58, 60, 58, 60, 6, 8, 6, 8, 14, 16, 14, 16, 6, 8, 6, 8, 14, 16, 14, 16, 49, 51, 49, 51, 57, 59, 57, 59, 49, 51, 49, 51, 57, 59, 57, 59]
# , [5, 8, 5, 8, 13, 16, 13, 16, 5, 8, 5, 8, 13, 16, 13, 16, 49, 52, 49, 52, 57, 60, 57, 60, 49, 52, 49, 52, 57, 60, 57, 60, 5, 8, 5, 8, 13, 16, 13, 16, 5, 8, 5, 8, 13, 16, 13, 16, 49, 52, 49, 52, 57, 60, 57, 60, 49, 52, 49, 52, 57, 60, 57, 60]
# , [2, 4, 2, 4, 14, 16, 14, 16, 2, 4, 2, 4, 14, 16, 14, 16, 50, 52, 50, 52, 62, 64, 62, 64, 50, 52, 50, 52, 62, 64, 62, 64, 2, 4, 2, 4, 14, 16, 14, 16, 2, 4, 2, 4, 14, 16, 14, 16, 49, 51, 49, 51, 61, 63, 61, 63, 49, 51, 49, 51, 61, 63, 61, 63]
# , [1, 4, 1, 4, 13, 16, 13, 16, 1, 4, 1, 4, 13, 16, 13, 16, 49, 52, 49, 52, 61, 64, 61, 64, 49, 52, 49, 52, 61, 64, 61, 64, 1, 4, 1, 4, 13, 16, 13, 16, 1, 4, 1, 4, 13, 16, 13, 16, 49, 52, 49, 52, 61, 64, 61, 64, 49, 52, 49, 52, 61, 64, 61, 64]
]

# 调用函数
# final_matrix, is_logical = boolean_operations(*blocks)



# lists = [
#     [2, 3, 1, 4],
#     [3, 1, 4, 2],
#     [1, 4, 2, 3]
# ]

# probabilities = [0.125]*8

# weighted_matrix = weighted_logical_sum(blocks, probabilities)
# print("加权逻辑矩阵的和:\n", weighted_matrix)

# A = np.array([[1, 1], [1, 0]])  # 示例矩阵
# n = 5  # 幂次

# result = np.linalg.matrix_power(A, n)
# print(result)


# powered_matrix = np.linalg.matrix_power(weighted_matrix, 143)  
# print(powered_matrix)

# print("row number of 1 in diagonal:", find_rows_with_ones_on_diagonal(powered_matrix))
# print((np.linalg.matrix_power(weighted_matrix, 7))[108])




# import numpy as np

# def list_to_matrix(lst):
#     """将列表转换为逻辑矩阵"""
#     n = len(lst)
#     matrix = np.zeros((n, n), dtype=int)
#     for i, val in enumerate(lst):
#         matrix[i, val - 1] = 1  # 生成单位矩阵风格的逻辑矩阵
    
#     return (np.array(matrix)).T

# def matrix_to_list(matrix):
#     """将矩阵转换回列表"""
#     return [np.argmax(row) + 1 for row in (np.array(matrix)).T]

# def compute_matrix_powers(lst):
#     """计算逻辑矩阵的幂直到不再变化"""
#     matrix = list_to_matrix(lst)
#     origin_matrix = matrix.copy()
#     prev_matrix = None
#     powers = []
#     i=0
#     while not np.array_equal(matrix, prev_matrix):
#         prev_matrix = matrix.copy()
#         powers.append(matrix_to_list(matrix))
#         matrix = np.dot(origin_matrix, matrix)  # 进行普通矩阵乘法

#     return powers

# # 示例用法
# lst = [41, 47, 41, 47, 53, 59, 53, 59, 65, 71, 65, 71, 41, 47, 41, 47, 54, 60, 54, 60, 65, 71, 65, 71, 41, 47, 41, 47, 53, 59, 53, 59, 65, 71, 65, 71, 109, 111, 109, 111, 122, 124, 122, 124, 133, 135, 133, 135, 110, 112, 110, 112, 122, 124, 122, 124, 133, 135, 133, 135, 109, 111, 109, 111, 121, 123, 121, 123, 133, 135, 133, 135, 41, 47, 41, 47, 54, 60, 54, 60, 65, 71, 65, 71, 42, 48, 42, 48, 54, 60, 54, 60, 65, 71, 65, 71, 41, 47, 41, 47, 53, 59, 53, 59, 65, 71, 65, 71, 109, 111, 109, 111, 122, 124, 122, 124, 133, 135, 133, 135, 110, 112, 110, 112, 122, 124, 122, 124, 133, 135, 133, 135, 109, 111, 109, 111, 121, 123, 121, 123, 133, 135, 133, 135]


# powers = compute_matrix_powers(lst)
# for i, power in enumerate(powers, 1):
#     print(f"s^{i}: {power}")






# payoffs
# [[ 4.  3. 10.  9.  4.  3. 15. 17. 13. 15. 11. 13.]
#  [ 6.  8. 11. 13. 10. 13. 14. 16.  9. 11.  9. 12.]
#  [15.  9.  9. 12. 18.  4.  9.  6.  3.  9. 12.  1.]]



# 发现如果选取模式5，即玩家1更新，玩家2不更新，此时居然不收敛，八个矩阵中唯一个不收敛的


