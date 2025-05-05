作者发现，在使用半张量积对玩家的策略相乘从而表示成局势时，状态空间中的数字按大小排列等价于将局势按字典序排列
举个栗子
    有两个玩家，每个玩家两个策略，令他们为1，2
    此时，所有局势按字典序从小到大排列为11，12，21，22
    而他们相当于状态空间中的1，2，3，4
因此，作者在这里跳过了半张量积中的换位，降幂等算子，直接利用状态空间中的状态(数字)对应的局势信息，
根据设定的SUR，来计算下一时刻将要跳转的状态。
Game.py的核心思路就是这样，
至于其它的几个模块，主要是一些辅助工作，如画出状态轨迹图，利用BFS寻找最短路径，以及生成随机序列模拟演化过程等


Game.py 是博弈模型构建，然后最后生成各个异步矩阵，虽然被设计用于带记忆的，但其实只要在参数设置时把记忆长度改为1
就能够用于普通的一步记忆

Draft.py 是用来画某一个特定更新序列下的状态轨迹图

ControllerGenerator.py 是用来生成一个最佳的异步更新序列，并基于这个更新序列给出状态转移图

AverageStable.py 生成随机序列，并计算随机更新到达不懂点的最短长度，平均长度等


The author found that when multiplying the strategies of players using the semi-tensor product to represent the situation, 
arranging the numbers in the state space in ascending order is equivalent to listing the situation in lexicographical order. 
For example, there are two players, each with two strategies. Let them be 1,2. 
    At this time, all the situations are arranged in lexicographical order from smallest to largest as 11,12,21. 22 And they are equivalent to 1,2,3,4 in the state space. 
    Therefore, the author skips the transposition and idempotence operators in the semi-tensor product here and directly uses the situation information corresponding to the states (numbers) 
    in the state space to calculate the state to be jumped to at the next moment based on the set SUR. The core idea of Game.py is like this. 

As for the other several modules, they are mainly some auxiliary tasks, such as drawing state trajectory diagrams, 
using BFS to find the shortest path, and generating random sequences to simulate the evolution process, etc. 

Game.py is about constructing Game models and finally generating various asynchronous matrices, although it is designed for memory-driven ones. 
But in fact, as long as the memory length is changed to 1 when setting the parameters, it can be used for ordinary one-step memory. 

Draft.py is used to draw the state trajectory diagram under a specific update sequence. 

ControllerGenerator.py is used to generate an optimal asynchronous update sequence. 

Based on this update sequence, the state transition graph AverageStable.py is given to generate a random sequence, and the shortest length, average length, etc. 
for the random update to reach the fixed point are calculated
