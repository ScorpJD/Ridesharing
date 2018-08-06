# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:07:55 2018

@author: Dylan
"""

import numpy as np
import random
import copy


# random.seed(1)

class Node(object):
    '''
    顾客点类：
    c_id:Number,顾客点编号
    x:Number,点的横坐标
    y:Number,点的纵坐标
    demand:Number,点的需求量
    ready_time:Number,点的最早访问时间
    due_time:Number,点的最晚访问时间
    service_time:Number,点的服务时间
    belong_veh:所属车辆编号
    '''

    def __init__(self, c_id, x, y, demand, ready_time, due_time, service_time):
        self.c_id = c_id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.belong_veh = None


class Vehicle(object):
    '''
    车辆类：
    v_id:Number,车辆编号
    cap:Number,车的最大载重量
    load:Number,车的载重量
    distance:Number,车的行驶距离
    violate_time:Number,车违反其经过的各点时间窗时长总和
    route:List,车经过的点index的列表
    start_time:List,车在每个点的开始服务时间
    '''

    def __init__(self, v_id: int, cap: int):
        self.v_id = v_id
        self.cap = cap
        self.load = 0
        self.distance = 0
        self.violate_time = 0
        self.route = [0]
        self.start_time = [0]

    # 插入节点
    def insert_node(self, node: int, index: int = 0) -> None:
        if index == 0:
            self.route.append(node)
        else:
            self.route.insert(index, node)
        # node.belong_veh = self.v_id
        self.update_info()

    # 根据索引删除节点
    def del_node_by_index(self, index: int) -> None:
        self.route.pop(index)
        self.update_info()

    # 根据对象删除节点
    def del_node_by_node(self, node: Node) -> None:
        self.route.remove(node.c_id)
        self.update_info()

    # 更新载重、距离、开始服务时间、时间窗违反
    def update_info(self) -> None:
        # 更新载重
        cur_load = 0
        for n in self.route:
            cur_load += nodes[n].demand
        self.load = cur_load
        # 更新距离
        cur_distance = 0
        for i in range(len(self.route) - 1):
            cur_distance += distance_matrix[self.route[i]][self.route[i + 1]]
        self.distance = cur_distance
        # 更新违反时间窗时长总和(硬时间窗,早到等待，不可晚到)
        arrival_time = 0
        self.start_time = [0]
        cur_violate_time = 0
        for i in range(1, len(self.route)):
            arrival_time += distance_matrix[self.route[i - 1]][self.route[i]] + nodes[self.route[i - 1]].service_time
            if arrival_time > nodes[self.route[i]].due_time:
                cur_violate_time += arrival_time - nodes[self.route[i]].due_time
            elif arrival_time < nodes[self.route[i]].ready_time:
                arrival_time = nodes[self.route[i]].ready_time
            self.start_time.append(arrival_time)
        self.violate_time = cur_violate_time

    def __str__(self):  # 重载print()
        routes = [n for n in self.route]
        return '车{}:距离[{:.4f}];载重[{}];时间违反[{:.4f}]\n路径{}\n开始服务时间{}\n'.format(self.v_id, self.distance, self.load,
                                                                             self.violate_time, routes, self.start_time)


# 读取数据文件，返回车辆最大载重，最大车辆数，所有Node组成的列表
def read_data(path: str) -> (int, int, list):
    with open(path, 'r', ) as f:
        lines = f.readlines()
    capacity = (int)(lines[4].split()[-1])
    max_vehicle = (int)(lines[4].split()[0])
    lines = lines[9:]
    nodes = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 7:
            node = Node(*info)  #initialize vehicle node
            nodes.append(node)  #vehicle array
    return capacity, max_vehicle, nodes


# 计算距离矩阵
def cal_distance_matrix(nodes: list) -> np.array:
    distance_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if i != j:
                dis = np.sqrt((nodes[i].x - nodes[j].x) ** 2 + (nodes[i].y - nodes[j].y) ** 2)  #euclidean distance
                distance_matrix[i][j] = distance_matrix[j][i] = dis
    return distance_matrix


# 计算一个解的目标函数值，包括惩罚部分:f(vehicles) = distance + p_l*Q +p_t*T
def cal_obj(vehicles: list, p_l: float = 1, p_t: float = 1) -> float:
    distance = sum([v.distance for v in vehicles])
    T = sum([v.violate_time for v in vehicles])
    Q = sum([max(v.load - v.cap, 0) for v in vehicles])
    return distance + p_l * Q + p_t * T


# 判断一个解是否可行，并更新惩罚系数p_l和p_t
def check_feasible(vehicles: list, p_l: float = 1, p_t: float = 1, sita: float = 0.5) -> (bool, float, float):
    T = sum([v.violate_time for v in vehicles])
    Q = sum([max(v.load - v.cap, 0) for v in vehicles])
    if Q == 0 and p_l >= 0.001:
        p_l /= (1 + sita)
    elif Q != 0 and p_l <= 2000:
        p_l *= (1 + sita)
    if T == 0 and p_t >= 0.001:
        p_t /= (1 + sita)
    elif T != 0 and p_t <= 2000:
        p_t *= (1 + sita)
    if T == 0 and Q == 0:
        return True, p_l, p_t
    else:
        return False, p_l, p_t


# solomon_i1:种子顾客选取策略为最远顾客
def solomon_i1(max_vehicle: int, capacity_list: list, alpha1: float = 1, lam: float = 1, mu: float = 1) -> (
list, float):
    # 传入max_vehicle,capacity_list,nodes,alpha1=1,lam=1,mu=1
    # 返回插入的解对应车组成的list和总距离
    vehicles = []
    assigned_node_id = set([0])  # 已被分配过的点的集合

    for num_veh in range(max_vehicle):  # 外层循环
        if len(assigned_node_id) == len(nodes):
            break
        # 新安排一辆车服务一条路径
        veh = Vehicle(num_veh, capacity_list[num_veh])
        # 寻找第一个插入点：未分配的点中距离原点最远的点
        max_dis = 0
        for i in range(1, len(nodes)):
            if i in assigned_node_id:
                continue
            elif max_dis < distance_matrix[0][i]:
                max_dis = distance_matrix[0][i]
                init_node_id = i
        veh.insert_node(init_node_id)  # 路径的第二个点为第一个插入点
        assigned_node_id.add(init_node_id)  # 加入已分配过的点的集合
        veh.insert_node(0)  # 将仓库0加入路径末尾

        # 为每条路插入顾客，每次至多插入len(nodes)-1个顾客
        for num_inserted in range(len(nodes) - 1):
            min_set_c1 = np.zeros((2, len(nodes) - 1));
            is_feasible = False  # 判断是否还有可以插入的点（不违反时间窗约束）
            # 该for循环是为了计算每个顾客点u的最优c1以及插入的位置
            for u in range(1, len(nodes)):
                if u in assigned_node_id:
                    min_set_c1[0][u - 1] = u - 1
                    min_set_c1[1][u - 1] = np.inf  # 已被分配过的点的min_c1设置为inf
                    continue
                else:
                    c1 = np.array([0] * (len(veh.route) - 1), dtype='float')
                    for i in range(len(veh.route) - 1):  # 计算c1_1和c1_2
                        # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
                        temp_veh = copy.copy(veh)
                        temp_veh.route = copy.copy(veh.route)
                        temp_veh.insert_node(u, i + 1)

                        if temp_veh.violate_time != 0:
                            c1[i] = np.inf
                        else:
                            c1_1 = distance_matrix[veh.route[i]][u] + \
                                   distance_matrix[u][veh.route[i + 1]] - \
                                   mu * distance_matrix[veh.route[i]][veh.route[i + 1]]
                            c1_2 = temp_veh.start_time[i + 2] - veh.start_time[i + 1]
                            c1[i] = alpha1 * c1_1 + (1 - alpha1) * c1_2
                            is_feasible = True

                    # 获得一个点插入的所有位置中c1增量的最小值和插入的位置
                    min_c1 = np.min(c1)
                    min_c1_index = np.argmin(c1)
                    min_set_c1[0][u - 1] = min_c1_index  # 第一行存储第u个点的最佳插入位置
                    min_set_c1[1][u - 1] = min_c1  # 第二行存储第u个点的最小c1增量

            if is_feasible == False:  # 该条路径没有可以插入的点，则结束该条路
                vehicles.append(veh)
                break

            # 计算每个顾客点的c2
            c2 = [lam * distance_matrix[0][u] - min_set_c1[1][u - 1] for u in range(1, len(nodes))]
            # 找出最大c2对应的顾客点和插入位置
            insert_node_id = c2.index(max(c2)) + 1
            insert_position = min_set_c1[0][c2.index(max(c2))] + 1
            # 插入上述点
            veh.insert_node(insert_node_id, int(insert_position))
            # 判断载重约束
            if veh.load > veh.cap:
                veh.del_node_by_node(nodes[insert_node_id])
                vehicles.append(veh)
                break
            else:
                assigned_node_id.add(insert_node_id)

    # 为每个顾客点添加所属车编号
    for v in vehicles:
        for i in range(1, len(v.route) - 1):
            nodes[v.route[i]].belong_veh = v.v_id

    return vehicles, sum([v.distance for v in vehicles])


# APRCH:自适应并行算法(没有调整参数)
def aprch(capacity: int, w_dis=0.6, w_urg=0.11) -> (list, float):
    # 输入capacity,w_d,w_u,w_w
    # 返回插入的解对应车组成的list和总距离
    w_wai = 1 - w_dis - w_urg
    FAILED = 1e6
    assigned_node_id = set()  # 已被分配过的点的集合
    num_veh_least = int(sum([n.demand for n in nodes]) / capacity) + 1
    vehicles = [Vehicle(num_veh, capacity) for num_veh in range(num_veh_least)]
    # 主循环
    while len(assigned_node_id) < len(nodes) - 1:
        fit_all = np.ones((len(vehicles), len(nodes) - 1)) * np.inf
        # 计算每辆车末尾插入每个点的fitness
        for k in vehicles:
            for n in range(1, len(nodes)):
                if n not in assigned_node_id:
                    fit_dis = distance_matrix[k.route[-1]][n]   # try add user[n] into vehicle[k]'s route & fit_dis is the distance of the new adding route
                    # suppose the average speed is 1 so the new time of fit_dis is fit_dis itself
                    # fit_urg is the remaining time before the due_time of user[n] if add him into vehicle[k]
                    fit_urg = nodes[n].due_time - (k.start_time[-1] + nodes[k.route[-1]].service_time + fit_dis)
                    # fit_wai is the value of the maximum of the possible remaining time
                    fit_wai = max(0,
                                  nodes[n].ready_time - (k.start_time[-1] + nodes[k.route[-1]].service_time + fit_dis))
                    if fit_urg >= 0 and k.load + nodes[n].demand <= capacity:
                        # 因n从1开始，所以存储索引为 n-1
                        fit_all[k.v_id][n - 1] = w_dis * fit_dis + w_urg * fit_urg + w_wai * fit_wai
                    else:
                        fit_all[k.v_id][n - 1] = FAILED

        # 判断是否需要新开一辆车
        for row in range(len(vehicles)):
            # 因n从1开始，所以实际点编号为存储索引 + 1
            node_veh_failed = set(np.ravel(np.argwhere(fit_all[row] == FAILED)) + 1)
            if row == 0:
                nodes_failed = node_veh_failed
            else:
                nodes_failed = nodes_failed & node_veh_failed
        if len(nodes_failed) > 0:
            # 新加入一辆车
            vehicles.append(Vehicle(num_veh_least, capacity))
            num_veh_least += 1
            fit_all = np.ones((len(vehicles), len(nodes) - 1)) * np.inf
            # 重新计算每辆车末尾插入每个点的fitness
            for k in vehicles:
                for n in range(1, len(nodes)):
                    if n not in assigned_node_id:
                        fit_dis = distance_matrix[k.route[-1]][n]
                        fit_urg = nodes[n].due_time - (k.start_time[-1] + nodes[k.route[-1]].service_time + fit_dis)
                        fit_wai = max(0, nodes[n].ready_time - (
                                    k.start_time[-1] + nodes[k.route[-1]].service_time + fit_dis))
                        if fit_urg >= 0 and k.load + nodes[n].demand:
                            # 因n从1开始，所以存储索引为 n-1
                            fit_all[k.v_id][n - 1] = w_dis * fit_dis + w_urg * fit_urg + w_wai * fit_wai
                        else:
                            fit_all[k.v_id][n - 1] = FAILED

        # 选择最优的插入车和点
        veh_index, node_index = np.where(fit_all == np.min(fit_all))
        veh_index = veh_index[0]
        # 点编号 = 其存储索引 + 1
        node_index = node_index[0] + 1
        vehicles[veh_index].insert_node(node_index)
        assigned_node_id.add(node_index)

    # 为每条路末尾加上0
    for k in vehicles:
        k.insert_node(0)
        # 为每个顾客点更新所属车编号
    for v in vehicles:
        for i in range(1, len(v.route) - 1):
            nodes[v.route[i]].belong_veh = v.v_id
    return vehicles, sum([v.distance for v in vehicles])


# 禁忌搜索
# 输入：初始解，点，禁忌长度，迭代次数，最优解不变跳出代数
def tabu_search(cur_vehs: list, tabu_tenure: int = 20, iter_max=200, p_l=1, p_t=1) -> (list, float):
    iter_break = 0.2 * iter_max
    count_break = 0
    iteration = 0
    num_nodes = len(nodes) # nodes: 顾客表
    tabu = np.zeros((num_nodes, len(cur_vehs)))  # 禁忌表存储将点c插入到车v的操作对
    # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
    global_best_vehs = [copy.copy(cur_vehs[i]) for i in range(len(cur_vehs))]
    for i in range(len(cur_vehs)):
        global_best_vehs[i].route = copy.copy(cur_vehs[i].route)
    global_best_obj = cal_obj(global_best_vehs)  # 当前解得penalized objective function value
    while iteration < iter_max:        #当循环次数还未达到上限的时候
        iteration += 1
        neigbor_best_obj = neigbor_best_no_tabu_obj = np.inf  # 邻域最优解obj， 初始值设为np.inf最大值
        # 最优的插入顾客编号、插入车辆编号
        best_cus = best_veh = 0
        # 没被禁忌的最优的插入顾客编号、插入车辆编号
        best_cus_no_tabu = best_veh_no_tabu = 0
        cur_pos = 0  # 每个点在当前车所属位置
        # 计算每个点插入每个其他车的所有位置时候的目标函数值
        rand = random.randint(0, 10)
        for c in range(1, num_nodes):
            for i in range(1, len(cur_vehs[nodes[c].belong_veh].route)):  # 寻找插入点在当前所在车中的位置
                if cur_vehs[nodes[c].belong_veh].route[i] == c:
                    cur_pos = i
                    break
            for k in range(len(cur_vehs)):  # 遍历每一辆车，选择插入位置
                     for p in range(1, len(cur_vehs[k].route)):
                        cur_vehs[k].insert_node(c, p)  # 插入点
                        cur_vehs[nodes[c].belong_veh].del_node_by_node(nodes[c])  # 删除点
                        inserted_obj = cal_obj(cur_vehs, p_l, p_t)  # 插入后的目标函数值
                        if inserted_obj < neigbor_best_obj:      #tabu moves but requires improving the best obj
                            neigbor_best_obj = inserted_obj
                            best_cus = c
                            best_veh = k
                            best_pos = p
                            ori_pos = cur_pos
                        if inserted_obj < neigbor_best_no_tabu_obj and tabu[c][k] < iteration:      #non tabu moves which requires the tabu list value
                            neigbor_best_no_tabu_obj = inserted_obj
                            best_cus_no_tabu = c
                            best_veh_no_tabu = k
                            best_pos_no_tabu = p
                        # 还原cur_vehs
                        cur_vehs[nodes[c].belong_veh].insert_node(c, cur_pos)
                        cur_vehs[k].del_node_by_node(nodes[c])
        # 判断最优邻域解是否可行，并更新惩罚系数
        cur_vehs[best_veh].insert_node(best_cus, best_pos)  # 插入点
        cur_vehs[nodes[best_cus].belong_veh].del_node_by_node(nodes[best_cus])  # 删除点
        status, p_l, p_t = check_feasible(cur_vehs, p_l, p_t)
        if neigbor_best_obj < global_best_obj and status:
            # 删除长度为2的空路径
            for v in cur_vehs:
                if len(v.route) <= 2:
                    cur_vehs.remove(v)
            global_best_obj = neigbor_best_obj
            # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
            global_best_vehs = [copy.copy(cur_vehs[i]) for i in range(len(cur_vehs))]
            for i in range(len(cur_vehs)):
                global_best_vehs[i].route = copy.copy(cur_vehs[i].route)
            tabu[best_cus][nodes[best_cus].belong_veh] = iteration + tabu_tenure + rand
            nodes[best_cus].belong_veh = best_veh  # 更新被操作的节点所属车编号
            count_break = 0
        else:
            # 还原cur_vehs
            cur_vehs[nodes[best_cus].belong_veh].insert_node(best_cus, ori_pos)
            cur_vehs[best_veh].del_node_by_node(nodes[best_cus])
            # 以未被禁忌的最优邻域解更新cur_vehs
            cur_vehs[best_veh_no_tabu].insert_node(best_cus_no_tabu, best_pos_no_tabu)  # 插入点
            cur_vehs[nodes[best_cus_no_tabu].belong_veh].del_node_by_node(nodes[best_cus_no_tabu])  # 删除点
            tabu[best_cus_no_tabu][nodes[best_cus_no_tabu].belong_veh] = iteration + tabu_tenure + rand
            nodes[best_cus_no_tabu].belong_veh = best_veh_no_tabu  # 更新被操作的节点所属车编号
            count_break += 1
        # 连续一定代数不改变全局最优解，则停止迭代
        if count_break == iter_break:
            break
        # 更新禁忌表
    #        tabu = np.where(tabu > 0,tabu-1,tabu)
    # 为每个顾客点更新所属车编号
    for v in global_best_vehs:
        for i in range(1, len(v.route) - 1):
            nodes[v.route[i]].belong_veh = v.v_id

    return global_best_vehs, global_best_obj


# 交换一条路内的两个顾客
def exchange_one_route(vehicle: Vehicle, node_pos1: int, node_pos2: int) -> None:
    vehicle.route[node_pos1], vehicle.route[node_pos2] = vehicle.route[node_pos2], vehicle.route[node_pos1]
    vehicle.update_info()


# 交换两条路间的两个顾客
def exchange_two_route(vehicle1: Vehicle, node_pos1: int, vehicle2: Vehicle, node_pos2: int) -> None:
    vehicle1.route[node_pos1], vehicle2.route[node_pos2] = vehicle2.route[node_pos2], vehicle1.route[node_pos1]
    vehicle1.update_info()
    vehicle2.update_info()


# 交换两条不同路径中两个客户后面的部分
def cross_exchange(vehicle1: Vehicle, node_pos1: int, vehicle2: Vehicle, node_pos2: int) -> None:
    node_list1 = vehicle1.route[node_pos1 + 1:]
    node_list2 = vehicle2.route[node_pos2 + 1:]
    for i in range(len(node_list1)):
        vehicle1.route.pop()
    for i in range(len(node_list2)):
        vehicle2.route.pop()
    for i in range(len(node_list1)):
        vehicle2.insert_node(node_list1[i])
    for i in range(len(node_list2)):
        vehicle1.insert_node(node_list2[i])
    vehicle1.update_info()
    vehicle2.update_info()


# 变邻域搜索结合模拟退火 vns_sa
# 输入：初始解，点，禁忌长度，禁忌搜索迭代次数
def vns_sa(cur_vehs: list, tabu_tenure: int = 20, tabu_iter_max: int = 200) -> (list, float):
    global_T = 100
    cool_cof = 0.4
    neigbor_num = 1
    T = global_T
    # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
    global_best_vehs = [copy.copy(cur_vehs[i]) for i in range(len(cur_vehs))]
    for i in range(len(cur_vehs)):
        global_best_vehs[i].route = copy.copy(cur_vehs[i].route)
    global_best_obj = cal_obj(global_best_vehs)

    # cross_exchange:交换两条不同路径中两个客户后面的部分
    while neigbor_num == 1:
        cur_obj = cal_obj(cur_vehs)
        cur_vehs_num = len(cur_vehs)
        rand_veh1 = random.randint(0, cur_vehs_num - 1)
        while len(cur_vehs[rand_veh1].route) <= 2:  # 对路径长度小于等于2的无需操作
            rand_veh1 = random.randint(0, cur_vehs_num - 1)
        rand_veh2 = random.randint(0, cur_vehs_num - 1)
        while len(cur_vehs[rand_veh2].route) <= 2 or rand_veh1 == rand_veh2:
            rand_veh2 = random.randint(0, cur_vehs_num - 1)
        choosed_veh_len1 = len(cur_vehs[rand_veh1].route)
        rand_node1 = random.randint(0, choosed_veh_len1 - 2)
        choosed_veh_len2 = len(cur_vehs[rand_veh2].route)
        rand_node2 = random.randint(0, choosed_veh_len2 - 2)
        while rand_node1 + rand_node2 == 0:  # 不能两个都是0
            rand_node2 = random.randint(0, choosed_veh_len2 - 2)
            # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
        temp_vehs = [copy.copy(cur_vehs[i]) for i in range(len(cur_vehs))]
        for i in range(len(cur_vehs)):
            temp_vehs[i].route = copy.copy(cur_vehs[i].route)
        cross_exchange(cur_vehs[rand_veh1], rand_node1, cur_vehs[rand_veh2], rand_node2)
        # 更新每个顾客点的所属车编号
        for v in cur_vehs:
            for i in range(1, len(v.route) - 1):
                nodes[v.route[i]].belong_veh = v.v_id
        # 禁忌搜索
        local_vehs, local_obj = tabu_search(cur_vehs, tabu_tenure, tabu_iter_max)
        status, p_l, p_t = check_feasible(local_vehs)
        # Improved or not?
        if local_obj < global_best_obj and status:
            global_best_obj = local_obj
            # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
            global_best_vehs = [copy.copy(local_vehs[i]) for i in range(len(local_vehs))]
            for i in range(len(local_vehs)):
                global_best_vehs[i].route = copy.copy(local_vehs[i].route)
        deta_obj = local_obj - cur_obj
        # Move or not? Use Metropolis probability
        if deta_obj <= 0 and status:
            cur_vehs = local_vehs
            # 为每个顾客点更新所属车编号
            for v in cur_vehs:
                for i in range(1, len(v.route) - 1):
                    nodes[v.route[i]].belong_veh = v.v_id
        else:
            rand_prob = random.random()
            allow_prob = np.exp(-deta_obj / T)
            if rand_prob <= allow_prob and status:
                cur_vehs = local_vehs
            else:
                neigbor_num += 1
                cur_vehs = temp_vehs
                # 为每个顾客点更新所属车编号
                for v in cur_vehs:
                    for i in range(1, len(v.route) - 1):
                        nodes[v.route[i]].belong_veh = v.v_id
        T *= cool_cof

    T = global_T
    # exchange_two_route:随机交换两条路上的任意两个点
    while neigbor_num == 2:
        cur_obj = cal_obj(cur_vehs)
        cur_vehs_num = len(cur_vehs)
        rand_veh1 = random.randint(0, cur_vehs_num - 1)
        while len(cur_vehs[rand_veh1].route) <= 2:  # 对路径长度小于等于2的无需操作
            rand_veh1 = random.randint(0, cur_vehs_num - 1)
        rand_veh2 = random.randint(0, cur_vehs_num - 1)
        while len(cur_vehs[rand_veh2].route) <= 2 or rand_veh1 == rand_veh2:
            rand_veh2 = random.randint(0, cur_vehs_num - 1)
        choosed_veh_len1 = len(cur_vehs[rand_veh1].route)
        rand_node1 = random.randint(1, choosed_veh_len1 - 2)
        choosed_veh_len2 = len(cur_vehs[rand_veh2].route)
        rand_node2 = random.randint(1, choosed_veh_len2 - 2)
        # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
        temp_vehs = [copy.copy(cur_vehs[i]) for i in range(len(cur_vehs))]
        for i in range(len(cur_vehs)):
            temp_vehs[i].route = copy.copy(cur_vehs[i].route)
        exchange_two_route(cur_vehs[rand_veh1], rand_node1, cur_vehs[rand_veh2], rand_node2)
        # 更新每个顾客点的所属车编号
        for v in cur_vehs:
            for i in range(1, len(v.route) - 1):
                nodes[v.route[i]].belong_veh = v.v_id
        # 禁忌搜索
        local_vehs, local_obj = tabu_search(cur_vehs, tabu_tenure, tabu_iter_max)
        status, p_l, p_t = check_feasible(local_vehs)
        # Improved or not?
        if local_obj < global_best_obj and status:
            global_best_obj = local_obj
            # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
            global_best_vehs = [copy.copy(local_vehs[i]) for i in range(len(local_vehs))]
            for i in range(len(local_vehs)):
                global_best_vehs[i].route = copy.copy(local_vehs[i].route)
        deta_obj = local_obj - cur_obj
        # Move or not? Use Metropolis probability
        if deta_obj <= 0 and status:
            cur_vehs = local_vehs
            # 为每个顾客点更新所属车编号
            for v in cur_vehs:
                for i in range(1, len(v.route) - 1):
                    nodes[v.route[i]].belong_veh = v.v_id
        else:
            rand_prob = random.random()
            allow_prob = np.exp(-deta_obj / T)
            if rand_prob <= allow_prob and status:
                cur_vehs = local_vehs
                # 为每个顾客点更新所属车编号
                for v in cur_vehs:
                    for i in range(1, len(v.route) - 1):
                        nodes[v.route[i]].belong_veh = v.v_id
            else:
                neigbor_num += 1
                cur_vehs = temp_vehs
                # 为每个顾客点更新所属车编号
                for v in cur_vehs:
                    for i in range(1, len(v.route) - 1):
                        nodes[v.route[i]].belong_veh = v.v_id
        T *= cool_cof

    T = global_T
    # exchange_one_route:只随机交换一条路上相邻的两个点
    while neigbor_num == 3:
        cur_obj = cal_obj(cur_vehs)
        cur_vehs_num = len(cur_vehs)
        rand_veh = random.randint(0, cur_vehs_num - 1)
        while len(cur_vehs[rand_veh].route) <= 4:  # 对路径长度小于等于4的无需操作
            rand_veh = random.randint(0, cur_vehs_num - 1)
        choosed_veh_len = len(cur_vehs[rand_veh].route)
        rand_node = random.randint(1, choosed_veh_len - 3)
        # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
        temp_vehs = [copy.copy(cur_vehs[i]) for i in range(len(cur_vehs))]
        for i in range(len(cur_vehs)):
            temp_vehs[i].route = copy.copy(cur_vehs[i].route)
        exchange_one_route(cur_vehs[rand_veh], rand_node, rand_node + 1)
        # 禁忌搜索
        local_vehs, local_obj = tabu_search(cur_vehs, tabu_tenure, tabu_iter_max)
        status, p_l, p_t = check_feasible(local_vehs)
        # Improved or not?
        if local_obj < global_best_obj and status:
            global_best_obj = local_obj
            # 双重潜拷贝(copy)实现深拷贝(deepcopy),提升性能
            global_best_vehs = [copy.copy(local_vehs[i]) for i in range(len(local_vehs))]
            for i in range(len(local_vehs)):
                global_best_vehs[i].route = copy.copy(local_vehs[i].route)
        # Move or not? Use Metropolis probability
        deta_obj = local_obj - cur_obj
        if deta_obj <= 0 and status:
            cur_vehs = local_vehs
            # 为每个顾客点更新所属车编号
            for v in cur_vehs:
                for i in range(1, len(v.route) - 1):
                    nodes[v.route[i]].belong_veh = v.v_id
        else:
            rand_prob = random.random()
            allow_prob = np.exp(-deta_obj / T)
            if rand_prob <= allow_prob and status:
                cur_vehs = local_vehs
                # 为每个顾客点更新所属车编号
                for v in cur_vehs:
                    for i in range(1, len(v.route) - 1):
                        nodes[v.route[i]].belong_veh = v.v_id
            else:
                neigbor_num += 1
                cur_vehs = temp_vehs

        T *= cool_cof
    # 为每个顾客点更新所属车编号
    for v in global_best_vehs:
        for i in range(1, len(v.route) - 1):
            nodes[v.route[i]].belong_veh = v.v_id

    return global_best_vehs, global_best_obj


# %%主程序
if __name__ == '__main__':
    import time

    # 读取数据
    path = r'VRPTW_solomon\solomon_100\R101.txt'
    capacity, max_vehicle, nodes = read_data(path)  # 获取车的载重，最大车数，顾客节点
    distance_matrix = cal_distance_matrix(nodes)  # 将距离矩阵赋值给车辆的类变量

    #%%solomon构造初始解
    capacity_list =[capacity]*max_vehicle
    start = time.clock()
    vehicles_sol,distance_sol = solomon_i1(max_vehicle,capacity_list)
    end = time.clock()
    print('Solomon耗时{:.2f}'.format(end-start))
    print('Solomon耗时最优解{:.6f}'.format(distance_sol))

    # # %%aprch构造初始解
    # start = time.clock()
    # vehicles_aprch, distance_aprch = aprch(capacity, 0.69, 0.11)
    # end = time.clock()
    # print('Aprch耗时{:.2f}'.format(end - start))
    # print('Aprch耗时最优解{:.6f}'.format(distance_aprch))

   #%%TS
    tabu_tenure = 20
    tabu_iter_max = 200
    start = time.clock()
    vehicles_ts,distance_ts = tabu_search(vehicles_sol,tabu_tenure,tabu_iter_max)
    end = time.clock()
    print('TS耗时{:.2f}'.format(end-start))
    print('TS最优解{:.6f}'.format(distance_ts))

#    #%%VNS
#    tabu_tenure = 20
#    tabu_iter_max = 200
#    start = time.clock()
#    vehicles_vns,distance_vns = vns_sa(vehicles_sol,tabu_tenure,tabu_iter_max)
#    end = time.clock()
#    print('VNS耗时{:.2f}s'.format(end-start))
#    print('VNS最优解{:.6f}'.format(distance_vns))