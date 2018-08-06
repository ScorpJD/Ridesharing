import numpy as np
import copy
import time

class Vertex(object):
    def __init__(self, v_id, x, y):
        self.v_id = v_id
        self.x = x
        self.y = y

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

    def __init__(self, c_id, ov, dv, ready_time, due_time, pat):
        self.c_id = c_id
        self.ov = ov
        self.dv = dv
        self.demand = 1
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = 0
        self.tag = 0
        self.pat = pat
        self.belong_veh = -1

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

    def __init__(self, v_id: int, ov, dv, edt, ldt, pat, cap: int, max_time: int):
        self.v_id = v_id
        self.ov = ov
        self.dv = dv
        self.edt = edt
        self.ldt = ldt
        self.pat = pat
        self.cap = cap
        self.start_service_time = 0
        self.load = 0
        self.violate_load = 0
        self.distance = 0
        self.violate_time = 0
        self.pickup_violate_time = 0
        self.dropoff_violate_time = 0
        self.route = []
        self.dropoff_route =[]
        self.start_time = [edt]
        self.end_time = []
        self.o_additional = {}
        self.d_additional = {}
        self.veh_time = 0
        self.max_time = max_time

    # 插入节点
    def insert_node(self, node: int, index: int = -1) -> None:
        if index == -1:
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

    def insert_o_additional(self,node1: int,node2: int) -> None:
        self.o_addtional[node1] = node2
        self.update_info()

    def del_o_additional(self,node1: int) -> Node:
        if node1 in self.o_addtional.keys():
            del self.o_addtional[node1]
            self.update_info()

    def insert_d_additional(self, node1: int, node2, int) -> None:
        self.d_addtional[node1] = node2
        self.update_info()

    def del_d_additional(self,node1: int) -> None:
        if node1 in self.d_addtional:
            del self.d_addtional[node1]
            self.update_info()

    def insert_dropoff_node(self,node: int, index: int = -1) -> None:
        if index == -1:
            self.dropoff_route.append(node)
        else:
            self.dropoff_route.insert(index, node)
        self.update_info()

    def del_dropoff_node(self,index: int = -1) -> None:
        self.dropoff_route.pop(index)
        self.update_info()

    def del_dropoff_node_by_node(self, node: Node) -> Node:
        self.dropoff_route.remove(node.c_id)
        self.update_info()

    # 更新载重、距离、开始服务时间、时间窗违反
    def update_info(self) -> None:
        # 更新载重
        cur_load = 0
        for n in range(len(self.route)):
            cur_load += nodes[self.route[n]].demand
        self.load = cur_load
        if self.load > self.cap:
            self.violate_load = self.load - self.cap
        # 更新距离
        cur_distance = 0
        for i in range(len(self.route) - 1):
            cur_distance += distance_matrix[nodes[self.route[i]].ov][nodes[self.route[i + 1]].ov]
        self.distance = cur_distance
        # 更新违反时间窗时长总和(硬时间窗,早到等待，不可晚到)
        self.start_time = [self.edt]
        # self.start_time = [0]
        cur_pickup_violate_time = 0
        if len(self.route) != 0:
            # return
            arrival_time = distance_matrix[nodes[self.route[0]].ov][self.ov] + self.start_time[0]
            if arrival_time > nodes[self.route[0]].due_time:
                cur_pickup_violate_time += arrival_time - nodes[self.route[0]].due_time
            elif arrival_time <= nodes[self.route[0]].ready_time:
                arrival_time = nodes[self.route[0]].ready_time
                self.start_time[0] = arrival_time - distance_matrix[nodes[self.route[0]].ov][self.ov]
            self.start_time.append(arrival_time)
            nodes[self.route[0]].service_time = arrival_time
            for i in range(1, len(self.route)):
                arrival_time = distance_matrix[nodes[self.route[i - 1]].ov][nodes[self.route[i]].ov] + \
                                nodes[self.route[i - 1]].service_time
                if arrival_time > nodes[self.route[i]].due_time:
                    cur_pickup_violate_time += arrival_time - nodes[self.route[i]].due_time
                elif arrival_time < nodes[self.route[i]].ready_time:
                    arrival_time = nodes[self.route[i]].ready_time
                self.start_time.append(arrival_time)
                nodes[self.route[i]].service_time = arrival_time
            self.pickup_violate_time = cur_pickup_violate_time
            if self.start_time[1] - distance_matrix[self.ov][nodes[self.route[0]].ov] > self.edt:
                self.start_time[0] = self.start_time[1] - distance_matrix[self.ov][nodes[self.route[0]].ov]
        ### shortest path problem -> sort the dropoff routes
        cur_dropoff_violate_time = 0
        self.end_time = []
        dropoff_time = self.start_time[-1]
        # print(self.route[-1])
        # print(nodes[self.route[-1]])
        # print(nodes[self.dropoff_route[0]])
        if len(self.dropoff_route) != 0 and len(self.route) != 0:
            dropoff_time += distance_matrix[nodes[self.route[-1]].ov][nodes[self.dropoff_route[0]].dv]
            if dropoff_time > nodes[self.dropoff_route[0]].pat:
                cur_dropoff_violate_time += dropoff_time - nodes[self.dropoff_route[0]].pat
            self.end_time.append(dropoff_time)
            #self.end_time[0] = dropoff_time
            for i in range(1, len(self.dropoff_route)):
                dropoff_time += distance_matrix[nodes[self.dropoff_route[i]].dv][nodes[self.dropoff_route[i - 1]].dv]
                if dropoff_time > nodes[self.dropoff_route[i]].pat:
                    cur_dropoff_violate_time += dropoff_time - nodes[self.dropoff_route[i]].pat
                self.end_time.append(dropoff_time)
            self.end_time.append(dropoff_time + distance_matrix[nodes[self.dropoff_route[-1]].dv][self.dv])
        else:
            self.end_time.append(distance_matrix[self.dv][self.ov] + self.start_time[0])
        self.veh_time = self.end_time[-1] - self.start_time[0]
        if self.veh_time > self.max_time:
            self.violate_time = self.veh_time - self.max_time


    def __str__(self):  # 重载print()
        routes = [n for n in self.route]
        return '车{}:耗时[{:.4f}];载重[{}];(驾驶)时间违反[{:.4f}]' \
               '\npickup路径{}\ndropoff路径{}\n开始服务时间{}\n结束服务时间{}\n上车前走动{}\n下车后走动{}\n\n'.\
            format(self.v_id, self.veh_time, self.load,self.violate_time,
                   self.route, self.dropoff_route, self.start_time, self.end_time, self.o_additional,self.d_additional)

def read_vertex(path: str) -> list:
    with open(path, 'r', ) as f:
        lines = f.readlines()
    # capacity = (int)(lines[4].split()[-1])
    # max_vehicle = (int)(lines[4].split()[0])
    lines = lines[1:]
    vertexes = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 3:
            vertex = Vertex(*info)
            vertexes.append(vertex)
    return vertexes

def read_vehicle(path: str) -> list:
    with open(path, 'r', ) as f:
        lines = f.readlines()
    # capacity = (int)(lines[4].split()[-1])
    # max_vehicle = (int)(lines[4].split()[0])
    lines = lines[1:]
    vehicles = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 8:
            vehicle = Vehicle(*info)
            vehicles.append(vehicle)
    return vehicles

def read_nodes(path: str) -> list:
    with open(path, 'r', ) as f:
        lines = f.readlines()
    # capacity = (int)(lines[4].split()[-1])
    # max_vehicle = (int)(lines[4].split()[0])
    lines = lines[1:]
    nodes = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 6:
            node = Node(*info)
            nodes.append(node)
    return nodes

def cal_distance(vertexs: list) -> np.array:
    distance_matrix = np.zeros((len(vertexs), len(vertexs)), )
    for i in range(len(vertexs)):
        for j in range(i + 1, len(vertexs)):
            if i != j:
                distance = np.sqrt((vertexs[i].x - vertexs[j].x) ** 2 + (vertexs[i].y - vertexs[j].y) ** 2)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    #print(distance_matrix)
    return distance_matrix

def cal_obj(veh:list, p_l, p_t, p_f, times_ck: int = 0) -> float:
    cost_time = sum([v.veh_time for v in veh])
    #print(veh)
    for i in range(len(veh)):
        # print(len(veh[i].o_additional))
        for j in veh[i].o_additional.keys():
            cost_time += distance_matrix[nodes[j].ov][nodes[veh[i].o_additional[j]].ov]
        for k in veh[i].d_additional.keys():
            cost_time += distance_matrix[nodes[k].dv][nodes[veh[i].d_addtional[j]].dv]
    Q = sum([max(v.load - v.cap, 0) for v in veh])
    T = sum([v.violate_time for v in veh])
    Phi = p_f * cost_time * len(nodes) ** 0.5 * times_ck
    return cost_time + p_l * Q + p_t * T + p_f * Phi

def check_feasible(vehicles: list, p_l: float, p_t: float, sita: float = 0.5) -> (bool, float, float):
    Q = sum([max(v.load - v.cap, 0) for v in vehicles])
    T = sum([v.violate_time for v in vehicles])
    if Q == 0 and p_l >= 0.001:
        p_l /= (1 + sita)
    elif Q != 0 and p_l <= 2000:
        p_l *= (1 + sita)
    if T == 0 and p_t >= 0.001:
        p_t /= (1 + sita)
    elif T != 0 and p_t <= 2000:
        p_t *= (1 + sita)
    if T <= 10 and Q == 0:
        return True, p_l, p_t
    else:
        return False, p_l, p_t

def send_to_nearest(node: int, assigned:list, isOv: bool) -> bool:
    # using in initial solution
    dict = {}
    for i in range(len(assigned)):
        if isOv:
            dict[assigned[i]] = distance_matrix[nodes[node].ov][nodes[assigned[i]].ov]
        else:
            dict[assigned[i]] = distance_matrix[nodes[node].dv][nodes[assigned[i]].dv]
    if isOv:
        for i in range(len(dict)):
            nearest = sorted(dict.items(), key=lambda x: x[1])[i][0]
            if vehicles[nodes[nearest].belong_veh].cap - vehicles[nodes[nearest].belong_veh].load >= 1:
                nodes[nearest].demand += 1
                vehicles[nodes[nearest].belong_veh].insert_dropoff_node(0, nodes[nearest].c_id)
            ### ????
                if vehicles[i].violate_time == 0 and vehicles[i].violate_load == 0 \
                        and vehicles[i].pickup_violate_time == 0 and vehicles[i].dropoff_violate_time == 0:
                    vehicles[i].o_additional[node] = nearest
                    nodes[node].belong_veh = nodes[nearest].belong_veh
                    break
                elif vehicles[i].violate_time == 0 and vehicles[i].violate_load == 0 \
                        and vehicles[i].pickup_violate_time == 0 and vehicles[i].dropoff_violate_time != 0:
                    vehicles[i].o_additional[node] = nearest
                    nodes[node].belong_veh = nodes[nearest].belong_veh
                    send_to_nearest(node, vehicles[i].dropoff_route, False)
                else:
                    nodes[nearest].demand -= 1
                    vehicles[nodes[nearest].belong_veh].del_dropoff_node(0)
            else:
                continue
    else:
        #for i in range(len(dict)):
        nearest = sorted(dict.items(), key=lambda x: x[1])[0][0]
        vehicles[i].d_additional[node] = nearest
            ### ????
            # if vehicles[i].violate_time == 0 and vehicles.violate_load == 0 \
            #         and vehicles[i].pickup_violate_time == 0 and vehicles[i].dropoff_violate_time == 0:
            #     vehicles[i].d_additional[node] = nearest
            #     break
            # elif vehicles[i].violate_time == 0 and vehicles.violate_load == 0 \
            #         and vehicles[i].pickup_violate_time == 0 and vehicles[i].dropoff_violate_time != 0:
            #     vehicles[i].o_additional[node] = nearest
            #     send_to_nearest(node, vehicles[i].dropoff_route, False)
            # else:
            #     nodes[nearest].demand -= 1
            #     vehicles[nodes[nearest].belong_veh].del_node_by_index(0)

def initial_solution() -> list:
    assigned = []
    #print(nodes)
    for i in range(len(nodes)):
        nodes[i].tag = (int)((nodes[i].ready_time + nodes[i].due_time)/2)
    #print(nodes)
    cur_nodes = copy.deepcopy(nodes)
    #print(cur_nodes)
    cur_nodes.sort(key=lambda x: x.pat, reverse=True)
    for i in range(len(vehicles)):
        # print(cur_nodes[i].c_id)
        # if cur_nodes[i].c_id == nodes[cur_nodes[i].c_id].c_id:
        #     print('yes')
        vehicles[i].insert_node(cur_nodes[i].c_id)
        vehicles[i].insert_dropoff_node(cur_nodes[i].c_id, 0)
        nodes[cur_nodes[i].c_id].belong_veh = vehicles[i].v_id
        assigned.append(cur_nodes[i].c_id)
    for i in range(len(vehicles), len(cur_nodes)):
        # print(cur_nodes[i].c_id)
        for j in range(len(vehicles)):
            # print(vehicles[i].violate_time)
            # print(vehicles[i].violate_load)
            # print(vehicles[i].pickup_violate_time)
            # print(vehicles[i].dropoff_violate_time)
            # print()
            # print()
            # if vehicles[i].violate_time == 0 and vehicles[i].violate_load == 0 \
            #     and vehicles[i].pickup_violate_time == 0 and vehicles[i].dropoff_violate_time == 0:
                # print('yes')
            if vehicles[j].cap - vehicles[j].load > 0:
                vehicles[j].insert_node(cur_nodes[i].c_id)
                vehicles[j].insert_dropoff_node(cur_nodes[i].c_id, 0)
                nodes[cur_nodes[i].c_id].belong_veh = vehicles[j].v_id
                assigned.append(cur_nodes[i].c_id)
                break
    if len(assigned) < len(cur_nodes):
        print('yes')
        for i in range(len(cur_nodes)):
            if cur_nodes[i].c_id not in assigned:
                send_to_nearest(cur_nodes[i].c_id, assigned, True)
    return vehicles

def tabu_search(cur_vehs: list, iter_max: int = 200, time_limit: int = 150, p_l = 1, p_t = 1, p_f = 0.0001)\
        -> (list, float):
    '''
    tabu_search: main search
    input:
        cur_veh: initial solution
        tabu_tenure: tabu length
        iter_max: the maximum iteration number
        p_l: penalty weight (alpha)
        p_t: penalty weight (beta)
    output:
    '''
    iter_break = 0.2 * iter_max
    tabu_tenure = np.ceil(5 * np.log10(len(nodes)))
    count_break = 0
    iteration = 0
    num_nodes = len(nodes)
    tabu = np.zeros((num_nodes, len(cur_vehs)))
    #global_best_vehs = [copy.deepcopy(cur_vehs[i]) for i in range(len(cur_vehs))]
    global_best_vehs = copy.deepcopy(cur_vehs)
    for i in range(len(global_best_vehs)):
        print(global_best_vehs[i])
    global_best_obj = cal_obj(global_best_vehs,p_l, p_t, p_f)
    print(global_best_obj)
    searching_time = 0
    list = []
    drop_list = []
    while iteration <= iter_max and searching_time < time_limit:
        start = time.clock()
        iteration += 1
        neighbor_best_obj = neighbor_best_no_tabu_obj = np.inf
        # the best move
        best_cus = best_veh = 0
        # the best non-tabu moves
        best_cus_no_tabu = best_veh_no_tabu = 0
        best_pos = best_pos_no_tabu = 0
        best_drop_pos = best_drop_pos_no_tabu = 0
        # record the original position
        cur_pos = 0
        cur_drop_pos = 0
        # penalty components
        times_ck = np.zeros((num_nodes, len(cur_vehs)))
        for c in range(len(nodes)):
            if nodes[c].c_id in cur_vehs[nodes[c].belong_veh].o_additional.keys():
                continue
            for i in range(len(cur_vehs[nodes[c].belong_veh].route)): # search the position of node c
                if cur_vehs[nodes[c].belong_veh].route[i] == c:
                    cur_pos = i
                    break
            for i in range(len(cur_vehs[nodes[c].belong_veh].dropoff_route)):
                if cur_vehs[nodes[c].belong_veh].dropoff_route[i] == c:
                    cur_drop_pos = i
                    break
            for k in range(len(cur_vehs)):
                if nodes[c].belong_veh != k:
                    for p in range(len(cur_vehs[k].route) + 1):
                        # 修改
                        cur_vehs[k].insert_node(c, p)
                        if nodes[c] in cur_vehs[nodes[c].belong_veh].o_additional.values():
                            list = [key for key,value in cur_vehs[nodes[c].belong_veh].o_additional.items() if value == nodes[c]]
                            for key in list:
                                del cur_vehs[nodes[c].belong_veh].o_additional[key]
                                cur_vehs[k].o_additional[key] = nodes[c]
                        else:
                            cur_vehs[nodes[c].belong_veh].del_node_by_node(nodes[c])
                        for q in range(len(cur_vehs[k].dropoff_route) + 1):
                            cur_vehs[k].insert_dropoff_node(c, q)  ### insert dropoff node
                            if nodes[c] in cur_vehs[nodes[c].belong_veh].d_additional.values():
                                drop_list = [key for key, value in cur_vehs[nodes[c].belong_veh].d_additional.items() if
                                        value == nodes[c]]
                                for key in drop_list:
                                    del cur_vehs[nodes[c].belong_veh].d_additional[key]
                                    cur_vehs[k].d_additional[key] = nodes[c]
                            else:
                                # print(nodes[c].c_id)
                                # print(cur_vehs[nodes[c].belong_veh].dropoff_route)
                                cur_vehs[nodes[c].belong_veh].del_dropoff_node_by_node(nodes[c])
                            times_ck[c][k] += 1
                            inserted_obj = cal_obj(cur_vehs, p_l, p_t, p_f, times_ck[c][k])
                            if inserted_obj < neighbor_best_obj:
                                neighbor_best_obj = inserted_obj
                                best_cus = c
                                best_veh = k
                                best_pos = p
                                best_drop_pos = q
                                ori_pos = cur_pos
                                ori_drop_pos = cur_drop_pos
                            if inserted_obj < neighbor_best_no_tabu_obj and tabu[c][k] == 0:
                                neighbor_best_no_tabu_obj = inserted_obj
                                best_cus_no_tabu = c
                                best_veh_no_tabu = k
                                best_pos_no_tabu = p
                                best_drop_pos_no_tabu = q
                            cur_vehs[nodes[c].belong_veh].insert_dropoff_node(c, cur_drop_pos)
                            cur_vehs[k].del_dropoff_node_by_node(nodes[c])
                            for key in drop_list:
                                del cur_vehs[k].d_additional[key]
                                cur_vehs[nodes[c].belong_veh].d_additional[key] = nodes[c]
                        # 还原
                        cur_vehs[nodes[c].belong_veh].insert_node(c, cur_pos)
                        cur_vehs[k].del_node_by_node(nodes[c])
                        for key in list:
                            del cur_vehs[k].o_additional[key]
                            cur_vehs[nodes[c].belong_veh].o_additional[key] = nodes[c]
        # check if the best neighbor is feasible or not, update the penalty weight
        cur_vehs[best_veh].insert_node(best_cus, best_pos)
        cur_vehs[best_veh].insert_dropoff_node(best_cus, best_drop_pos)
        cur_vehs[nodes[best_cus].belong_veh].del_node_by_node(nodes[best_cus])
        cur_vehs[nodes[best_cus].belong_veh].del_dropoff_node_by_node(nodes[best_cus])
        status, p_l, p_t = check_feasible(cur_vehs, p_l, p_t)
        if neighbor_best_obj < global_best_obj and status:
            global_best_obj = neighbor_best_obj
            global_best_vehs = copy.deepcopy(cur_vehs)
            tabu[best_cus][best_veh] = tabu_tenure
            nodes[best_cus].belong_veh = best_veh
            #count_break = 0
        else:
            cur_vehs[nodes[best_cus].belong_veh].insert_node(best_cus, ori_pos)
            cur_vehs[nodes[best_cus].belong_veh].insert_dropoff_node(best_cus, ori_drop_pos)
            cur_vehs[best_veh].del_node_by_node(nodes[best_cus])
            cur_vehs[best_veh].del_dropoff_node_by_node(nodes[best_cus])
            # non-tabu
            cur_vehs[best_veh_no_tabu].insert_node(best_cus_no_tabu, best_pos_no_tabu)
            cur_vehs[best_veh_no_tabu].insert_dropoff_node(best_cus_no_tabu, best_drop_pos_no_tabu)
            cur_vehs[nodes[best_cus_no_tabu].belong_veh].del_node_by_node(nodes[best_cus_no_tabu])
            cur_vehs[nodes[best_cus_no_tabu].belong_veh].del_dropoff_node_by_node(nodes[best_cus_no_tabu])
            # tabu[best_cus_no_tabu][best_veh_no_tabu] = iteration + tabu_tenure
            tabu[best_cus_no_tabu][best_veh_no_tabu] = tabu_tenure
            nodes[best_cus_no_tabu].belong_veh = best_veh_no_tabu
            #count_break += 1
        # if count_break == iter_break:
        #     print('break')
        #     break
        tabu = np.where(tabu > 0, tabu - 1, tabu)
        end = time.clock()
        searching_time += (start - end)
    for v in range(len(global_best_vehs)):
        for i in range(len(global_best_vehs[v].route)):
            nodes[global_best_vehs[v].route[i]].belong_veh = global_best_vehs[v].v_id
    print(global_best_obj)
    return global_best_vehs, global_best_obj

if __name__ == '__main__':
    vertex_path = r'vertex.txt'
    node_path = r'request.txt'
    vehicle_path = r'vehicle.txt'
    iter_max = 200
    time_limit = 200
    p_l = 1
    p_t = 1
    p_f = 0.0001
    vertexes = read_vertex(vertex_path)
    vehicles = read_vehicle(vehicle_path)
    nodes = read_nodes(node_path)
    distance_matrix = cal_distance(vertexes)
    start = time.clock()
    init_vehs = initial_solution()
    # for i in range(len(init_vehs)):
    #     print(init_vehs[i])
    # for i in range(len(init_vehs)):
    #     print(init_vehs[i])
    # for i in range(len(nodes)):
    #     print(nodes[i].belong_veh)
    best_vehs, best_obj = tabu_search(init_vehs, iter_max, time_limit, p_l, p_t, p_f)
    for i in range(len(best_vehs)):
        print(init_vehs[i])
    end = time.clock()
    print('TS耗时{:.9f}'.format(end - start))
