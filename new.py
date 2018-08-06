import copy
import time
import numpy as np
import sys


class Vertex(object):
    def __init__(self, v_id, x, y):
        self.v_id = v_id
        self.x = x
        self.y = y
        #self.type = type


class Request(object):
    def __init__(self, ov, dv, ept, lpt, pat):
        self.ov = ov
        self.dv = dv
        self.ept = ept
        self.lpt = lpt
        self.pat = pat
        self.tag = 0
        self.belong_veh = None


class Vehicle(object):
    def __init__(self, d_id, ov, dv, edt, ldt, pat, maxr):
        self.d_id = d_id
        self.ov = ov
        self.dv = dv
        self.edt = edt
        self.ldt = ldt
        self.pat = pat
        self.maxr = maxr
        self.cap = 3


class Ridesharing_List(object):
    def __init__(self, d_id):
        self.d_id = d_id
        self.start_time = 0
        self.pickup_time = {}
        self.pickup_list = []
        self.dropoff_list = []
        self.dropoff_time = {}
        self.o_addition = {}
        self.d_addition = {}
        self.total_time = 0
        self.end_time = 0
        self.load = 0

    def update_info(self):
        #print(len(self.pickup_list))
        self.pickup_time[self.pickup_list[0].ov] = self.start_time + \
                                                   time_matrix[vehicles[self.d_id].ov][self.pickup_list[0].ov]
        for i in range(len(self.pickup_list) - 1):
            self.pickup_time[self.pickup_list[i + 1].ov] = self.pickup_time[self.pickup_list[i].ov] + \
                                                           time_matrix[self.pickup_list[i + 1].ov][self.pickup_list[i].ov]
        d_list = copy.copy(self.pickup_list)
        if len(self.o_addition) > 0:
            # print('occour')
            for r in self.o_addition.keys():
                d_list.append(r)
        self.dropoff_list = copy.copy(arrange_list(self.pickup_list[-1].ov, d_list, False))
        # print(self.dropoff_list)
        self.dropoff_time[self.dropoff_list[0].dv] = self.pickup_time[self.pickup_list[-1].ov] + \
                                                  time_matrix[self.pickup_list[-1].ov][self.dropoff_list[0].dv]
        for i in range(len(self.dropoff_list) - 1):
            self.dropoff_time[self.dropoff_list[i + 1].dv] = self.dropoff_time[self.dropoff_list[i].dv] + \
                                                      time_matrix[self.dropoff_list[i].dv][self.dropoff_list[i + 1].dv]
        self.end_time = self.dropoff_time[self.dropoff_list[-1].dv]
        self.total_time = self.end_time - self.start_time
        if self.total_time > vehicles[self.d_id].maxr:
            print('occour')
    # sort the pickup_time & dropoff_time
        # update the total_time
        # self.pickup_time = sorted(self.pickup_time.items(), key=lambda x: x[1])
        # self.dropoff_time = sorted(self.dropoff_time.items(), key=lambda x: x[1])
        # v = self.sorted(self.dropoff_time.items(), key=lambda x: x[1])[-1][0]
        # t = self.sorted(self.dropoff_time.items(), key=lambda x: x[1])[-1][1]
        # # self.total_time = t - self.start_time + time_matrix[v][Vehicle[d_id].dv]
        # self.load = len(self.pickup_time)


class Solution(object):
    def __init__(self):
        self.ridesharing_list = []
        self.total_time = 0
        self.s_pair = {}

    def add_request(self, d_id, ov, time):
        #self.ridesharing_list.append(rl[d_id])
        #rl[d_id].pickup_time[ov] = time;
        self.s_pair[ov] = d_id

    def del_request(self, d_id, ov, dv):
        #rl[d_id].pickup_time.pop[ov]
        #rl[d_id].dropoff_time.pop[dv]
        self.s_pair.pop(ov)

    def update_info(self):
        # sort the pickup_time & dropoff_time
        # update the total_time
        # for i in range(1, len(self.ridesharing_list)):
        #     self.total_time += self.ridesharing_list[i - 1].total_time
        for i in range(len(self.ridesharing_list)):
            self.ridesharing_list[i].update_info()

    # def output(self):
        # for i in range(len(self.ridesharing_list)):
        #     print('Vehicle{}: start_point V{}'.format(i + 1, ))




def read_request(path: str) -> list:
    with open(path, 'r', ) as f:
        lines = f.readlines()
    # capacity = (int)(lines[4].split()[-1])
    # max_vehicle = (int)(lines[4].split()[0])
    lines = lines[1:]
    requests = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 5:
            request = Request(*info)
            requests.append(request)
    return requests

def read_vertex(path: str) -> list:
    with open(path, 'r', ) as f:
        lines = f.readlines()
    # capacity = (int)(lines[4].split()[-1])
    # max_vehicle = (int)(lines[4].split()[0])
    lines = lines[1:]
    vertexs = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 3:
            vertex = Vertex(*info)
            vertexs.append(vertex)
    return vertexs

def read_vehicle(path: str) -> list:
    with open(path, 'r', ) as f:
        lines = f.readlines()
    # capacity = (int)(lines[4].split()[-1])
    # max_vehicle = (int)(lines[4].split()[0])
    lines = lines[1:]
    vehicles = []
    for line in lines:
        info = [int(j) for j in line.split()]
        if len(info) == 7:
            vehicle = Vehicle(*info)
            vehicles.append(vehicle)
    return vehicles

def cal_time(vertexs: list) -> np.array:
    time_matrix = np.zeros((len(vertexs), len(vertexs)), )
    for i in range(len(vertexs)):
        for j in range(i + 1, len(vertexs)):
            if i != j:
                time = np.sqrt((vertexs[i].x - vertexs[j].x) ** 2 + (vertexs[i].y - vertexs[j].y) ** 2)
                time_matrix[i][j] = time_matrix[j][i] = time
    return time_matrix

def arrange_list(v , pickup_list : list, flag: bool) -> list:
    s = []
    i = 1
    sumpath = 0
    n = len(pickup_list)
    s.append(v)
    result = []
    while True:
        q = 0
        Detemp = sys.maxsize
        while True:
            #l = 0
            if flag:
                k = pickup_list[q].ov
            else:
                k = pickup_list[q].dv
            flag = 0
            if k in s:
                flag = 1
            if (flag == 0) and (time_matrix[k][s[i - 1]] < Detemp):
                j = k
                temp = pickup_list[q]
                Detemp = time_matrix[k][s[i - 1]]
            q += 1
            if q >= n:
                break
        s.append(j)
        result.append(temp)
        i += 1
        sumpath += Detemp
        if i >= n + 1:
            break
        # for i in range(len(result)):
        #     print(result[i].ov)
    return result


def checkConstraint(rl: Ridesharing_List, re: Request) -> bool:
    if len(rl.pickup_list) == 0:
        #rl.pickup_time[re.ov] = time_matrix[re.ov][vehicles[rl.d_id].ov]
        #temp = time_matrix[re.ov][vehicles[rl.d_id].ov]
        # if temp + vehicles[rl.d_id].edt > re.lpt:
        # #     #check more constraint
        #      return False
        # temp_start_time = re.tag - temp
        # if temp_start_time < vehicles[rl.d_id].edt:
        #     return False
        # temp = time_matrix[re.ov][re.dv] + temp
        # temp_end = temp + time_matrix[re.dv][vehicles[rl.d_id].dv]
        # temp_total = temp_end - rl.start_time
        # if temp_total > vehicles[rl.d_id].maxr:
        #     return False
        return True
    temp_list = copy.copy(rl.pickup_list)
    temp_list.append(re)

    temp_list = copy.copy(arrange_list(vehicles[rl.d_id].ov, temp_list, True))
    temp_runt = time_matrix[vehicles[rl.d_id].ov][temp_list[0].ov]

    for i in range(len(temp_list) - 1):
        temp_runt += time_matrix[temp_list[i].ov][temp_list[i+1].ov]
    if temp_runt + time_matrix[temp_list[-1].ov][vehicles[rl.d_id].dv] > vehicles[rl.d_id].maxr:
        return False
        # 是否要找最短路径
    #     dict[i] = time_matrix[i][vehicles[rl.d_id].ov]
    # temp = rl.pickup_time[dict.sorted(dict.items(), key=lambda x: x[1])[-1][0]] = temp + vehicles[rl.d_id].edt + \
    #                 time_matrix[dict.sorted(dict.items(), key=lambda x: x[1])[-1][0]][vehicles[rl.d_id].ov]
    # dict.popitem()


    return True

def sendTo_nearest(r: Request, s : Solution, assigned : list):
    dict = {}
    for i in range(len(assigned)):
        # print(assigned[i].ov)
        dict[assigned[i]] = time_matrix[r.ov][assigned[i].ov]
    for i in range(len(dict)):
        nr = sorted(dict.items(), key=lambda x: x[1])[i][0]
        d_id = nr.belong_veh
        if s.ridesharing_list[d_id].load < vehicles[d_id].cap:
            s.ridesharing_list[d_id].o_addition[r] = nr
            s.ridesharing_list[d_id].load += 1
            # print(d_id)
            # print(nr.ov)
            break

def init_s() -> Solution:
    assigned = []
    for r in requests:
        r.tag = (int)((r.ept + r.lpt)/2)
    requests.sort(key = lambda x:x.pat, reverse = True)
    init_solution = Solution()
    record = []
    for i in range(len(vehicles)):
        rl = Ridesharing_List(vehicles[i].d_id)

        #rl.d_id = vehicles[i].d_id
        if checkConstraint(rl, requests[i]):
            #rl.pickup_time[requests[i].ov] = 0
            rl.pickup_list.append(requests[i])
            rl.start_time = requests[i].tag - time_matrix[requests[i].ov][vehicles[rl.d_id].ov]
            rl.pickup_time[requests[i].ov] = time_matrix[requests[i].ov][vehicles[rl.d_id].ov]
            rl.dropoff_time[requests[i].dv] = time_matrix[requests[i].ov][requests[i].dv] + rl.pickup_time[requests[i].ov]
            rl.end_time = rl.dropoff_time[requests[i].dv] + time_matrix[requests[i].dv][vehicles[rl.d_id].dv]
            #rl.total_time = rl.end_time - rl.start_time
            assigned.append(requests[i])

            # time_matrix[vehicles[i - 1].ov][requests[i - 1].ov] + vehicles[i - 1].edt
            requests[i].belong_veh = vehicles[i].d_id
            rl.load += 1
            init_solution.ridesharing_list.append(rl)
        else:
            record.append(vehicles[i].d_id)
    for i in range(len(record)):
        rl = Ridesharing_List(record[i])
        init_solution.ridesharing_list.append(rl)
    # if len(init_solution.ridesharing_list) < len(vehicles):
    #     for i in range(len(vehicles) - len(init_solution.ridesharing_list)):
    #         rl = Ridesharing_List(i + len(init_solution.ridesharing_list))
    #         init_solution.ridesharing_list.append(rl)
    for i in range(len(requests) - len(vehicles)):
        for j in range(len(vehicles)):
            if init_solution.ridesharing_list[j].load + 1 > vehicles[j].cap:
                continue
            else:
                break
        if checkConstraint(init_solution.ridesharing_list[j], requests[i + len(vehicles)]):
            init_solution.ridesharing_list[j].load += 1
            init_solution.ridesharing_list[j].pickup_list.append(requests[i + len(vehicles)])
            assigned.append(requests[i + len(vehicles)])
            requests[i + len(vehicles)].belong_veh = vehicles[j].d_id
    if len(assigned) == len(requests):

        for i in range(len(init_solution.ridesharing_list)):
            # for j in range(len(init_solution.ridesharing_list[i].pickup_list)):
            #     print(init_solution.ridesharing_list[i].pickup_list[j].ov)
            result_list = copy.copy(arrange_list(vehicles[init_solution.ridesharing_list[i].d_id].ov, init_solution.ridesharing_list[i].pickup_list, True))
            # print(result_list)
            # length = len(result_list) - 1
            # requests.sort(key=lambda x: x.ov, reverse=True)
            # for j in range(len(result_list)):
            #     init_solution.ridesharing_list[i].pickup_list[j] = result_list[j]
            init_solution.ridesharing_list[i].pickup_list = copy.copy(result_list)
            # init_solution.ridesharing_list[i].start_time = init_solution.ridesharing_list[i].pickup_list[0].tag - time_matrix[init_solution.ridesharing_list[i].pickup_list[0].ov][vehicles[init_solution.ridesharing_list[i].d_id].ov]
        init_solution.update_info()
        return init_solution
    else:
        # print(len(init_solution.ridesharing_list))
        for r in requests:
            if r not in assigned:
                sendTo_nearest(r, init_solution, assigned)
        for i in range(len(init_solution.ridesharing_list)):
            result_list = arrange_list(vehicles[init_solution.ridesharing_list[i].d_id].ov, init_solution.ridesharing_list[i].pickup_list, True)
            for j in range(len(result_list)):
                print(result_list[j])
            length = len(result_list) - 1
            requests.sort(key=lambda x: x.ov, reverse=True)
            for j in range(length):
                init_solution.ridesharing_list[i].pickup_list[j] = requests[result_list[j + 1]]

        # arrange_list()
        init_solution.update_info()
    return init_solution

def cal_obj(s: Solution) -> float:
    total_time = Solution.total_time
    #T = sum([v.total_time - )])



def tabu_search(init_solution: Solution, tabu_tenure:int = 20, iter_max = 200, p_l = 1, p_t = 1) -> Solution:
    iter_break = 0.2 * iter_max
    count_break = 0
    iteration = 0
    num_requests = len(requests)
    tabu_table = np.zeros(num_requests, len(vehicles))
    global_best = copy.deepcopy(init_solution)
    global_best_obj = cal_obj(global_best)

    return

if __name__ == '__main__':
    request_path = r'data\request.txt'
    vertex_path = r'data\vertex.txt'
    vehicle_path = r'data\vehicle.txt'
    requests = read_request(request_path)
    vertexs = read_vertex(vertex_path)
    vehicles = read_vehicle(vehicle_path)
    time_matrix = cal_time(vertexs)
    start = time.clock()
    init_solution = init_s()
    # for j in range(len(init_solution.ridesharing_list)):
    #     print(init_solution.ridesharing_list[j].d_id)
    #     # print(init_solution.ridesharing_list[j].start_time)
    #     for m in range(len(init_solution.ridesharing_list[j].pickup_list)):
    #         print(init_solution.ridesharing_list[j].pickup_list[m].ov)
    #     print()
    #best_solution = tabu_search(init_solution)
    end = time.clock()
    print('TS耗时{:.9f}'.format(end - start))
    #print(best_solution)

