import numpy as np


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