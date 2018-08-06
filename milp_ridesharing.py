from gurobipy import *
import util



def creat_model_data():
    request_path = r'data\request.txt'
    vertex_path = r'data\vertex.txt'
    vehicle_path = r'data\vehicle.txt'
    requests = util.read_request(request_path)
    vertexs = util.read_vertex(vertex_path)
    vehicles = util.read_vehicle(vehicle_path)
    time_matrix = util.cal_time(vertexs)
    distance_matrix = time_matrix * 2
    tim = multidict({})
    for i,j in range(time_matrix.shape[0]):
        tim[i,j] = time_matrix[i][j]
    print(tim)


    print()

def milp_optimize():
    print()

if __name__ == '__main__':
    m = Model('netflow')

    creat_model_data()
    milp_optimize()