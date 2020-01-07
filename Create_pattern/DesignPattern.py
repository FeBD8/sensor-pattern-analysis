import numpy
import utility
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
Script that analyze all HF_out searching every sure adjacency and so creating a new adj_matrix

sure_adjacency is when from only one sensor on the next instant there are two sensors on.
E.g. : [0 0 0 1 0]
       [0 0 1 1 0]
The adjacency is from room of index [3] to room of index [2] 


"""

def func(x, a, b):
    """
    function used to create a curve fit
    """
    return a * numpy.exp(-b * x)


def adj_difference(real_matrix, prob_matrix):

    """
    function that sets all sure adjacencies to 1 and after that returns the number of missing adjacencies
    between the new adj_matrix and the real one

    :param real_matrix: the real matrix used for simulation
    :type real_matrix: np.array
    :param prob_matrix: the new matrix that this script creates
    :type prob_matrix: np.array
    :return: the number of missing adjacencies
    :rtype: int
    """
    for r,row in enumerate(prob_matrix):
        for col,value in enumerate(row):
            if value != 0:
                prob_matrix[r][col] = 1
    differences = 0
    for rw,row in enumerate(real_matrix):
        for column,col in enumerate(row):
            differences += real_matrix[rw][column] - probably_adj_matrix[rw][column]
    return differences


if __name__ == '__main__':
    configurator = utility.open_json(
        "/home/riccardo/PycharmProjects/sensor-pattern-analysis/motion-simulator/configurations.json")
    rooms = []  # rooms counter
    probably_adj_matrix = []
    for room in configurator["room"]:  # doing correct matrix
        rooms.append(room)
    adj_matrix = numpy.zeros((len(rooms), len(rooms)), dtype=int)
    probably_adj_matrix = numpy.zeros((len(rooms), len(rooms)), dtype=int)
    i = 0
    for room in configurator["room"]:
        ad = 0
        for rm in rooms:
            for adj in configurator["room"][room]:
                if adj == rm:
                    adj_matrix[i][ad] = 1
            ad+=1
        i += 1

    y_differences = []
    x_time = []
    HF_out = utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/Create_pattern/sensor_real.csv")
    for i, r in HF_out.iterrows():  # read each row of HF_out
        old_chamber = 0  # room that doesn't change sensor between two rows
        new_chamber = 0  # room that changes sensor between two rows
        if i != 0:
            on_sensor = 0
            for rw in range(1, len(rooms) + 1):  # on sensor counter
                if r[rw] == 1:
                    on_sensor += 1
            if on_sensor == 2:
                for index,value in enumerate(HF_out.iloc[i]):  # for that analyze sensor data of actual row and previous row
                    for ind,previous_value in enumerate(HF_out.iloc[i - 1]):
                        if ind == index:
                            if str(previous_value) == str(0) and str(value) == str(1):
                                print(i)
                                new_chamber = ind
                            if str(previous_value) == str(1) and str(value) == str(1):
                                old_chamber = ind
                if old_chamber != new_chamber and old_chamber != 0 and new_chamber != 0:
                    probably_adj_matrix[old_chamber - 1][new_chamber - 1] += 1  # add symmetrical adjacencies
                    probably_adj_matrix[new_chamber - 1][old_chamber - 1] += 1
            #y_differences.append(adj_difference(adj_matrix, probably_adj_matrix))
            #x_time.append(int(float(r[0])))
        # if i%200 == 0 or i==1:
        # print("Row: "+str(i)+" difference:"+str(adj_difference(adj_matrix,probably_adj_matrix)))
    for i,r in enumerate(rooms):
        print(probably_adj_matrix[i])
        print(adj_matrix[i])
    #plt.xlabel('time(s)')
    #plt.ylabel('adj_difference')
    #plt.plot(x_time, y_differences)
    # x_linspace=numpy.linspace(0, len(HF_out)-1,len(HF_out)-1)
    # popt= curve_fit(func, x_linspace, y_differences)
    # plt.plot(x_linspace, func(x_linspace, *popt[0]))
    plt.show()
