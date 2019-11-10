import numpy

import utility
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b):
    return a * numpy.exp(-b * x)

def adj_difference(real_matrix,prob_matrix):
    r=0
    for row in prob_matrix:
        col=0
        for value in row:
            if value!=0:
                prob_matrix[r][col]=1
            col+=1
        r+=1
    rw=0
    differences=0
    for row in real_matrix:
        column=0
        for col in row:
            differences+=real_matrix[rw][column]-probably_adj_matrix[rw][column]
            column+=1
        rw+=1
    return differences


if __name__ == '__main__':
    configurator = utility.open_json(
        "/home/riccardo/PycharmProjects/sensor-pattern-analysis/motion-simulator/configurations.json")
    rooms = []  # rooms counter
    adj_matrix = []
    probably_adj_matrix = []
    for room in configurator["room"]:  # doing correct matrix
        rooms.append(room)
    for room in configurator["room"]:
        room_adj = numpy.zeros(len(rooms), dtype=int)
        probably_adj_matrix.append(numpy.zeros(len(rooms), dtype=int))
        i = 0
        for rm in rooms:
            for adj in configurator["room"][room]:
                if adj == rm:
                    room_adj[i] = 1
            i += 1
        adj_matrix.append(room_adj)
    y_differences = []
    x_time = []
    filter_output = utility.get_reader(
        "/home/riccardo/PycharmProjects/sensor-pattern-analysis/motion-simulator/HF_out.csv")
    HF_out = utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/motion-simulator/HF_out.csv")
    row_index = 0
    for row in filter_output:   # read each row of HF_out
        old_chamber = 0  # room that doesn't change sensor between two rows
        new_chamber = 0  # room that changes sensor between two rows
        if row_index != 0:
            on_sensor = 0
            for i in range(1, len(rooms) + 1):  # on sensor counter
                if row[i] == str(1):
                    on_sensor += 1
            if on_sensor == 2:
                index = 0
                for value in HF_out.iloc[row_index-1]:  # for that analyze sensor data of actual row and previous row
                    ind = 0
                    for previous_value in HF_out.iloc[row_index - 2]:
                        if ind == index:
                            if str(previous_value) == str(0) and str(value) == str(1):
                                new_chamber=index
                            if str(previous_value) == str(1) and str(value) == str(1):
                                old_chamber=ind
                        ind += 1
                    index += 1
                if old_chamber!=new_chamber and old_chamber!=0 and new_chamber!=0:
                    probably_adj_matrix[old_chamber-1][new_chamber-1]+=1
                    probably_adj_matrix[new_chamber - 1][old_chamber - 1] += 1
            y_differences.append(adj_difference(adj_matrix,probably_adj_matrix))
            x_time.append(int(float(row[0])))
        row_index += 1
        #if row_index%200 == 0 or row_index==1:
            #print("Row: "+str(row_index)+" difference:"+str(adj_difference(adj_matrix,probably_adj_matrix)))
    plt.xlabel('time(s)')
    plt.ylabel('adj_difference')
    plt.plot(x_time,y_differences)
    x_linspace=numpy.linspace(0, row_index-1,row_index-1)
    popt= curve_fit(func, x_linspace, y_differences)
    plt.plot(x_linspace, func(x_linspace, *popt[0]))
    plt.show()

