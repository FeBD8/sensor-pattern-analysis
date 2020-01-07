import sys

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import animation
import utility
import numpy as np

""" There are two execution types :

 -Time triggered: to execute: python3 Main.py -time config.json 
 
 -Event triggered: to execute: python3 Main.py -evtr config.json 
 
 """


def set_figure():
    figure = plt.figure()
    figure.set_dpi(100)
    figure.set_size_inches(configurator["info"]["figure_size"][0], configurator["info"]["figure_size"][1])
    return figure


def color_heat_map(i):
    dic = configurator["evaluation_thresholds"]
    rgba = configurator["rgba_tuple"]
    colormap = []
    for bel in df.iloc[i][len(configurator2["room"]) + 2:(len(configurator2["room"]) + 1) * 2]:
        for color in dic:
            if dic[color][1] >= bel >= dic[color][0]:  # se il valore  è compreso nella soglia setto il colore
                tupla = (rgba[color][0], rgba[color][1], rgba[color][2], rgba[color][3])
                colormap.append(tupla)
    nodes.set_color(colormap)


def set_probability(i):
    x = 0
    for bel in df.iloc[i][len(configurator2["room"]) + 2:(len(configurator2["room"]) + 1) * 2]:
        value = '%.3f' % bel
        if bel > 0.8:
            probabilities[x].set_color("cyan")
            labels[df.columns[1:len(configurator2["room"]) + 1][x]].set_color("cyan")
        else:
            probabilities[x].set_color("black")
            labels[df.columns[1:len(configurator2["room"]) + 1][x]].set_color("black")
        probabilities[x].set_text(value)
        x += 1


def set_time(i):
    time.set_text("Time : " + str(df["Time"][i]))  # il tempo lo prendo da HF_out


def animation_logic(i):
    set_lamp_position(i)
    set_time(i)
    set_sensor_output(i)
    set_probability(i)
    color_heat_map(i)
    set_correct_room(i)
    set_ev_level(i)


def set_ev_level(index):
    float_value = float(df_filter.iloc[index, 1])  # prendo la colonna efficienza di output_evaluation
    value = '%.3f' % float_value
    text = "Evaluation level : " + value
    ev_level.set_text(text)


def set_ground_truth_position(ground_truth, n, room, somma):
    ground_truth[n].set_visible(False)
    ground_truth[n] = set_ground_truth(somma - 1)
    position = [pos[room][0] - 2.1*(6/len(room_counter)) - (somma / 8)*(6/len(room_counter)),
                    pos[room][1] - 0.6*(3/len(room_counter))]  # set person image position near each room
    img_extent = [position[0] + 1.2*(6/len(room_counter)), position[0], position[1] - 1.2*(6/len(room_counter)), position[1]]
    ground_truth[n].set_extent(img_extent)
    ground_truth[n].set_visible(True)


def set_correct_room(i):
    time_value = df["Time"][i]
    y = 0
    for truth_room in df_ground_truth["Room"]:  # for every row of out.csv
        if time_value == round(df_ground_truth["Time"][y]):  # if row's time is equal to hf_out time
            for n in range(0, n_of_person):
                if df_ground_truth["Person"][y] == n:  # finding row's person number
                    x = 0  # it's index room where i'm doing the for loop
                    for room in df.columns[1:len(configurator2["room"]) + 1]:
                        if room == truth_room:
                            person_counter[n][x] += 1
                            somma = 0  # analyze if there is maximum person image in a room
                            for p in range(0, n_of_person):
                                somma += person_counter[p][x]
                            if somma == 1:
                                set_ground_truth_position(ground_truth,n,room,somma)
                            if somma == 2:
                                set_ground_truth_position(ground_truth,n,room,somma)
                            if somma == 3:
                                set_ground_truth_position(ground_truth,n,room,somma)
                            if somma > 3:
                                ground_truth[n].set_visible(False)
                                person_excess[x].set_text(
                                    str(somma - 3) + "+..")  # set text where too much people in one room
                                person_excess[x].set_visible(True)
                        else:  # if the change isn't in the room
                            if person_counter[n][x] == 1:
                                person_counter[n][x] -= 1
                                somma = 0
                                for p in range(0, n_of_person):
                                    somma += person_counter[p][x]
                                    if somma == 1 and person_counter[p][x] == 1:
                                        set_ground_truth_position(ground_truth,p,room,somma)
                                    if somma == 2 and person_counter[p][x] == 1:
                                        set_ground_truth_position(ground_truth,p,room,somma)
                                    if somma == 3 and person_counter[p][x] == 1:
                                        set_ground_truth_position(ground_truth,p,room,somma)
                                if somma > 3:
                                    person_excess[x].set_text(str(somma - 3) + "+..")
                                    person_excess[x].set_visible(True)
                                else:
                                    person_excess[x].set_visible(False)
                        x += 1
        y += 1
    if n_of_person == 1:
        text = "Ground truth : " + df["Room"][i]
        correct_room.set_text(text)


def set_sensors_lamp():
    image = plt.imread("lamp.png")
    return ax.imshow(image)


def set_ground_truth(i):
    if i == 0:
        image = plt.imread("person.png")
        return ax.imshow(image)
    if i == 1:
        image1 = plt.imread("person_blue.png")
        return ax.imshow(image1)
    if i == 2:
        image2 = plt.imread("person_red.png")
        return ax.imshow(image2)


def set_lamp_position(i):
    x = 0
    series = df.iloc[i][1:len(configurator2["room"]) + 1]
    for room in df.columns[1:len(configurator2["room"]) + 1]:
        for column, val in series.iteritems():
            if (room == column):
                if (val == 1):
                    position = [pos[room][0] + 1.5*(5/len(room_counter)), pos[room][1]]
                    img_extent = [position[0] + 1*(6/len(room_counter)), position[0] , position[1] - 1*(6/len(room_counter)), position[1] ]
                    sensor_lamp[x].set_extent(img_extent)
                    sensor_lamp[x].set_visible(True)
                else:
                    sensor_lamp[x].set_visible(False)
        x += 1


def set_sensor_output(i):
    series = df.iloc[i][1:len(
        configurator2["room"]) + 1]  # iloc mi seleziona riga e colonna in questo caso prendo il valore dei sensori
    text = "Sensors output"
    for column, val in series.iteritems():  # iteritems va a iterare su un valore con pandas e mi da colonna e valore
        text = text + "\n" + column + " : " + str(val)
    sensor_output.set_text(text)  # metto il testo nelle posizioni che ho preso dal json prima nel main


def animate_filter(i):
    animation_logic(i)
    nodi = [nodes]
    ret_list = [sensor_output, ev_level, time,
                correct_room] + nodi + probabilities + sensor_lamp + ground_truth + person_excess
    return ret_list


def animate_evtr(i):
    return animate_filter(int(round(event["Time"][i])))


def init_filter():
    nodi = [nodes]
    ret_list = [sensor_output, ev_level, time,
                correct_room] + nodi + probabilities + sensor_lamp + ground_truth
    return ret_list


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Chiamare il programma in maniera corretta")
        sys.exit(1)

    configurator = utility.open_json(sys.argv[2])
    fig = set_figure()  # setto dimensioni figura

    ax = plt.axes(xlim=(configurator["info"]["x_lim"][0], configurator["info"]["x_lim"][1]),
                  ylim=(
                      configurator["info"]["y_lim"][0], configurator["info"]["x_lim"][1]))  # definisco intervalle assi
    time = plt.text(configurator["time"]["position"][0], configurator["time"]["position"][1], "",
                    fontsize=configurator["time"]["font_size"])  # scelgo posizione e fontsize del tempo
    sensor_output = plt.text(configurator["text_area_s"]["position"][0],
                             configurator["text_area_s"]["position"][1], "Sensors output",
                             fontsize=configurator["text_area_s"]["font_size"])  # posizione output sensori
    configurator2 = utility.open_json(configurator["info"]["adj_file"])

    df = utility.read_file(configurator["info"]["input_file"])
    ev_level = plt.text(configurator["ev_level"]["position"][0], configurator["ev_level"]["position"][1], "",
                        fontsize=configurator["ev_level"]["font_size"])
    correct_room = plt.text(configurator["correct_room"]["position"][0],
                            configurator["correct_room"]["position"][1], "",
                            fontsize=configurator["correct_room"]["font_size"])
    n_of_person = configurator2["info"]["person_number"]
    number_person = plt.text(configurator["text_area_numb"]["position"][0],
                            configurator["text_area_numb"]["position"][1], "Number of people:"+ str(n_of_person),
                            fontsize=configurator["text_area_numb"]["font_size"])
    df_ground_truth = utility.read_file(configurator["info"]["ground_truth_file"])
    G = nx.Graph()
    sensor_lamp = []
    ground_truth = []
    room_counter = []
    person_counter = []  # contains counter of ground truth for each person in the house
    for room in df.columns[1:len(configurator2["room"]) + 1]:
        room_counter.append(0)  #  conto le stanze
        sensor_lamp.append(set_sensors_lamp())
        G.add_node(room)
    for n in range(0, n_of_person):
        room_i = []
        for room in df.columns[1:len(configurator2["room"]) + 1]:
            room_i.append(0)
        ground_truth.append(set_ground_truth(0))
        person_counter.append(room_i)
    for room in df.columns[1:len(configurator2["room"]) + 1]:
        for adj in configurator2["room"][room]:
            G.add_edge(room, adj)
    pos = nx.spring_layout(G, center=[10, 13], scale=5)
    nodes = nx.draw_networkx_nodes(G, pos=pos, with_labels="true", node_size=3400-(160*len(room_counter)), ax=ax,node_color="lightblue")
    edges = nx.draw_networkx_edges(G, pos=pos, with_labels="true", ax=ax, font_size=8,alpha=0.5)
    labels = nx.draw_networkx_labels(G, pos=pos, font_size=int(17-(0.2*len(room_counter))))  # 17
    probabilities = []
    person_excess = []
    for room in df.columns[1:len(configurator2["room"]) + 1]:
        person_excess.append(plt.text(pos[room][0] - 20/len(room_counter) - (3 / 8)/len(room_counter), pos[room][1] - 3/len(room_counter), "", fontsize=15-(0.4*len(room_counter))))
        probabilities.append(plt.text(pos[room][0] - 0.4, pos[room][1] - 0.6, "", fontsize=15-(0.4*len(room_counter))))
    df_filter = utility.read_file(configurator["info"]["evaluation_file"])
    if sys.argv[1] == "-time":
        anim = animation.FuncAnimation(fig, animate_filter, frames=len(df.index) - 1, init_func=init_filter,
                                       interval=configurator["info"]["time_speed"], blit=False,
                                       repeat=False)
        #  animate_filter è la funzione che viene richiamata ad ogni frame
    if sys.argv[1] == "-evtr":
        frame_time = plt.text(configurator["text_area_frame"]["position"][0],
                                configurator["text_area_frame"]["position"][1],
                                "Frame time: " + str(configurator["info"]["time_speed_evtr"]/1000)+"s",
                                fontsize=configurator["text_area_frame"]["font_size"])
        event = utility.read_file(configurator["info"]["event_file"])
        anim = animation.FuncAnimation(fig, animate_evtr, frames=len(event.index) - 1, init_func=init_filter,
                                       interval=configurator["info"]["time_speed_evtr"], blit=False,
                                       repeat=False)
    plt.tight_layout()
    plt.show()
