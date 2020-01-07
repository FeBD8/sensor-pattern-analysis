import sys

import networkx as nx
from matplotlib import gridspec
import numpy as np
import ReadFile
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

"""
    Script used to evaluate the output of the histogram filter between different config.json and 
    evaluate this difference.
    It also show a graph where adjacencies are made with every change in HF_out top probabilities.
"""


def open_json(file_name):
    with open(file_name) as json_config:
        data_config = json.load(json_config)
    json_config.close()
    return data_config


def set_up(data_config):
    """
    Function that return the HF_out given by the chosen config.json
    """
    data = ReadFile.ReadFile(data_config["info"]["output_file_path"] + data_config["info"]["output_file_name"]).df
    return data


def dictionary_room(data_config):
    """
    Dictionary [key='the name of the room' : value='the index of the column with the probability of the room']
    :param data_config: config.json or config_house1.json or config_house2.json ; all the file with the correct adjacency values
    :return: dictionary
    :rtype: dict
    """
    key = data_config["info"]["room_name"]
    value = data_config["info"]["columns_name"][1:]
    dictionary = {}
    for i in range(0, len(key)):
        dictionary[key[i]] = value[i]
    return dictionary


def dict_histog(data_config):
    """
    Dictionary [key='the name of the room' : value='the first one or two letters used to identify a room']

    :param data_config: config.json or config_house1.json or config_house2.json ; all the file with the correct adjacency values
    :return: dictionary
    :rtype: dict
    """
    key = data_config["info"]["room_name"]
    value = data_config["info"]["state_domain"]
    dictionary = {}
    for i in range(0, len(key)):
        dictionary[key[i]] = value[i]
    return dictionary


def found_max(probability_array: pd.Series, data_config):
    """
    Found the max value between the probability of the rooms given by the Histogram Filter

    :param probability_array: the row of HF_out
    :type: pd.Series
    :param data_config: config.json or config_house1.json or config_house2.json ; all the file with the correct adjacency values
    :return: max value of probability and its index
    """
    constant = len(data_config["info"]["state_domain"]) + 2
    res = probability_array.iloc[constant:].apply(float)
    val_max = res.max()
    index_max = res.idxmax()
    return val_max, index_max


def color_bar(sim_config, filter_adj):
    """
    Function that create the colorbar for the histogram plot, if the adjacency is correct the color is green instead if
    it is wrong the color is red

    :param sim_config: the configurations.json used for doing TestMotionSimulator
    :param filter_adj: all the adjacencies founded and sorted in descending order, edges of the Graph are used because
    in this way duplicate adjacencies are removed easily
    :type: networkx.Graph.edges
    :return: array for color that the histogram plot need
    """
    colors = np.empty(len(filter_adj), dtype=object)
    dict_adj = sim_config["room"]
    for i, k in enumerate(filter_adj):
        for adj in dict_adj[k[0]]:
            if k[1] == adj:
                colors[i] = 'green'
                break
            else:
                colors[i] = 'red'
    return colors


def main_static(file_correct, file_wrong):
    """
    Main function that compare the efficiency between two HF_out given by two different config.json.
    The function plot the evaluation,the difference from the evaluation and the sum of this difference,the histogram of
    the adjacencies founded and the graph of these adjacencies.
    All the adjacencies are created when the max probability of the Histogram Filter change.
    The final figure is saved in motion-simulator/Evaluate_images/

    :param file_correct: is the correct config.json (config.json,config_house1.json,config_house2.json)
    :param file_wrong: one of the wrong file in Configurations_file
    """
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    folder = "Configurations_file/"
    conf_file = folder + file_correct  # file1 correct
    config = open_json(conf_file)
    conf_real_adj_file = folder + file_wrong  # file2 wrong
    config_error = open_json(conf_real_adj_file)
    df = pd.DataFrame(columns=['Time', 'Efficiency real'])
    df_error = pd.DataFrame(columns=['Time', 'Efficiency fully-connected'])
    data = set_up(config)  # HF_out
    data_error = set_up(config_error)
    i = 0
    dictionary_rooms = dictionary_room(config)
    differencies = []
    time = []
    integral = 0
    rooms_index_error = []
    while i < len(data.index):
        max_room, _ = found_max(data.iloc[i], config)
        max_room_error, index = found_max(data_error.iloc[i], config_error)
        rooms_index_error.append(index)
        state = data.loc[i, dictionary_rooms[data.loc[i, 'Room']]] / max_room
        state_error = data_error.loc[i, dictionary_rooms[data_error.loc[i, 'Room']]] / max_room_error
        temp = {'Time': data.loc[i, 'Time'], 'Efficiency real': state}
        temp_error = {'Time': data_error.loc[i, 'Time'], 'Efficiency fully-connected': state_error}
        df_error = df_error.append(temp_error, ignore_index=True)
        df = df.append(temp, ignore_index=True)
        differencies.append(state - state_error)
        integral += abs(differencies[i])
        time.append(data.loc[i, 'Time'])
        i += 1
    key_values = list(dictionary_rooms.keys())
    G = nx.Graph()
    for k in key_values:
        G.add_node(k)
    adj_dict = {}
    for i, c in enumerate(config["info"]["columns_name"][1:]):
        for n, col in enumerate(config["info"]["columns_name"][1:]):
            if i != n:
                adj_dict[str(c) + str(col)] = 0
    for i, r in enumerate(rooms_index_error):
        if rooms_index_error[i] != rooms_index_error[i - 1] and i != 0:
            adj_dict[str(rooms_index_error[i - 1]) + str(r)] += 1
            adj_dict[str(r) + str(rooms_index_error[i - 1])] += 1
            G.add_edge(key_values[list(dictionary_rooms.values()).index(str(rooms_index_error[i - 1]))],
                       key_values[list(dictionary_rooms.values()).index(str(r))])
    adj_dict = {k: 2 * v / sum(adj_dict.values(), 0.0) for k, v in adj_dict.items()}  # 2* because matrix is triangular
    edge_labels = {}
    for room in G.edges.keys():
        adj_value = '%.3f' % adj_dict[dictionary_rooms[room[0]] + dictionary_rooms[room[1]]]
        edge_labels[(room[0], room[1])] = adj_value
    df.to_csv(config["info"]["input_file_path"] + "/Output_evaluation/" + config["info"]["output_evaluation"],
              index=False)
    df_error.to_csv(
        config_error["info"]["input_file_path"] + "/Output_evaluation/" + config_error["info"]["output_evaluation"],
        index=False)
    fig = plt.figure()
    fig.set_size_inches(12, 9)
    plt.tight_layout()
    gs = gridspec.GridSpec(2, 3)
    ax0 = plt.subplot(gs[0, 0])  # row 0, col 0
    ax0.set(xlabel='Time', ylabel='Efficiency')
    df.plot(kind='line', x='Time', y='Efficiency real', ax=ax0, color="lightcoral")
    df_error.plot(kind='line', x='Time', y='Efficiency fully-connected', ax=ax0, linestyle='--', dashes=(6, 5), color="blue")
    ax1 = plt.subplot(gs[0, 1])  # row 0, col 1
    ax1.set_ylim([-1, 1])
    ax1.set(xlabel='Time', ylabel='Difference\nReal - Fully-connected')
    ax1.plot(time, differencies)
    ax2 = plt.subplot(gs[0, 2])  # row 0, col 2
    ax2.set(ylabel='Histogram filter normalized adjacencies')
    edge_labels_ordered = OrderedDict(sorted(edge_labels.items(), key=lambda x: x[1], reverse=True))
    dict_hist = dict_histog(config)
    objects = []
    performance = []
    for k in edge_labels_ordered.keys():
        objects.append(dict_hist[k[0]] + "-" + dict_hist[k[1]])
    for v in edge_labels_ordered.values():
        performance.append(float(v))
    bar_color = color_bar(open_json(config["info"]["adj_file"]), edge_labels_ordered)
    y_pos = np.arange(len(edge_labels_ordered.keys()))
    plt.bar(y_pos, performance, align='center', alpha=0.5, color=bar_color)
    plt.xticks(y_pos, objects, rotation=90)
    pos = nx.circular_layout(G, center=[1, 0], scale=1)  # graph
    ax3 = plt.subplot(gs[1, :])  # row 1, span all columns
    nx.draw_networkx_edge_labels(G, ax=ax3, pos=pos, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx(G, ax=ax3, node_size=1000, pos=pos, node_color="skyblue")
    integral_text = ax1.text(0, -0.5, 10, bbox={'facecolor': 'yellow', 'alpha': 0.7, 'pad': 10})
    integral_text.set_text("Integral: " + str(round(integral)))
    plt.tight_layout()
    plt.show()
    fig.savefig(config_error["info"]["input_file_path"] + "/Evaluate_images/" + config_error["info"]["img_evaluation"])


def main_dynamic(file_correct, hf_out_dynamic, window):
    """
    Main function that compare the efficiency between two HF_out given by one config.json and hf_out_dynamic.
    The function plot(1) the evaluation,the difference from the evaluation and the sum of this difference,the histogram of
    the adjacencies founded and the graph of these adjacencies.
    It also plot in another figure(2) the same as before except for the graph that is substituted by the Heatmap in
    ./Heatmap_images of the dynamic file given
    All the adjacencies are created when the max probability of the Histogram Filter change.
    The two finals figure is saved in motion-simulator/Evaluate_images/Dynamic

    :param file_correct: is the correct config.json (config.json,config_house1.json,config_house2.json)
    :param hf_out_dynamic: is the respective HF_out of config_fullyconn.json
    (config_fullyconn.json,config_fullyconn_house1.json,config_fullyconng_house2.json). This because the hf_out_dynamic
    is created from these files.
    :type: csv
    :param window: amount of minutes that identify every my dynamic file.(More details on Dynamic_Histogram_Filter)
    :type: int
    """
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    folder = "Configurations_file/"
    conf_file = folder + file_correct  # file1 correct
    config = open_json(conf_file)
    df = pd.DataFrame(columns=['Time', 'Efficiency'])
    df_error = pd.DataFrame(columns=['Time', 'Efficiency_error'])
    data = set_up(config)  # HF_out
    data_error = ReadFile.ReadFile(config["info"]["output_file_path"] + hf_out_dynamic).df
    i = 0
    dictionary_rooms = dictionary_room(config)
    differencies = []
    time = []
    integral = 0
    rooms_index_error = []
    while i < len(data.index):
        max_room, _ = found_max(data.iloc[i], config)
        max_room_error, index = found_max(data_error.iloc[i], config)
        rooms_index_error.append(index)
        state = data.loc[i, dictionary_rooms[data.loc[i, 'Room']]] / max_room
        state_error = data_error.loc[i, dictionary_rooms[data_error.loc[i, 'Room']]] / max_room_error
        temp = {'Time': data.loc[i, 'Time'], 'Efficiency': state}
        temp_error = {'Time': data_error.loc[i, 'Time'], 'Efficiency_error': state_error}
        df_error = df_error.append(temp_error, ignore_index=True)
        df = df.append(temp, ignore_index=True)
        differencies.append(state - state_error)
        integral += abs(differencies[i])
        time.append(data.loc[i, 'Time'])
        i += 1
    key_values = list(dictionary_rooms.keys())
    G = nx.Graph()
    for k in key_values:
        G.add_node(k)
    adj_dict = {}
    for i, c in enumerate(config["info"]["columns_name"][1:]):
        for n, col in enumerate(config["info"]["columns_name"][1:]):
            if i != n:
                adj_dict[str(c) + str(col)] = 0
    for i, r in enumerate(rooms_index_error):
        if rooms_index_error[i] != rooms_index_error[i - 1] and i != 0:
            adj_dict[str(rooms_index_error[i - 1]) + str(r)] += 1
            adj_dict[str(r) + str(rooms_index_error[i - 1])] += 1
            G.add_edge(key_values[list(dictionary_rooms.values()).index(str(rooms_index_error[i - 1]))],
                       key_values[list(dictionary_rooms.values()).index(str(r))])
    print(adj_dict)
    adj_dict = {k: 2 * v / sum(adj_dict.values(), 0.0) for k, v in adj_dict.items()}  # 2* because matrix is triangular
    edge_labels = {}
    for room in G.edges.keys():
        adj_value = '%.3f' % adj_dict[dictionary_rooms[room[0]] + dictionary_rooms[room[1]]]
        edge_labels[(room[0], room[1])] = adj_value
    df.to_csv(config["info"]["input_file_path"] + "/Output_evaluation/" + config["info"]["output_evaluation"],
              index=False)
    df_error.to_csv(
        config["info"]["input_file_path"] + "/Output_evaluation/" + "dynamic_" + str(window) + config["info"][
            "output_evaluation"],
        index=False)
    fig = plt.figure()
    fig.set_size_inches(14, 11)
    gs = gridspec.GridSpec(2, 3)
    ax0 = plt.subplot(gs[0, 0])  # row 0, col 0
    ax0.set(xlabel='Time', ylabel='Efficiency')
    df.plot(kind='line', x='Time', y='Efficiency', ax=ax0, color="lightcoral")
    df_error.plot(kind='line', x='Time', y='Efficiency_error', ax=ax0, linestyle='--', dashes=(8, 5), color="blue")
    ax1 = plt.subplot(gs[0, 1])  # row 0, col 1
    ax1.set_ylim([-1, 1])
    ax1.set(xlabel='Time', ylabel='Difference\nSimulation - Real')
    ax1.plot(time, differencies)
    integral_text = ax1.text(0, -0.5, 10, bbox={'facecolor': 'yellow', 'alpha': 0.7, 'pad': 10})
    integral_text.set_text("Integral: " + str(round(integral)))
    ax2 = plt.subplot(gs[0, 2])  # row 0, col 2
    ax2.set(ylabel='Histogram filter normalized adjacencies')
    edge_labels_ordered = OrderedDict(sorted(edge_labels.items(), key=lambda x: x[1], reverse=True))
    dict_hist = dict_histog(config)
    objects = []
    performance = []
    for k in edge_labels_ordered.keys():
        objects.append(dict_hist[k[0]] + "-" + dict_hist[k[1]])
    for v in edge_labels_ordered.values():
        performance.append(float(v))
    bar_color = color_bar(open_json(config["info"]["adj_file"]), edge_labels_ordered)
    y_pos = np.arange(len(edge_labels_ordered.keys()))
    plt.bar(y_pos, performance, align='center', alpha=0.5, color=bar_color)
    plt.xticks(y_pos, objects, rotation=90)


    ax3 = plt.subplot(gs[1,:])  # row 1, span all columns
    pos = nx.circular_layout(G, center=[1, 0], scale=1)  # graph
    nx.draw_networkx_edge_labels(G, ax=ax3, pos=pos, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx(G, ax=ax3, node_size=1000, pos=pos, node_color="skyblue")
    plt.tight_layout()
    #plt.show()
    fig.savefig(config["info"]["input_file_path"] + "/Evaluate_images/Dynamic//dynamic_house2/1dynamic" + str(window) + "_" + config["info"][
        "img_evaluation"])

    ax3.clear()
    heatmap = plt.imread("./Heatmap_images/Heatmap" + str(window) + ".png")
    ax3.axis("off")
    ax3.imshow(heatmap)
    fig.savefig(config["info"]["input_file_path"] + "/Evaluate_images/Dynamic/dynamic_house2/2dynamic" + str(window) + "_" + config["info"][
        "img_evaluation"])

if __name__ == "__main__":
    """main_static("config.json","config.json")"""

    for w in [700]:
        main_dynamic("config_house1.json", "dynamic"+str(w)+"minutes_HF_out_house1_fullyconn.csv", w)
