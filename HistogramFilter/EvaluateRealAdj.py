import sys

import networkx as nx
from matplotlib import gridspec

import ReadFile
import json
import pandas as pd
import matplotlib.pyplot as plt

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
    data = ReadFile.ReadFile(data_config["info"]["input_file_path"] + data_config["info"]["output_file_name"]).df
    return data


"""
    Dictionary [key='the name of the room' : value='the index of the column with the probability of the room' 
"""


def dictionary_room(data_config):
    key = data_config["info"]["room_name"]
    value = data_config["info"]["columns_name"][1:]
    dictionary = {}
    for i in range(0, len(key)):
        dictionary[key[i]] = value[i]
    return dictionary


"""
    Found the max value between the probability of the rooms 
"""


def found_max(probability_array: pd.Series, data_config):
    constant = len(data_config["info"]["state_domain"]) + 2
    res = probability_array.iloc[constant:].apply(float)
    val_max = res.max()
    index_max = res.idxmax()
    return val_max, index_max


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    conf_file = "config.json"
    config = open_json(conf_file)
    conf_real_adj_file = "config_notequals.json"
    config_error = open_json(conf_real_adj_file)
    df = pd.DataFrame(columns=['Time', 'Efficiency'])
    df_error = pd.DataFrame(columns=['Time', 'Efficiency_error'])
    data = set_up(config)  # HF_input
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

    adj_dict = {"bel(A)bel(K)": 0, "bel(A)bel(L)": 0, "bel(A)bel(T)": 0, "bel(A)bel(B)": 0, "bel(K)bel(A)": 0,
                "bel(K)bel(L)": 0, "bel(K)bel(T)": 0, "bel(K)bel(B)": 0, "bel(L)bel(A)": 0, "bel(L)bel(K)": 0,
                "bel(L)bel(T)": 0, "bel(L)bel(B)": 0, "bel(T)bel(A)": 0, "bel(T)bel(K)": 0, "bel(T)bel(L)": 0,
                "bel(T)bel(B)": 0, "bel(B)bel(A)": 0, "bel(B)bel(K)": 0, "bel(B)bel(L)": 0, "bel(B)bel(T)": 0
                }
    edge_labels = {}
    for i, r in enumerate(rooms_index_error):
        if rooms_index_error[i] != rooms_index_error[i - 1] and i != 0:
            adj_dict[str(rooms_index_error[i - 1]) + str(r)] += 1
            adj_dict[str(r) + str(rooms_index_error[i - 1])] += 1
            G.add_edge(key_values[list(dictionary_rooms.values()).index(str(rooms_index_error[i - 1]))],
                       key_values[list(dictionary_rooms.values()).index(str(r))])
    adj_dict = {k: 2*v / sum(adj_dict.values(), 0.0) for k, v in adj_dict.items()}  #2* because matrix is triangular
    # TODO grafici con piu stanze
    for room in G.edges.keys():
        adj_value = '%.3f' % adj_dict[dictionary_rooms[room[0]] + dictionary_rooms[room[1]]]
        edge_labels[(room[0], room[1])] = adj_value
    df.to_csv(config["info"]["input_file_path"] + config["info"]["output_evaluation"], index=False)
    df_error.to_csv(config_error["info"]["input_file_path"] + config_error["info"]["output_evaluation"], index=False)
    fig = plt.figure()
    fig.set_size_inches(12, 9)
    gs = gridspec.GridSpec(2, 2)
    ax0=plt.subplot(gs[0, 0])  # row 0, col 0
    ax0.set(xlabel='Time', ylabel='Efficiency')
    df.plot(kind='line', x='Time', y='Efficiency', ax=ax0)
    df_error.plot(kind='line', x='Time', y='Efficiency_error', ax=ax0)
    ax1 = plt.subplot(gs[0, 1])  # row 0, col 1
    ax1.set_ylim([-1, 1])
    ax1.set(xlabel='Time', ylabel='Difference\nSimulation - Real')
    ax1.plot(time, differencies)
    pos = nx.spring_layout(G, center=[1, 0], scale=1)
    ax2 = plt.subplot(gs[1, :])  # row 1, span all columns
    nx.draw_networkx_edge_labels(G, ax=ax2, pos=pos, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx(G, ax=ax2, node_size=1000, pos=pos,node_color="skyblue")
    integral_text = ax1.text(0, -0.5, 10, bbox={'facecolor': 'yellow', 'alpha': 0.7, 'pad': 10})
    integral_text.set_text("Integral: " + str(round(integral)))
    plt.show()
    fig.savefig(config_error["info"]["input_file_path"] + config_error["info"]["img_evaluation"])
