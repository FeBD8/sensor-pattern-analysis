import json
import sys

import numpy as np
from matplotlib import gridspec

import Heatmap
import Belief
import ReadFile
import pandas as pd
import matplotlib.pyplot as plt
import copy

DEBUG = True


def open_json(conf):
    with open(conf) as json_config:
        data_config = json.load(json_config)
    json_config.close()
    return data_config


def system_set_up(data_config):
    """
    Function that loads all the value from json file
    :param data_config: fullyconn.json file
    :return: Belief: the class with all probability values and rf:HF_input
    """
    rf = ReadFile.ReadFile(
        data_config["info"]["input_file_path"] + data_config["info"]["input_file_name"])  # legge HF_input
    bel = data_config["probability"]["bel_t0"]  # probabilita di essere in quella stanza all inizio
    pos = data_config["info"]["state_domain"]  # sono le iniziali delle stanze
    prob_state = []
    prefix = "prob"
    for i in pos:
        prob_state.append(
            data_config["probability"][prefix + i])  # probabilita di quale stanza si va dalla stanza in cui si è
    prefix = "s"
    ser = []
    movement_transaction = data_config["info"]["movement_transaction"]
    for i in pos:
        ser.append(data_config["sensor_error_probability"][
                       prefix + i])  # dopo la misura di un sensore probabilita di essere in quella stanza dove è avvenuto il cambiamento
    belief = Belief.Belief(bel, pos, prob_state, ser, movement_transaction)
    return belief, rf


def adj_dict(data_config):
    """
    Function that create two dictionaries

    :return:adj_dict [key='adjacencies made with the two index of probability' :value='number of adjacenies founded
    (start_value=0)']
    dict_col_index [key='columns probability names' : value='index of the column'
    """
    adj_dict = {}
    dict_col_index = {}
    for i, c in enumerate(data_config["info"]["columns_name"][1:]):
        dict_col_index[c] = i
        for n, col in enumerate(data_config["info"]["columns_name"][1:]):
            if i != n:
                adj_dict[str(c) + str(col)] = 0
    return adj_dict, dict_col_index


def found_max(df, df_1, adj_dict, dict_col_index, prob_state, correct_matrix):
    """
    Function that found every change between the probability given by the filter and update the Transition Matrix with Function
    new_prob_state.
    """
    res = df.iloc[1:]
    res_1 = df_1.iloc[1:]
    index_max = res.idxmax()
    index_max_1 = res_1.idxmax()
    if index_max != index_max_1:
        adj_dict[index_max_1 + index_max] += 1
        new_prob_state(prob_state, adj_dict, dict_col_index, index_max_1, correct_matrix)


def new_prob_state(prob_state, adj_dict, dict_col_index, index_1, correct_matrix):
    """
    Function that update the new Transitions Matrix.
    """
    prob_state[dict_col_index[index_1]][dict_col_index[index_1]] = correct_matrix["bel_t0"][0]
    for c in columns[1:]:
        if c != index_1:
            prob_state[dict_col_index[index_1]][dict_col_index[c]] = (
                    adj_dict[index_1 + c] / normalized_prob(index_1, adj_dict, correct_matrix))


def normalized_prob(index_1, adj_dict, correct_matrix):
    """
    Function that normalized all Transitions Matrix values of the row taken into consideration to the value of 1-autovalue.
    """
    sum = 0
    for c in columns[1:]:
        if c != index_1:
            sum += adj_dict[index_1 + c]
    normalized_sum = sum / (1-correct_matrix["bel_t0"][0])
    return normalized_sum


def check_measure(new_measure, previous_measure):
    out_sum = sum([x - y for x, y in zip(new_measure, previous_measure)])  # con zip faccio una tupla
    if out_sum == 0:
        return {}  # se dalla misura precedente non cambia nulla non ritornare nulla in transactions

    new_measure_str = [str(int(x)) for x in new_measure]
    previous_measure_str = [str(int(x)) for x in previous_measure]
    out_str = [x + y for x, y in zip(previous_measure_str, new_measure_str)]
    return out_str


def crate_file_output(df1: pd.DataFrame, df2, data_config, window):
    df3 = ReadFile.ReadFile(data_config["info"]["input_file_path"] +
                            data_config["info"][
                                "ground_truth_file_name"]).df  # questo df3 è out.csv che rappresenta dove la persona è
    df1["Room"] = ""  # df1 è HF_input e df2 sono le probabilità

    for i, row in df1.copy().iterrows():
        cond = df3['Time'] <= row["Time"]
        r = df3[cond].tail(1)['Room'].tolist()[
            0]  # prendo la riga in questione del ciclo e da li trovo la stanza da aggiungere
        df1.loc[i, 'Room'] = r  # aggiungo le stanze in cui la persona si trova realmente da out.csv

    df = pd.merge(df1, df2, how='inner')
    df.to_csv(
        data_config["info"]["input_file_path"] + "HF_out/dynamic" + str(window) + "minutes_" + data_config["info"][
            "output_file_name"],
        index=False)


def dictionary_room(data_config):
    key = data_config["info"]["columns_name"][1:]
    value = data_config["info"]["room_name"]
    dictionary = {}
    for i in range(0, len(key)):
        dictionary[key[i]] = value[i]
    return dictionary


def main(config_file, correct_file, window):
    """
    Main that create an HF_out with a dynamic Filter. It starts with a fully connected matrix and when it founds a
    adjacency it updates the row belonging to this adjacency(e.g.: atrium-kitchen the row is the first because atrium is
    the first room).
    Every auto adjacency(e.g.: atrium-atrium) is fixed to 1 / number of rooms
    The other values of the row are normalized to 1-auto adjacency

    :param window: fixed time before Histogram Filter becomes dynamic
    """
    global columns
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    folder = "Configurations_file/"
    conf_file = folder + config_file
    config = open_json(conf_file)
    belief, rf = system_set_up(config)  # carica e inizializza
    data_in = rf.df  # carico HF_input
    columns = config["info"]["columns_name"]
    sim_file = open_json(config["info"]["adj_file"])
    correct_file = open_json(folder + correct_file)
    correct_matrix = correct_file["probability"]
    df = pd.DataFrame(columns=columns)
    i = 0
    sensor_measures_previous = [0 for x in range(0, len(belief.bel))]
    adj_dic, dict_col_index = adj_dict(config)  # matrice adiacenze e matrice colonna-indice
    col_dic = dictionary_room(config)  # matrice colonna-stanza
    nps = []
    sim_adj = sim_file["room"]
    pos = config["info"]["state_domain"]  # sono le iniziali delle stanze
    for colum in columns[1:]:
        row = np.full(len(columns) - 1, 1 / (len(columns) - 1))
        nps.append(row)
    n_prob_state = np.array(nps)  # create the initial n_prob_state
    while i < len(data_in.index):
        time = data_in.iloc[i, 0]  # prendo la colonna time di HF_input
        sensor_measures = list(data_in.iloc[i])[1:]  # prendo le colonne con le misure dei sensori
        sensor_measures_str = [str(int(x)) for x in sensor_measures]
        transactions = check_measure(sensor_measures, sensor_measures_previous)
        if len(transactions) > 0:  # se qualche sensore è cambiato
            belief.bel_projected_upgrade()
            belief.bel_upgrade(transactions)

        tmp = {}
        values = [time] + belief.bel

        for j in range(0, len(columns)):
            tmp[columns[j]] = values[j]
        df = df.append(tmp, ignore_index=True)
        if i != 0:
            found_max(df.iloc[i], df.iloc[i - 1], adj_dic, dict_col_index, n_prob_state, correct_matrix)
        sensor_measures_previous = sensor_measures  # assegno a previous la misura precedente dei sensori
        i += 1
        if i > (60 * window) and i != 0:
            belief.prob_state = n_prob_state
    rooms = config["info"]["room_name"]
    fig = plt.figure()
    fig.set_size_inches(14, 11)
    gs = gridspec.GridSpec(1, 2)
    for colum in columns[1:]:  # make the wrong adjacencies negative for a better heatmap plot
        for c, col in enumerate(columns[1:]):
            neg = -1
            if col != colum:
                for room in sim_adj[col_dic[colum]]:
                    if room == col_dic[col]:
                        n_prob_state[dict_col_index[colum]][c] = abs(n_prob_state[dict_col_index[colum]][c])
                        break
                    else:
                        n_prob_state[dict_col_index[colum]][c] = n_prob_state[dict_col_index[colum]][c] * neg
                        neg = 1
    print(n_prob_state)
    print(adj_dic)

    # Graphic side
    ax0 = plt.subplot(gs[0, 0])  # row 0, col 0
    im0, cbar0 = Heatmap.heatmap_offset(n_prob_state, rooms, rooms, sim_adj, columns, dict_col_index, col_dic, ax=ax0,
                                        cmap="RdYlGn", cbarlabel="probability",
                                        cbar_kw={'ticks': [-0.8, 0.8]})
    cbar0.remove()
    texts = Heatmap.annotate_heatmap_GrRed(im0, data=n_prob_state, valfmt="{x:.3f}")

    ax1 = plt.subplot(gs[0, 1])  # row 0, col 1
    data2 = copy.copy(n_prob_state)
    for colum in pos[:]:  # see difference between these values and ideal ones
        for c, col in enumerate(columns[1:]):
            offset = 0
            if col == "bel(" + colum + ")":
                data2[dict_col_index["bel(" + colum + ")"]][c] = 0.5
            else:
                for room in sim_adj[col_dic["bel(" + colum + ")"]]:
                    if room == col_dic[col]:
                        if data2[dict_col_index["bel(" + colum + ")"]][c] - correct_matrix["prob" + colum][c] >= 0:
                            data2[dict_col_index["bel(" + colum + ")"]][c] = 0.5
                            offset = 0
                            break
                        else:
                            norm_0_5 = 0.5 / correct_matrix["prob" + colum][c]
                            data2[dict_col_index["bel(" + colum + ")"]][c] = 0.5 - norm_0_5 * (
                                        correct_matrix["prob" + colum][c] - data2[dict_col_index["bel(" + colum + ")"]][
                                    c])
                            offset = 0
                            break
                    else:
                        if data2[dict_col_index["bel(" + colum + ")"]][c] <= - 0.5:
                            data2[dict_col_index["bel(" + colum + ")"]][c] = 0
                            offset = 0
                            break
                        else:
                            offset = 0.5
                if offset == 0.5:
                    data2[dict_col_index["bel(" + colum + ")"]][c] += offset
    im1, cbar1 = Heatmap.heatmap(data2, rooms, rooms, sim_adj, columns, dict_col_index, col_dic, ax=ax1,
                                 cmap="Reds",
                                 cbar_kw={'ticks': [np.min(data2), 0.5]})
    cbar1.ax.set_yticklabels(['Bad', 'Good'])
    texts = Heatmap.annotate_heatmap_Reds(im1, data=n_prob_state, data2=data2, valfmt="{x:.3f}")

    # plt.tight_layout()
    fig.savefig("./Heatmap_images/" + "Heatmap" + str(window))
    crate_file_output(data_in, df, config, window)


if __name__ == "__main__":
    for v in [10]:
        main("config_fullyconn.json", "config.json", v)
