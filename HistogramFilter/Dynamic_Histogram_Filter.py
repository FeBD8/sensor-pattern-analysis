import json
import sys

import numpy as np
import Heatmap
import Belief
import ReadFile
import pandas as pd
import matplotlib.pyplot as plt


def open_json(conf):
    with open(conf) as json_config:
        data_config = json.load(json_config)
    json_config.close()
    return data_config


def system_set_up(data_config):
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
    adj_dict = {}
    dict_prob = {}
    for i, c in enumerate(data_config["info"]["columns_name"][1:]):
        dict_prob[c] = i
        for n, col in enumerate(data_config["info"]["columns_name"][1:]):
            if i != n:
                adj_dict[str(c) + str(col)] = 0
    return adj_dict, dict_prob


def found_max(df, df_1, adj_dict, dict_prob, prob_state, sim_adj, col_dic):
    res = df.iloc[1:]
    res_1 = df_1.iloc[1:]
    index_max = res.idxmax()
    index_max_1 = res_1.idxmax()
    if index_max != index_max_1:
        adj_dict[index_max_1 + index_max] += 1
        new_prob_state(prob_state, adj_dict, dict_prob, index_max_1, sim_adj, col_dic)


def new_prob_state(prob_state, adj_dict, dict_prob, index_1, sim_adj, col_dic):
    prob_state[dict_prob[index_1]][dict_prob[index_1]] = 0.2
    for c in columns[1:]:
        if c != index_1:
            prob_state[dict_prob[index_1]][dict_prob[c]] = (adj_dict[index_1 + c] / normalized_prob(index_1, adj_dict))


def normalized_prob(index_1, adj_dict):
    sum = 0
    for c in columns[1:]:
        if c != index_1:
            sum += adj_dict[index_1 + c]
    normalized_sum = sum / 0.8
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
    df.to_csv(data_config["info"]["input_file_path"] + "HF_out/dynamic" + str(window) +"minutes_"+ data_config["info"][
        "output_file_name"] ,
              index=False)


def dictionary_room(data_config):
    key = data_config["info"]["columns_name"][1:]
    value = data_config["info"]["room_name"]
    dictionary = {}
    for i in range(0, len(key)):
        dictionary[key[i]] = value[i]
    return dictionary


def main(window):
    global columns
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    folder = "Configurations_file/"
    conf_file = folder + "config_fullyconn.json"
    config = open_json(conf_file)
    belief, rf = system_set_up(config)  # carica e inizializza
    data_in = rf.df  # carico HF_input
    columns = config["info"]["columns_name"]
    sim_file = open_json(config["info"]["adj_file"])
    df = pd.DataFrame(columns=columns)
    i = 0
    sensor_measures_previous = [0 for x in range(0, len(belief.bel))]
    adj_dic, dict_prob = adj_dict(config)  # matrice adiacenze e matrice colonna-indice
    col_dic = dictionary_room(config)  # matrice colonna-stanza
    nps = []
    sim_adj = sim_file["room"]
    for colum in columns[1:]:
        row = np.full(len(columns) - 1, 1 / (len(columns) - 1))
        nps.append(row)
    n_prob_state = np.array(nps)
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
            found_max(df.iloc[i], df.iloc[i - 1], adj_dic, dict_prob, n_prob_state, sim_adj, col_dic)
        sensor_measures_previous = sensor_measures  # assegno a previous la misura precedente dei sensori
        i += 1
        if i % (60 * window) == 0 and i != 0:
            belief.prob_state = n_prob_state
    rooms = config["info"]["room_name"]
    fig, ax = plt.subplots()

    for colum in columns[1:]:
        for c, col in enumerate(columns[1:]):
            if col!=colum:
                for room in sim_adj[col_dic[colum]]:
                    if room == col_dic[col]:
                        n_prob_state[dict_prob[colum]][c] = abs(n_prob_state[dict_prob[colum]][c])
                        break
                    elif colum != col:
                        n_prob_state[dict_prob[colum]][c]= n_prob_state[dict_prob[colum]][c] * -1
    print(n_prob_state)
    print(adj_dic)
    im, cbar = Heatmap.heatmap(n_prob_state, rooms, rooms, ax=ax,
                               cmap="RdYlGn", cbarlabel="probability",
                               cbar_kw={'ticks': [np.min(n_prob_state),0.1, np.max(n_prob_state)]})
    cbar.ax.set_yticklabels(['Wrong\nlink','Weak\nReal\nlink', 'Real\nlink '])
    texts = Heatmap.annotate_heatmap(im, data=n_prob_state, valfmt="{x:.3f}")
    fig.tight_layout()
    fig.savefig("./Heatmap_images/" + "Heatmap" + str(window))
    crate_file_output(data_in, df, config, window)


if __name__ == "__main__":
    for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 166]:
        main(v)
