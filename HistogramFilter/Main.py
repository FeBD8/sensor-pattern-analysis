import json
import sys

import Belief
import ReadFile
import pandas as pd


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


def check_measure(new_measure, previous_measure):
    out_sum = sum([x - y for x, y in zip(new_measure, previous_measure)])  # con zip faccio una tupla
    if out_sum == 0:
        return {}  # se dalla misura precedente non cambia nulla non ritornare nulla in transactions

    new_measure_str = [str(int(x)) for x in new_measure]
    previous_measure_str = [str(int(x)) for x in previous_measure]
    out_str = [x + y for x, y in zip(previous_measure_str, new_measure_str)]
    return out_str


def crate_file_output(df1: pd.DataFrame, df2, data_config):
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
    df.to_csv(data_config["info"]["input_file_path"] + "/HF_out/" + data_config["info"]["output_file_name"],
              index=False)


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    folder = "Configurations_file/"
    conf_file = folder + "config.json"
    config = open_json(conf_file)

    belief, rf = system_set_up(config)  # carica e inizializza
    data_in = rf.df  # carico HF_input
    columns = config["info"]["columns_name"]
    df = pd.DataFrame(columns=columns)
    i = 0
    sensor_measures_previous = [0 for x in range(0, len(belief.bel))]

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

        sensor_measures_previous = sensor_measures  # assegno a previous la misura precedente dei sensori
        i += 1

    crate_file_output(data_in, df, config)
