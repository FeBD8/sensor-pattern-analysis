import json
import Belief
import ReadFile
import pandas as pd


def open_json():
    with open("config_ap_two.json") as json_config:
        data_config = json.load(json_config)
    json_config.close()
    return data_config


def system_set_up():
    data_config = open_json()
    rf = ReadFile.ReadFile(data_config["info"]["input_file_path"]+data_config["info"]["input_file_name"])
    bel = data_config["probability"]["bel_t0"]
    pos = data_config["info"]["state_domain"]
    prob_state = []
    prefix = "prob"
    for i in pos:
        prob_state.append(data_config["probability"][prefix+i])
    prefix = "s"
    ser = []
    movement_transaction = data_config["info"]["movement_transaction"]
    exit_transaction = data_config["info"]["exit_transaction"]
    prob_exit_room = data_config["info"]["prob_exit_room"]

    for i in pos:
        ser.append(data_config["sensor_error_probability"][prefix+i])
    belief = Belief.Belief(bel, pos, prob_state, ser, movement_transaction, exit_transaction, prob_exit_room)
    return belief, rf


def check_measure(new_measure, previous_measure):
    out_sum = sum([x - y for x, y in zip(new_measure, previous_measure)])

    if out_sum == 0:
        return {}

    new_measure_str = [str(int(x)) for x in new_measure]
    previous_measure_str = [str(int(x)) for x in previous_measure]
    out_str = [x + y for x, y in zip(previous_measure_str, new_measure_str)]
    return out_str


def crate_file_output(df1: pd.DataFrame, df2):
    data_config = open_json()
    df3 = ReadFile.ReadFile(data_config["info"]["input_file_path"]+
                            data_config["info"]["ground_truth_file_name"]).df
    df1["Room"] = ""

    for i, row in df1.copy().iterrows():

        cond = df3['Time'] <= row["Time"]
        r = df3[cond].tail(1)['Room'].tolist()[0]
        df1.loc[i, 'Room'] = r

    df = pd.merge(df1, df2, how='inner')
    df.to_csv(data_config["info"]["input_file_path"]+data_config["info"]["output_file_name"], index=False)


if __name__ == "__main__":

    belief, rf = system_set_up()
    data_in = rf.df
    columns = open_json()["info"]["columns_name"]
    df = pd.DataFrame(columns=columns)
    i = 0
    sensor_measures_previous = [0 for x in range(0, len(columns)-1)]

    while i < len(data_in.index):

        time = data_in.iloc[i, 0]
        sensor_measures = list(data_in.iloc[i])[1:]
        sensor_measures_str = [str(int(x)) for x in sensor_measures]
        transactions = check_measure(sensor_measures, sensor_measures_previous)

        if len(transactions) > 0:
            belief.bel_upgrade(transactions)
            
        tmp = {}
        values = [time] + belief.bel

        for j in range(0, len(columns)):
            tmp[columns[j]] = values[j]

        df = df.append(tmp, ignore_index=True)

        sensor_measures_previous = sensor_measures
        i += 1

    crate_file_output(data_in, df)


