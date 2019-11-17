import sys

import ReadFile
import json
import pandas as pd
import matplotlib.pyplot as plt

"""
    Script used to evaluate the output of the histogram filter after a simulation.
    It calculates the relative frequency between the number of time that the person is 
    in a room and the room is the one  the highest probability to be in for the histogram filter. 
"""


def open_json(file_name):
    with open(file_name) as json_config:
        data_config = json.load(json_config)
    json_config.close()
    return data_config


def set_up(data_config):
    data = ReadFile.ReadFile(data_config["info"]["input_file_path"]+data_config["info"]["output_file_name"]).df
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


def found_max(probability_arrrey: pd.Series, data_config):
    constant = len(data_config["info"]["state_domain"]) + 2
    val_max = probability_arrrey.iloc[constant:].apply(float).max()
    return val_max


if __name__ == "__main__" :
    #if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)
    conf_file = "config.json"
    config = open_json(conf_file)
    conf_real_adj_file="config_error.json"
    config_error=open_json(conf_real_adj_file)
    df = pd.DataFrame(columns=['Time', 'Efficiency'])
    df_error=pd.DataFrame(columns=['Time', 'Efficiency_error'])

    data = set_up(config)  # HF_input
    data_error=set_up(config_error)
    i = 0
    dictionary_rooms = dictionary_room(config)
    differencies =[]
    time=[]
    while i < len(data.index):
        max_room = found_max(data.iloc[i], config)
        max_room_error=found_max(data_error.iloc[i], config_error)
        state = data.loc[i, dictionary_rooms[data.loc[i, 'Room']]]/max_room
        state_error = data_error.loc[i, dictionary_rooms[data_error.loc[i, 'Room']]]/max_room_error
        temp = {'Time': data.loc[i, 'Time'],'Efficiency': state}
        temp_error={'Time': data_error.loc[i, 'Time'],'Efficiency_error': state_error}
        df_error = df_error.append(temp_error, ignore_index=True)
        df = df.append(temp, ignore_index=True)
        differencies.append(state-state_error)
        if(state-state_error<-0.5):
            print(max_room)
            print(max_room_error)
            print(str(data.loc[i, dictionary_rooms[data.loc[i, 'Room']]]) + " state "+str(state) +"  "+ str(i))
            print(str(data_error.loc[i, dictionary_rooms[data_error.loc[i, 'Room']]]) + " state_error "+str(state_error) +"  "+ str(i))
        time.append(data.loc[i, 'Time'])
        i += 1

    df.to_csv(config["info"]["input_file_path"]+config["info"]["output_evaluation"], index=False)
    df_error.to_csv(config_error["info"]["input_file_path"]+config_error["info"]["output_evaluation"], index=False)
    fig, axes = plt.subplots(2)
    fig.set_size_inches(12,9)
    axes[0].set(xlabel='Time', ylabel='Efficiency')
    df.plot(kind='line', x='Time', y='Efficiency',ax=axes[0])
    df_error.plot(kind='line', x='Time', y='Efficiency_error',ax=axes[0])
    axes[1].set(xlabel='Time', ylabel='Difference\nSimulation - Real')
    axes[1].plot(time,differencies)
    plt.show()
    fig.savefig(config_error["info"]["input_file_path"]+config_error["info"]["img_evaluation"])



