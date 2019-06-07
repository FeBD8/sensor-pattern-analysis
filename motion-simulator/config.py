import Room
import Gateway
import Sensor
import pandas as pd
import json


class SystemConfig:

    """ Class which configures the simulator reading from configurations.json
        ...

        Attributes
        ------------------------
        data_config : dict
            contains all the values of the variables which have to be initialize by the user

        Methods
        -----------------------

        create_apartment(self) : list of Room
            configure the apartment in which the simulation takes place
            initialize the sensor in each Room
            initialize the Gateway to whom the sensor send the data

        init_start_time(self): float
            set the time in which the simulation starts

        init_stop_time(self): float
            set the time in which the simulation ends

        init_mean(self): float
            set the mean of the Normal distribution(on which the person permanence in a room is decided randomly)

        init_std(self): float
            set the standard deviation of the Normal distribution(on which the person permanence in a room is decided
                randomly)

        init_seed(self): int
            set the seed for numpy.random.normal() (** used only in test mod **)

        init_system_time_delta(self) : float
            set the quantum of time by which the timer is incremented

        init_sensor_sample_time(self) : float
            initialize sensors sample time

        init_time_epsilon(self) : float
            set the precision of the comparison between two float (to determine if they are similar enough)

        init_sensor_prob_error(self) : float
            set the probability of sensor to fail(not to reveal movement)

    """

    def __init__(self):
        with open("configurations.json") as json_config:
            self.data_config = json.load(json_config)
        json_config.close()

    @property
    def data_config(self):
        return self.__data_config

    @data_config.setter
    def data_config(self, data_config):
        self.__data_config = data_config

    def create_apartment(self):

        apartment = []
        col = ["Time"]
        col = col + (list(self.data_config["room"].keys()))

        g = Gateway.Gateway(pd.DataFrame(columns=['Time', 'Room', 'State']), pd.DataFrame(columns=col), col[1:])

        for i in self.data_config["room"]:
            tmp = Room.Room(i, 0, Sensor.Sensor(i, g, self.data_config["time"]["sensor_sleep_time"],
                                                self.data_config["other"]["sensor_prob_error"]))
            apartment.append(tmp)

        for i in apartment:
            i.adjacencies = []
            for room_name in self.data_config["room"][i.name]:
                for j in apartment:
                    if room_name == j.name:
                        i.adjacencies.append(j)

        return apartment, g

    def init_start_time(self):
        return self.data_config["time"]["START_TIME"]

    def init_stop_time(self):
        return self.data_config["time"]["STOP_TIME"]

    def init_mean(self):
        return self.data_config["time"]["mu_waiting_time"]

    def init_std(self):
        return self.data_config["time"]["std_waiting_time"]

    def init_seed(self):
        return self.data_config["time"]["seed_waiting_time"]

    def init_system_time_delta(self):
        return self.data_config["time"]["system_time_delta"]

    def init_sensor_sample_time(self):
        return self.data_config["time"]["sensor_sample_time"]

    def init_time_epsilon(self):
        return self.data_config["time"]["epsilon"]

    def init_sensor_prob_error(self):
        return self.data_config["other"]["sensor_prob_error"]

