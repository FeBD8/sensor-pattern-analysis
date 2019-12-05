import Human as human
import pandas as pd
import Time as t
import UniformDDP as ddp
import config
import SimulationInfo as si
import ErrorLogger
import sys


def sensor_sample(apartment, current_time, mat, gateway, n_of_person, times):
    """
    make the sensor sample

    :param apartment: list of Room
        the rooms in the apartment
    :param current_time: int
        the timer of the system
    :param mat: Human
        the person in the house
    :param gateway: gateway
        system to store the data
    :return:
    """

    states = []
    for j in range(0, n_of_person):
        for i in apartment:
            i.alert_sensor(current_time, mat[j], times)  # verifico la misura di tutti i sensori
            states.append(i.sensor.state)

    gateway.update_df_HF(current_time, states)


def simulate(movement_tracker, time, mat, sensor_sample_time, gateway, n_of_person, times):
    """
    simulate the movement of the person

    :param movement_tracker: pandas.DataFrame
        dataframe of the person's movements
    :param time: float
        time of the system
    :param mat: Human
        the person in the apartment
    :param sensor_sample_time: float
        sensor sample time
    :return: tuple
        the DataFrame completed and the person
    """
    time_next_sample = time.START_TIME + sensor_sample_time  # misuro ogni secondo
    while time.current_time < time.STOP_TIME:
        for i in range(0, n_of_person):
            if time.check_time_delta(time.current_time, mat[i].time_next_move):
                mat[i].move(time.current_time)  # if timer is equal to human's timer-decision : human moves
                movement_tracker = movement_tracker.append({'Time': time.truncate(time.current_time, 3),
                                                            'Room': mat[i].current_room.name, 'Person': i},
                                                           ignore_index=True)
                times = times.append({'Time': time.truncate(time.current_time, 3)},
                                     ignore_index=True)

        if time.check_time_delta(time.current_time, time_next_sample):  # misuro ogni secondo con i sensori
            sensor_sample(apartment, time.current_time, mat, gateway,
                          n_of_person,
                          times)  # scrive cambiamenti dei sensori sia con alert sensor che con updategateway
            time_next_sample = time.current_time + sensor_sample_time  # misuro ogni secondo
        time.increase_time()  # aumento il tempo di un decimo di secondo
    #times.drop_duplicates("Time", inplace=True)
    return movement_tracker, times, mat


def check_running_mode(configurator):
    if configurator.init_test_mode() == "on":
        print("Running in test mode...")
    else:
        print("Simulating...")


def create_simulation_info(model, n_of_person):
    print("Creating simulation info file....")
    temp_swt_time_seed = []
    temp_lwt_time_seed = []
    for i in range(0, n_of_person):
        temp_lwt_time_seed.append(model[i][0].seed)
        temp_swt_time_seed.append(model[i][1].seed)

    sim_info = si.SimulationInfo(temp_lwt_time_seed, temp_swt_time_seed, n_of_person)
    sim_info.create_file()


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #    print("Manca il nome del file json")
    #    sys.exit(1)

    configurator = config.SystemConfig("configurations.json")
    check_running_mode(configurator)
    sensor_error_logger = ErrorLogger.ErrorLogger()
    n_of_person = configurator.init_person_number()

    time = t.Time(configurator.init_start_time(), configurator.init_stop_time(), configurator.init_time_epsilon(),
                  configurator.init_system_time_delta())

    sensor_sample_time = configurator.init_sensor_sample_time()
    apartment, gateway = configurator.create_apartment(sensor_error_logger)

    model = []
    mat = []
    for i in range(0, n_of_person):
        model.append([ddp.UniformDDp(configurator.init_long_model_lower(i), configurator.init_long_model_upper(i),
                                     configurator.init_long_model_seed(i), configurator.init_test_mode()),
                      ddp.UniformDDp(configurator.init_short_model_lower(i), configurator.init_short_model_upper(i),
                                     configurator.init_short_model_seed(i), configurator.init_test_mode())])
        mat.append(human.Human(apartment, model[i]))

    movement_tracker = pd.DataFrame(columns=['Time', 'Room', 'Person'])
    times = pd.DataFrame(columns=['Time'])
    for i in range(0, n_of_person):
        mat[i].chose_start_room(configurator.init_p_of_staying(i), configurator.init_p_type_behaviour(i))

    sensor_sample(apartment, time.current_time, mat, gateway, n_of_person, times)
    for i in range(0, n_of_person):
        movement_tracker = movement_tracker.append({'Time': time.current_time, 'Room': mat[i].current_room.name,
                                                    'Person': i}, ignore_index=True)  # inizializzo il file di out.csv
    times = times.append({'Time': time.current_time}, ignore_index=True)
    time.increase_time()

    ret = simulate(movement_tracker, time, mat, sensor_sample_time, gateway, n_of_person, times)

    ret[0].to_csv(configurator.name_output_gran_truth(), index=False)  # ha tutti i movimenti della persona in out.csv
    ret[2][0].current_room.sensor.gateway.dataframe.to_csv(configurator.name_output_sensor(),
                                                           index=False)  # scrive i cambiamenti dei sensori
    ret[1].to_csv(configurator.output_time(), index=False)  # scrivo ogni cambiamento
    gateway.df_HF.to_csv(configurator.name_output_sim(), index=False)  # scrive tutti 0/1 delle stanze per istante

    create_simulation_info(model, n_of_person)
