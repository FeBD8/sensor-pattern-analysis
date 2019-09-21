import sys
import matplotlib.pyplot as plt
from matplotlib import animation
import utility


def init_text(config):
    for state in config["text_position"]:
        plt.text(config["text_position"][state][0], config["text_position"][state][1], state, fontsize=17)


def found_state(row):
    for i in range(1, len(row)):
        if row[i] == 1:
            return str(i)


def update(conf, index):
    state = conf["index"][found_state(df.iloc[index])]
    fsm_state.set_color(conf["state_color"][state])
    fsm_state.center = conf["state_position"][state]


def init():
    update(configurator, 0)

    ax.add_patch(fsm_state)
    return fsm_state,


def animate(i):
    update(configurator, i)
    return fsm_state,


def set_image():
    img_center = plt.imread(configurator["image"]["gateway"])
    ax.imshow(img_center, extent=[configurator["image"]["position_g"][0], configurator["image"]["position_g"][1],
                                  configurator["image"]["position_g"][2], configurator["image"]["position_g"][3]])
    sensor_img = plt.imread(configurator["image"]["sensor"])

    for position in configurator["image"]["sensor_positions"]:
        ax.imshow(sensor_img, extent=[configurator["image"]["sensor_positions"][position][0],
                                      configurator["image"]["sensor_positions"][position][1],
                                      configurator["image"]["sensor_positions"][position][2],
                                      configurator["image"]["sensor_positions"][position][3]])


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Correct command = Animate_FSM.py name_of_the_configuration_file.json")
        sys.exit(1)

    configurator = utility.open_json(sys.argv[1])

    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(configurator["info"]["figure_size"][0], configurator["info"]["figure_size"][1])

    ax = plt.axes(xlim=(configurator["info"]["x_lim"][0], configurator["info"]["x_lim"][1]),
                  ylim=(configurator["info"]["y_lim"][0], configurator["info"]["y_lim"][1]))

    set_image()
    init_text(configurator)
    df = utility.read_file(configurator["info"]["input_file_Animate_FSM"])

    fsm_state = plt.Circle((2, 2), configurator["info"]["state_radius"], fc='b',  alpha=0.5)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(df.index)-1,
                                   interval=configurator["info"]["time_speed"], blit=True, repeat=False)

    plt.show()
