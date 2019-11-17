import utility

files=[]
files.append(utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/Create_pattern/RealRoomSensor/atrium.csv"))
files.append(utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/Create_pattern/RealRoomSensor/toilet.csv"))
files.append(utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/Create_pattern/RealRoomSensor/bedroom.csv"))
files.append(utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/Create_pattern/RealRoomSensor/kitchen.csv"))
files.append(utility.read_file("/home/riccardo/PycharmProjects/sensor-pattern-analysis/Create_pattern/RealRoomSensor/livingroom.csv"))
errors=0
wrong_rows={"Atrium":[],"Kitchen":[],"Livingroom":[],"Toilet":[],"Bedroom":[]}
for file in files:
    for i,r in file.iterrows():
        if i!=0:
            if r["Value"]==file["Value"][i-1]:
                errors+=1
                wrong_rows[r["Class"]].append(i+2)  # index start from row 2
print(wrong_rows)  # all are rows between two days