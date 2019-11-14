from datetime import datetime

import pandas as pd

df = pd.read_csv('sequenceofMS.csv', parse_dates=['Timestamp'])
df = df.set_index(df['Timestamp'])
df=df.resample('1S').pad()
rows={"Atrium":[],"Kitchen":[],"Livingroom":[],"Toilet":[],"Bedroom":[]}
for room in rows:
    rows[room].append(0)
i=0
for t,r in df.iterrows():
    for room in rows:
        if room==r["Class"]:
            if i == 0:
                rows[r["Class"]][i]=(r["Value"])
            else:
                rows[r["Class"]].append(r["Value"])
        else:
            if i!=0:
                rows[room].append(rows[room][i-1])
    print(t)
    i+=1
df2=pd.DataFrame(rows)
df2['TimeStamp']=df.index
df2.to_csv("sensor_real.csv",index_label="Time")
