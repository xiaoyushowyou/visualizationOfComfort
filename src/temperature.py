from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def modify_fuxinglu():
    df = pd.read_csv("weather_outside.csv")
    # df =  pd.read_csv('Fuxing_lu_2017_hourly.csv')
    print df.shape
    print df.head(5)
    df.iloc[:,8].replace('\s+', np.nan,regex=True,inplace=True)
    df.iloc[:,9].replace('\s+', np.nan,regex=True,inplace=True)
    #df = df.dropna(axis=0,how="any")
    #df['Indoor-Avg CO2']= df['Indoor-Avg CO2'].str.replace(",", "")
    df2 = df.iloc[:,8:10].as_matrix()
    
    df2 = df2.astype(np.float)
    print df2.shape
    abs_hum = 0.001225*6.112*np.exp(17.67*df2[:,1]/(df2[:,1]+243.5))*df2[:,0]*18.02/((273.15+df2[:,1])*100*0.08314)
    abs_hum = pd.DataFrame(abs_hum)
    df = pd.concat([df,abs_hum],axis=1)
    df.to_csv('Fuxing_lu_2017_hourly_withabshum.csv',index=False)

def modify():
    df =  pd.read_csv('outdoor.csv')
    print df.shape
    df.replace('\s+', '_',regex=True,inplace=True)
    #df['Indoor-Avg RH']= df['Indoor-Avg RH'].str.replace('\s+', '_')
    df = df[df['Outdoor-Avg Temp']!='_']
    #& df['Indoor-Avg Temp']!='_' & df['Indoor-Avg RH']!='_' &  df['Indoor-Avg CO2']!='_'
    print df.iloc[17,1]
    print df.shape
    df = df[df['Outdoor-Avg RH'] != '_']
    df = df[df['Indoor-Avg Temp'] !='_']

    print df.shape
    df = df[df['Indoor-Avg RH'] !='_']
    df = df[df['Indoor-Avg CO2'] !='_']
    print df.shape


    df =  pd.read_csv('outdoor.csv')
    df.iloc[:,1:5].replace('\s+', np.nan,regex=True,inplace=True)
    df = df.dropna(axis=0,how="any")
    df['Indoor-Avg CO2']= df['Indoor-Avg CO2'].str.replace(",", "")
    df2 = df.iloc[:,1:5].as_matrix()

    df2 = df2.astype(np.float)
    print df2[0]
    #corner = [40.0,50.0,40.0,50.0]
    #corner2 = [40.0,0.0,40.0,0.0]
    #cornoer3 = [-10.0,0.0,-10.0,0.0]
    df2[0] = [40.0,50.0,40.0,50.0]
    df2[1] = [40.0,0.0,40.0,0.0]
    df2[2] = [-10.0,0.0,-10.0,0.0]
    print df2[:5]
    outdoor_abs_hum = 0.001225*6.112*np.exp(17.67*df2[:,0]/(df2[:,0]+243.5))*df2[:,1]*18.02/((273.15+df2[:,0])*100*0.08314)
    indoor_abs_hum = 0.001225*6.112*np.exp(17.67*df2[:,2]/(df2[:,2]+243.5))*df2[:,3]*18.02/((273.15+df2[:,2])*100*0.08314)
    temp1 = np.column_stack([df2[:,0],outdoor_abs_hum,np.repeat(1,df2.shape[0])])
    temp2 = np.column_stack([df2[:,2],indoor_abs_hum,np.repeat(0,df2.shape[0])])
    df3 = pd.DataFrame(np.concatenate((temp1, temp2), axis=1),columns=['Out_Avg_Temp', 'Out_RH','Out_label','In_Avg_Temp', 'In_RH','In_label'])
    #df3 = pd.DataFrame(np.concatenate((temp1, temp2), axis=0),columns=['Avg_Temp', 'RH','label'])
    df3 = df3.iloc[0:10000,:]
    df3.to_csv('plot_data_1000.csv', columns=['Out_Avg_Temp', 'Out_RH','Out_label','In_Avg_Temp', 'In_RH','In_label'], index=False)
    #groups = df3.groupby('label')
    #fig,ax = plt.subplots()
    #for name, group in groups:
    #ax.plot(group.Avg_Temp, group.RH, marker='o', linestyle='', ms=3, alpha=0.5,label=name)
    #ax.legend()
#plt.show()

#print df3.shape
#df2 = pd.DataFrame(df2,columns=['Outdoor-Avg Temp', 'Outdoor-Avg RH', 'Indoor-Avg Temp','Indoor-Avg RH','Outdoor_Abs_Hum','Indoor_Abs_Hum'])

    #label outdoor is 1; indoor is 0
    #6.11* *df2[:,2]*18.02/((273.15+df[:,1])*100*.08314)

#np.repeat(243.5,df2.shape[0])
#plt.scatter(df2[:,0],outdoor_abs_hum)
#plt.show()
# df['Absolute_Humidity'] =


if __name__ == "__main__":
#modify()
    modify_fuxinglu()
