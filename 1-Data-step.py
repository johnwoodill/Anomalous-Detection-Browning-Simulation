
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = indat[indat['hour'] == 0]

def NN_dist(data=None, lat_lis=None, lon_lis=None):

    # In km
    r = 6371.0088

    data = data.dropna()
    mmsi = data.mmsi
    data = data.sort_values('mmsi')
    lat_lis = data['lat']
    lon_lis = data['lon']
    timestamp = data['hour'].iat[0]

    lat_mtx = np.array([lat_lis]).T * np.pi / 180
    lon_mtx = np.array([lon_lis]).T * np.pi / 180

    cos_lat_i = np.cos(lat_mtx)
    cos_lat_j = np.cos(lat_mtx)
    cos_lat_J = np.repeat(cos_lat_j, len(lat_mtx), axis=1).T

    lat_Mtx = np.repeat(lat_mtx, len(lat_mtx), axis=1).T
    cos_lat_d = np.cos(lat_mtx - lat_Mtx)

    lon_Mtx = np.repeat(lon_mtx, len(lon_mtx), axis=1).T
    cos_lon_d = np.cos(lon_mtx - lon_Mtx)

    mtx = r * np.arccos(cos_lat_d - cos_lat_i*cos_lat_J*(1 - cos_lon_d))

    # Build data.frame
    matdat = pd.DataFrame(mtx)
    matdat.columns = mmsi[:]
    matdat = matdat.set_index(mmsi[:])

    # Stack and form three column data.frame
    tmatdat = matdat.stack()
    lst = tmatdat.index.tolist()
    vessel_A = pd.Series([item[0] for item in lst])
    vessel_B = pd.Series([item[1] for item in lst])
    distance = tmatdat.values

    # Get lat/lon per mmsi
    posdat = data[['mmsi', 'lat', 'lon']]
    posdat = posdat.sort_values('mmsi')

    # Build data frame
    odat = pd.DataFrame({'timestamp': timestamp, 'vessel_A': vessel_A,
                         'vessel_B': vessel_B, 'distance': distance})
    odat = odat.sort_values(['vessel_A', 'distance'])

    # Get 05-NN
    odat = odat.sort_values('distance').groupby(
        'vessel_A', as_index=False).nth([0, 1, 2, 3, 4, 5])
    odat = odat.sort_values(['vessel_A', 'distance'])

    # Merge in vessel_B lat/lon
    posdat.columns = ['mmsi', 'vessel_B_lat', 'vessel_B_lon']
    odat = odat.merge(posdat, how='left', left_on='vessel_B', right_on='mmsi')

    # Merge in vessel_A lat/lon
    posdat.columns = ['mmsi', 'vessel_A_lat', 'vessel_A_lon']
    odat = odat.merge(posdat, how='left', left_on='vessel_A', right_on='mmsi')

    odat['NN'] = odat.groupby(['vessel_A'], as_index=False).cumcount()
    odat = odat.reset_index(drop=True)
    odat = odat[['timestamp', 'vessel_A', 'vessel_B', 'vessel_A_lat',
                 'vessel_A_lon', 'vessel_B_lat', 'vessel_B_lon', 'NN', 'distance']]
    odat = odat.sort_values(['vessel_A', 'NN'])

    # Data check: Ensure have 5 NN
    nn5 = odat.sort_values('NN').groupby('vessel_A').tail(1)
    
    nn5 = nn5[nn5['NN'] == 5]
    
    unique_nn5 = nn5['vessel_A'].unique()

    odat = odat[odat.vessel_A.isin(unique_nn5)]

    return odat




#xdis = rnorm(N, 0 ,1)
#ydis = rnorm(N, 0 ,1)
#xdis = cumsum(xdis)
#ydis = cumsum(ydis)
#plot(xdis, ydis, type="l")

#

N = 31*24

# Prior to event
indat = pd.DataFrame()
for i in range(0, 100):
    lat_dis1 = np.random.normal(0, .5, 288)
    lon_dis1 = np.random.normal(0, .5, 288)
    
    lat_dis2 = np.random.normal(0, 4, 119)
    lon_dis2 = np.random.normal(0, 4, 119)

    lat_dis3 = np.random.normal(0, .5, 337)
    lon_dis3 = np.random.normal(0, .5, 337)

    lat_dis = np.concatenate([lat_dis1, lat_dis2, lat_dis3])
    lon_dis = np.concatenate([lon_dis1, lon_dis2, lon_dis3])

    lat = np.cumsum(lat_dis)
    lon = np.cumsum(lon_dis)

    outdat = pd.DataFrame({"hour": range(N),
                            "mmsi": i,
                            "lon": lon,
                            "lat": lat})
    indat = pd.concat([indat, outdat])

indat.head()

test = indat[indat['mmsi'] == 0]

# Get single vessel path
sns.set_style("darkgrid")
plt.plot(test['lon'], test['lat'])
plt.show()


odat = indat.groupby('hour', as_index=False).apply(NN_dist)

odat.head()

dis1 = odat[odat['timestamp'] <= 288]
dis2 = odat[(odat['timestamp'] >= 289) & (odat['timestamp'] <= 407)]
dis3 = odat[odat['timestamp'] >= 408]

dis1 = dis1[dis1['distance'] != 0]
dis2 = dis2[dis2['distance'] != 0]
dis3 = dis3[dis3['distance'] != 0]

dis1 = dis1.groupby(['vessel_A', 'timestamp'], as_index=False)['distance'].mean()
dis2 = dis2.groupby(['vessel_A', 'timestamp'], as_index=False)['distance'].mean()
dis3 = dis3.groupby(['vessel_A', 'timestamp'], as_index=False)['distance'].mean()

sns.distplot(np.log(1 + dis1['distance']))

sns.distplot(np.log(1 + dis2['distance']))

sns.distplot(np.log(1 + dis3['distance']))






