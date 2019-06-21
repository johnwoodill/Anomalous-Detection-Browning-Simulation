import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as ssd
from scipy.spatial import distance
from sklearn.manifold import MDS
from sklearn.manifold import Isomap

# Calculate JSD metric
def f_js(x, y):
    return distance.jensenshannon(x, y)

# Function calculates NN=5 distances
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

# Calculate distance matrix
def jsd_matrix(dat, interval, NN=0):
    if interval == 'dayhour':
        dat = dat.groupby(['vessel_A', 'timestamp'],
                          as_index=False)['distance'].mean()
        x = []

        gb = dat.groupby(['timestamp'])['distance']
        lst = [gb.get_group(x) for x in gb.groups]
        x = []
        for i in range(len(lst)):
            for j in range(len(lst)):
                x += [(i, j, f_js(lst[i], lst[j]))]

        distMatrix = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
        distMatrix = np.matrix(distMatrix)

    return (distMatrix)



# Brownian motion 2-dimension simulation data
# 100 vessels
# location centered around 0 
# movement sd = 0.01
# N = Number of hours

N = 31*24

indat = pd.DataFrame()
for i in range(0, 100):

    # Get random sample data
    # Set random seed
    lat_dis = np.random.normal(0, .01, N)
    lon_dis = np.random.normal(0, .01, N)
    
    # Cumsum to get movement through space
    lat = np.cumsum(lat_dis)
    lon = np.cumsum(lon_dis)

    # Data frame to merge with indat
    outdat = pd.DataFrame({"hour": range(N),
                            "mmsi": i,
                            "lon": lon,
                            "lat": lat})
    # merge dat
    indat = pd.concat([indat, outdat])

# Sample data
indat.head()

# Plot single vessel
test = indat[indat['mmsi'] == 0]
plt.plot(test['lon'], test['lat'])


# Get NN distances for each vessel in each hour
odat = indat.groupby('hour', as_index=False).apply(NN_dist)

# Print distance range
print(f"Min Distance: {min(odat['distance'])}  Max Distance: {max(odat['distance'])}")

# Remove first observation vessel_A = vessel_A distance = 0
odat = odat[odat['distance'] != 0]

# Subset out event day and shock system
# Day 1 - 13
dis1 = odat[odat['timestamp'] <= 288]

# Day 14 - 17
dis2 = odat[(odat['timestamp'] >= 289) & (odat['timestamp'] <= 407)]

# Day 18 - 31
dis3 = odat[odat['timestamp'] >= 408]

# Shock system using abs value of normal distribution during event
# Set random seed
dist = np.abs(np.random.normal(50, 5, len(dis2)))

# Output of shock values
print(min(dist), max(dist))

# Apply shock
dis2.loc[:, 'distance'] = dis2.loc[:, 'distance'] + dist

#sdis = dis2.loc[:, 'distance'] + dist

# Merge data back
dis = pd.concat([dis1, dis2, dis3])

# Get average NN distance for each vessel in each day
dis = dis.groupby(['vessel_A', 'timestamp'], as_index=False)['distance'].mean()

# Distribution plots
sns.distplot(np.log(1 + dis1['distance']))
sns.distplot(np.log(1 + dis2['distance']))
#sns.distplot(np.log(sdis))
sns.distplot(np.log(1 + dis3['distance']))


# Calculate JSD matrix
dmat = jsd_matrix(dis, "dayhour")

# Check matrix
print(dmat)

np.save('/home/server/pi/homes/woodilla/Projects/Anomalous-Detection-Browning-Simulation/data/jsd_mat.npy', dmat)

#------------------------------------------------------------------------
# Metric-MDS 5-dimensions
nmds = MDS(n_components=5, metric=True, dissimilarity='precomputed')
nmds_dat = nmds.fit_transform(dmat)

ndat = pd.DataFrame({"x": nmds_dat[:, 0], "y": nmds_dat[:, 1]})

ndat['x2'] = ndat['x'].shift(-1)
ndat['y2'] = ndat['y'].shift(-1)

ndat['distance'] = np.sqrt( (ndat['x'] - ndat['x2'])**2 + (ndat['y'] - ndat['y2'])**2 )

# Calculate speed
ndat['speed'] = ndat['distance']/1

# Plot speed
plt = sns.scatterplot(x=range(len(ndat)) ,y=ndat['speed'], edgecolor="black")
plt.axvline(289, ls='--', color='white')
plt.axvline(407, ls='--', color='white')

#------------------------------------------------------------------------
# Nonmetric-MDS 5-dimensions
nmds = MDS(n_components=5, metric=False, dissimilarity='precomputed')
nmds_dat = nmds.fit_transform(dmat)

ndat = pd.DataFrame({"x": nmds_dat[:, 0], "y": nmds_dat[:, 1]})

ndat['x2'] = ndat['x'].shift(-1)
ndat['y2'] = ndat['y'].shift(-1)

ndat['distance'] = np.sqrt( (ndat['x'] - ndat['x2'])**2 + (ndat['y'] - ndat['y2'])**2 )

# Calculate speed
ndat['speed'] = ndat['distance']/1


# Plot speed
plt = sns.scatterplot(x=range(len(ndat)) ,y=ndat['speed'], edgecolor="black")
plt.axvline(289, ls='--', color='white')
plt.axvline(407, ls='--', color='white')

#------------------------------------------------------------------------
# nMDS using ISO Mapping
nmds = Isomap(n_components=5)
nmds_dat = nmds.fit_transform(dmat)
ndat = pd.DataFrame({"x": nmds_dat[:, 0], "y": nmds_dat[:, 1]})

ndat['x2'] = ndat['x'].shift(-1)
ndat['y2'] = ndat['y'].shift(-1)

ndat['distance'] = np.sqrt( (ndat['x'] - ndat['x2'])**2 + (ndat['y'] - ndat['y2'])**2 )

# Calculate speed
ndat['speed'] = ndat['distance']/1


# Plot speed
plt = sns.scatterplot(x=range(len(ndat)) ,y=ndat['speed'], edgecolor="black")
plt.axvline(289, ls='--', color='white')
plt.axvline(407, ls='--', color='white')



# Limit outliers
plt = sns.scatterplot(x=range(len(ndat)) ,y=ndat['speed'], edgecolor="black")
plt.axvline(289, ls='--', color='white')
plt.axvline(407, ls='--', color='white')
plt.set_ylim([0, .5])

