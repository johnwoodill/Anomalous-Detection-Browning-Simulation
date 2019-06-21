import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as ssd
from scipy.spatial import distance
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
import subprocess

# Build new simulation data
# subprocess.run("python 1-Data-step.py", shell=True)

# Load data set
dmat = np.load('/home/server/pi/homes/woodilla/Projects/Anomalous-Detection-Browning-Simulation/data/jsd_mat.npy')

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

