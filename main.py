from typing import Any, Union
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.arrays import ExtensionArray
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn import metrics
name_of_file = "c:\Alcohol Consumption in Russia.csv"
raw_data = pd.read_csv(name_of_file)
print(raw_data.head())
print(raw_data.dtypes)
regions = raw_data.region.unique()
complete_observation_regions = []
incomplete_observation_regions = []
# let's remove regions with incomplete data from dataset
for i in regions:
    flag = raw_data[raw_data['region'] == i]
    #    print(flag)
    if flag['wine'].isna().sum(axis=0) == 0 or flag['beer'].isna().sum(axis=0) == 0 or flag['vodka'].isna().sum(
            axis=0) == 0 or flag['champagne'].isna().sum(axis=0) == 0 or flag['brandy'].isna().sum(axis=0) == 0:
        complete_observation_regions.append(i)
    else:
        incomplete_observation_regions.append(i)
complete_observation_regions = sorted(complete_observation_regions)
incomplete_observation_regions = sorted(incomplete_observation_regions)
print('Regions with complete data are:  ', complete_observation_regions)
print('Total number of regions with complete data is:', len(complete_observation_regions))
print('Regions with incomplete data are:  ', incomplete_observation_regions)
print('Total number of regions with incomplete data is:', len(incomplete_observation_regions))
# Let's sort values of data by region name
raw_data_sorted = raw_data.sort_values('region')
print(raw_data_sorted.shape)
# Let's remove data for regions with incomplete data
raw_data_sorted_complete_data: Union[Union[Series, ExtensionArray, ndarray, DataFrame, None], Any] = raw_data_sorted[
    raw_data_sorted['region'].isin(complete_observation_regions)]
# print(raw_data_sorted_complete_data.shape)
# Drop old index...
raw_data_sorted_complete_data.reset_index(drop=True)
# Subset multi index
raw_data_sorted_complete_data.set_index(['region', 'year'], inplace=True)
# print(raw_data_sorted_complete_data)
# Sort year
raw_data_sorted_complete_data = raw_data_sorted_complete_data.sort_index(level=['region', 'year'])
print('names of columns in dataset')
print(raw_data_sorted_complete_data.columns)
print('')
print('Dataset with complete observations:')
print(raw_data_sorted_complete_data)
print('')
# Now let's take a look at our full data with lineplot
fig, ax = plt.subplots()
sns.lineplot(x='year', y='wine', data = raw_data_sorted_complete_data, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = raw_data_sorted_complete_data, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = raw_data_sorted_complete_data, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = raw_data_sorted_complete_data, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = raw_data_sorted_complete_data, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in all regions")

# Show the legend and plot the data
ax.legend()
plt.show()
fig.savefig('Alcohol consumption in all regions.png')
plt.clf()
print('descriptive statistics for all regions with complete data')
print('')
print(raw_data_sorted_complete_data.describe())
print('')
# Now let's take a look at some particular regions to see if there is a difference in the dataset
fig, ax = plt.subplots()
region_fo_displaying = raw_data_sorted_complete_data.loc[pd.IndexSlice['Republic of Dagestan',:], :]
sns.lineplot(x='year', y='wine', data = region_fo_displaying, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = region_fo_displaying, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = region_fo_displaying, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = region_fo_displaying, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = region_fo_displaying, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in Republic of Dagestan region")

# Show the legend and plot the data
ax.legend()
plt.show()
plt.clf()
fig.savefig('Alcohol consumption in Republic of Dagestan region.png')

# Now let's take a look at some particular regions to see if there is a difference in the dataset
fig, ax = plt.subplots()
region_fo_displaying = raw_data_sorted_complete_data.loc[pd.IndexSlice['Moscow',:], :]
sns.lineplot(x='year', y='wine', data = region_fo_displaying, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = region_fo_displaying, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = region_fo_displaying, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = region_fo_displaying, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = region_fo_displaying, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in Moscow region")

# Show the legend and plot the data
ax.legend()
plt.show()
plt.clf()
fig.savefig('Alcohol consumption in Moscow region.png')

# Now let's take a look at some particular regions to see if there is a difference in the dataset
fig, ax = plt.subplots()
region_fo_displaying = raw_data_sorted_complete_data.loc[pd.IndexSlice['Saint Petersburg',:], :]
sns.lineplot(x='year', y='wine', data = region_fo_displaying, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = region_fo_displaying, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = region_fo_displaying, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = region_fo_displaying, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = region_fo_displaying, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in Saint Petersburg region")

# Show the legend and plot the data
ax.legend()
plt.show()
plt.clf()
fig.savefig('Alcohol consumption in Saint Petersburg region.png')

# Now let's take a look at some particular regions to see if there is a difference in the dataset
fig, ax = plt.subplots()
region_fo_displaying = raw_data_sorted_complete_data.loc[pd.IndexSlice['Astrakhan Oblast',:], :]
sns.lineplot(x='year', y='wine', data = region_fo_displaying, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = region_fo_displaying, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = region_fo_displaying, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = region_fo_displaying, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = region_fo_displaying, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in Astrakhan Oblast region")

# Show the legend and plot the data
ax.legend()
plt.show()
plt.clf()
fig.savefig('Alcohol consumption in Astrakhan Oblast region.png')

# let's take a look at four different region consumption on one plot

fig, axes = plt.subplots(2, 2,  figsize=(20,12))
fig.suptitle('The difference of alcohol consumption in regions, litres per capita')
region_fo_displaying1 = raw_data_sorted_complete_data.loc[pd.IndexSlice[['Republic of Dagestan', 'Moscow', 'Saint Petersburg','Astrakhan Oblast'],:], :]

sns.lineplot(ax= axes[0,0], x = 'year', y = 'wine', data = region_fo_displaying1, hue = 'region')
axes[0,0].set_xlim((1998, 2016))

sns.lineplot(ax= axes[0,1], x = 'year', y = 'beer', data = region_fo_displaying1,  hue = 'region')
axes[0,1].set_xlim((1998, 2016))

sns.lineplot(ax= axes[1,0], x = 'year', y = 'vodka', data = region_fo_displaying1,  hue = 'region')
axes[1,0].set_xlim((1998, 2016))

sns.lineplot(ax= axes[1,1], x = 'year', y = 'champagne', data = region_fo_displaying1,  hue = 'region')
axes[1,1].set_xlim((1998, 2016))

plt.show()
plt.clf()
fig.savefig('Alcohol consumption in 4 different regions.png')
plt.clf()

# let's make dataframe with median for each region and kind of alcohol
Alcohol_consumption_regions = raw_data_sorted_complete_data.groupby(['region']).median()
print('Alcohol_consumption_regions - Median for alcohol consumption in regions')
print('')
print(Alcohol_consumption_regions)
print('')
#print(consumption_regions_1998)
consumption_regions_1998 = raw_data_sorted_complete_data.loc[pd.IndexSlice[:, [1998]], :]
consumption_regions_1998.reset_index(drop=True)
print(consumption_regions_1998)
#print(consumption_regions_2016)
consumption_regions_2016 = raw_data_sorted_complete_data.loc[pd.IndexSlice[:, [2016]], :]
consumption_regions_2016.reset_index(drop=True)
print(consumption_regions_2016)
# Let's add more information for dendrogram - fisrt and last year of observation
Alcohol_consumption_regions_1998_2016_median = Alcohol_consumption_regions
Alcohol_consumption_regions_1998_2016_median.insert(0, "wine_1998", consumption_regions_1998['wine'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(1, "wine_2016", consumption_regions_2016['wine'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(3, "beer_1998", consumption_regions_1998['beer'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(4, "beer_2016", consumption_regions_2016['beer'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(6, "vodka_1998", consumption_regions_1998['vodka'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(7, "vodka_2016", consumption_regions_2016['vodka'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(9, "champagne_1998", consumption_regions_1998['champagne'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(10, "champagne_2016", consumption_regions_2016['champagne'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(12, "brandy_1998", consumption_regions_1998['brandy'].values, True)
Alcohol_consumption_regions_1998_2016_median.insert(13, "brandy_2016", consumption_regions_2016['brandy'].values, True)
print('Alcohol consumpion in region in 1998, 2016 years with median ')
print(Alcohol_consumption_regions_1998_2016_median.columns.values)
print(Alcohol_consumption_regions_1998_2016_median)


# Let's make linkage matrix for the regions with median consumption
Z_median = linkage(Alcohol_consumption_regions, method='median')
Z_1998_2016_median = linkage(Alcohol_consumption_regions_1998_2016_median, method='average') #weighted complete single average centroid ward

# Let's calculate silhouette score metric to choose the optimal method of clustering
Number_of_clusters = [2, 3, 4, 5, 6]
dataset_without_index=Alcohol_consumption_regions_1998_2016_median.reset_index(drop=True)
Method_of_clustering = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
for iiii in  Number_of_clusters:
    for iiiii in Method_of_clustering:
        ZI = linkage(Alcohol_consumption_regions_1998_2016_median, method=iiiii)
        list_of_clusters = fcluster(ZI, iiii, criterion='maxclust')
        print('')
        print("Number of clusters :", iiii, 'Method :', iiiii,  'Silhouette :  ', metrics.silhouette_score(dataset_without_index, list_of_clusters))
        print('')

# Plot the dendrogram, using varieties as labels
dendrogram(Z_1998_2016_median,
           labels = complete_observation_regions,
           p=81,
           orientation='right',
           truncate_mode='level',
           leaf_rotation=0,
           leaf_font_size=6,
           get_leaves=True
           )
plt.title('Dendrogram on alcohol consumption')
plt.show()
plt.clf()
fig.savefig('Dendrogram of regions 1.png')

numclust = 4

fl = fcluster(Z_1998_2016_median,numclust,criterion='maxclust')
number_of_entries = len(fl)
print(len(fl))
print(fl)
print('')
dataset_without_index=Alcohol_consumption_regions_1998_2016_median.reset_index(drop=True)
print("Dataset without index")
print(dataset_without_index.head())
print('')
print('Silhouette :  ',  metrics.silhouette_score(Alcohol_consumption_regions_1998_2016_median, fl))

d = Counter(fl)
print(d)
dim = max(d.values())
print(dim)
fl = fl.tolist()
for i in range(numclust):
    c= fl.count(i+1)
    print('Cluster # ',i+1, ' contains :', c, ' entries')

# Now it's time to print and store names of regions belonging to each cluster:
regions_for_clusters = Alcohol_consumption_regions.index
array_names_of_region_per_cluster = [[0 for j in range(dim)] for i in range(numclust)]

print('')
for i in range(numclust):
        print('list of regions in cluster' , i+1, '  :')
        iii = 0

        for ii in range(number_of_entries):
            if fl[ii] == i+1 :
                print(regions_for_clusters[ii])
                array_names_of_region_per_cluster[i][iii] = regions_for_clusters[ii]
                iii += 1
        print('')
for row in array_names_of_region_per_cluster:
    print(*row)
# Now let's print data for each cluster
first_cluster = array_names_of_region_per_cluster[0][:]
first_cluster = list(filter(lambda num: num != 0, first_cluster))
print(first_cluster)
data_1st_cluster = raw_data_sorted_complete_data.loc[pd.IndexSlice[first_cluster, : ], :]
print(data_1st_cluster)

fig, ax = plt.subplots()
sns.lineplot(x='year', y='wine', data = data_1st_cluster, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_1st_cluster, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_1st_cluster, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_1st_cluster, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_1st_cluster, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of first cluster")

# Show the legend and plot the data
ax.legend()
plt.show()
fig.savefig('Alcohol consumption in regions of first cluster.png')
plt.clf()
# let's view descriptive statistics  for 1st claster regions
print('descriptive statistics  for 1st claster regions')
print('')
print(data_1st_cluster.describe())
print('')
print('descriptive statistics  for 1st claster regions 2011-2016')
print('')
data_1st_cluster20112016 = data_1st_cluster.loc[pd.IndexSlice[:, [2011, 2012, 2013, 2014, 2015, 2016]], :]
print(data_1st_cluster20112016.describe())
print('')
second_cluster = array_names_of_region_per_cluster[1][:]
second_cluster = list(filter(lambda num: num != 0, second_cluster))
print(second_cluster)
data_2nd_cluster = raw_data_sorted_complete_data.loc[pd.IndexSlice[second_cluster, : ], :]
print(data_2nd_cluster)

fig, ax = plt.subplots()
sns.lineplot(x='year', y='wine', data = data_2nd_cluster, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_2nd_cluster, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_2nd_cluster, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_2nd_cluster, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_2nd_cluster, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of second cluster")
# Show the legend and plot the data
ax.legend()
plt.show()
fig.savefig('Alcohol consumption in regions of second cluster.png')
plt.clf()
# let's view descriptive statistics  for 2nd claster regions
print('descriptive statistics  for 2nd claster regions')
print('')
print(data_2nd_cluster.describe())
print('')
print('descriptive statistics  for 2nd claster regions 2011-2016')
print('')
data_2nd_cluster20112016 = data_2nd_cluster.loc[pd.IndexSlice[:, [2011, 2012, 2013, 2014, 2015, 2016]], :]
print(data_2nd_cluster20112016.describe())
print('')
third_cluster = array_names_of_region_per_cluster[2][:]
third_cluster = list(filter(lambda num: num != 0, third_cluster))
print(third_cluster)
data_3d_cluster = raw_data_sorted_complete_data.loc[pd.IndexSlice[third_cluster, : ], :]
print(data_3d_cluster)

fig, ax = plt.subplots()
sns.lineplot(x='year', y='wine', data = data_3d_cluster, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_3d_cluster, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_3d_cluster, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_3d_cluster, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_3d_cluster, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of third cluster")
# Show the legend and plot the data
ax.legend()
plt.show()
fig.savefig('Alcohol consumption in regions of third cluster.png')
plt.clf()
# let's view descriptive statistics  for 3d claster regions
print('descriptive statistics for 3d claster regions')
print('')
print(data_3d_cluster.describe())
print('')
print('descriptive statistics  for 3d claster regions 2011-2016')
data_3d_cluster20112016 = data_3d_cluster.loc[pd.IndexSlice[:, [2011, 2012, 2013, 2014, 2015, 2016]], :]
print(data_3d_cluster20112016.describe())
print('')
fourth_cluster = array_names_of_region_per_cluster[3][:]
fourth_cluster = list(filter(lambda num: num != 0, fourth_cluster))
print(fourth_cluster)
data_4th_cluster = raw_data_sorted_complete_data.loc[pd.IndexSlice[fourth_cluster, : ], :]
print(data_4th_cluster)

fig, ax = plt.subplots()
sns.lineplot(x='year', y='wine', data = data_4th_cluster, ax=ax, label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_4th_cluster, ax=ax, label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_4th_cluster, ax=ax, label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_4th_cluster, ax=ax, label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_4th_cluster, ax=ax, label = 'champagne')

# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of fourth cluster")
# Show the legend and plot the data
ax.legend()
plt.show()
fig.savefig('Alcohol consumption in regions of fourth cluster.png')
plt.clf()
# let's view descriptive statistics  for 4th claster regions
print('descriptive statistics  for 4th claster regions')
print('')
print(data_4th_cluster.describe())
print('')
print('descriptive statistics  for 4th claster regions 2011-2016')
data_4th_cluster20112016 = data_4th_cluster.loc[pd.IndexSlice[:, [2011, 2012, 2013, 2014, 2015, 2016]], :]
print(data_4th_cluster20112016.describe())
print('')

print('descriptive statistics  for all regions 2011-2016')
all_regions_20112016 = raw_data_sorted_complete_data.loc[pd.IndexSlice[:, [2011, 2012, 2013, 2014, 2015, 2016]], :]
print(all_regions_20112016.describe())
print('')

fig, axes = plt.subplots(2, 2,  figsize=(20,12))
fig.suptitle('The difference of alcohol consumption in 4 clusters of regions, litres per capita')

sns.lineplot(x='year', y='wine', data = data_1st_cluster, ax=axes[0,0], label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_1st_cluster, ax=axes[0,0], label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_1st_cluster, ax=axes[0,0], label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_1st_cluster, ax=axes[0,0], label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_1st_cluster, ax=axes[0,0], label = 'champagne')

# Customize the labels and limits
axes[0,0].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of first cluster")
axes[0,0].legend()
sns.lineplot(x='year', y='wine', data = data_2nd_cluster, ax=axes[0,1], label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_2nd_cluster, ax=axes[0,1], label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_2nd_cluster, ax=axes[0,1], label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_2nd_cluster, ax=axes[0,1], label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_2nd_cluster, ax=axes[0,1], label = 'champagne')

# Customize the labels and limits
axes[0,1].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of second cluster")
# Show the legend and plot the data
axes[0,1].legend()


sns.lineplot(x='year', y='wine', data = data_3d_cluster, ax=axes[1,0], label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_3d_cluster, ax=axes[1,0], label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_3d_cluster, ax=axes[1,0], label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_3d_cluster, ax=axes[1,0], label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_3d_cluster, ax=axes[1,0], label = 'champagne')

# Customize the labels and limits
axes[1,0].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of third cluster")
axes[1,0].legend()

sns.lineplot(x='year', y='wine', data = data_4th_cluster, ax=axes[1,1], label = 'wine')
sns.lineplot(x = 'year', y = 'brandy', data = data_4th_cluster, ax=axes[1,1], label = 'brandy')
sns.lineplot(x = 'year', y = 'beer', data = data_4th_cluster, ax=axes[1,1], label = 'beer')
sns.lineplot(x = 'year', y = 'vodka', data = data_4th_cluster, ax=axes[1,1], label = 'vodka')
sns.lineplot(x = 'year', y = 'champagne', data = data_4th_cluster, ax=axes[1,1], label = 'champagne')

# Customize the labels and limits
axes[1,1].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Alcohol consumption in regions of fourth cluster")
# Show the legend and plot the data
axes[1,1].legend()

plt.show()
fig.savefig('The difference of alcohol consumption in 4 clusters of regions, litres per capita.png')


fig, axes = plt.subplots(2, 2,  figsize=(20,12))
fig.suptitle('Comparative alcohol consumption by cluster, liters per capita ')

sns.lineplot(x='year', y='wine', data = data_1st_cluster, ax=axes[0,0], label = '1st cluster')
sns.lineplot(x = 'year', y = 'wine', data = data_2nd_cluster, ax=axes[0,0], label = '2nd cluster')
sns.lineplot(x = 'year', y = 'wine', data = data_3d_cluster, ax=axes[0,0], label = '3d cluster')
sns.lineplot(x = 'year', y = 'wine', data = data_4th_cluster, ax=axes[0,0], label = '4th cluster')


# Customize the labels and limits
axes[0,0].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Consumption of wine")
axes[0,0].legend()
sns.lineplot(x='year', y='beer', data = data_1st_cluster, ax=axes[0,1], label = '1st cluster')
sns.lineplot(x = 'year', y = 'beer', data = data_2nd_cluster, ax=axes[0,1], label = '2nd cluster')
sns.lineplot(x = 'year', y = 'beer', data = data_3d_cluster, ax=axes[0,1], label = '3d cluster')
sns.lineplot(x = 'year', y = 'beer', data = data_4th_cluster, ax=axes[0,1], label = '4th cluster')


# Customize the labels and limits
axes[0,1].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Consumption of beer")
# Show the legend and plot the data
axes[0,1].legend()


sns.lineplot(x='year', y='vodka', data = data_1st_cluster, ax=axes[1,0], label = '1st cluster')
sns.lineplot(x = 'year', y = 'vodka', data = data_2nd_cluster, ax=axes[1,0], label = '2nd cluster')
sns.lineplot(x = 'year', y = 'vodka', data = data_3d_cluster, ax=axes[1,0], label = '3d cluster')
sns.lineplot(x = 'year', y = 'vodka', data = data_4th_cluster, ax=axes[1,0], label = '4th cluster')


# Customize the labels and limits
axes[1,0].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Consumption of vodka")
axes[1,0].legend()

sns.lineplot(x='year', y='champagne', data = data_1st_cluster, ax=axes[1,1], label = '1st cluster')
sns.lineplot(x = 'year', y = 'champagne', data = data_2nd_cluster, ax=axes[1,1], label = '2nd cluster')
sns.lineplot(x = 'year', y = 'champagne', data = data_3d_cluster, ax=axes[1,1], label = '3d cluster')
sns.lineplot(x = 'year', y = 'champagne', data = data_4th_cluster, ax=axes[1,1], label = '4th cluster')

# Customize the labels and limits
axes[1,1].set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Consumption of champagne")
# Show the legend and plot the data
axes[1,1].legend()

plt.show()
fig.savefig('Comparative alcohol consumption by cluster, liters per capita .png')


fig, ax = plt.subplots()
sns.lineplot(x='year', y='vodka', data = data_1st_cluster, ax=ax, label = '1st cluster')
sns.lineplot(x = 'year', y = 'vodka', data = data_2nd_cluster, ax=ax, label = '2nd cluster')
sns.lineplot(x = 'year', y = 'vodka', data = data_3d_cluster, ax=ax, label = '3d cluster')
sns.lineplot(x = 'year', y = 'vodka', data = data_4th_cluster, ax=ax, label = '4th cluster')


# Customize the labels and limits
ax.set(xlabel="Year", ylabel='litres per capita', xlim=(1998,2016), title="Consumption of brandy")
plt.show()
fig.savefig('Comparative consumption of brandy by cluster, liters per capita .png')
