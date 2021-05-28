import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# import dataset of your own
dataset = pd.read_csv('wifilocal.csv')
df = pd.DataFrame(dataset)


y = dataset.loc[:, 'Room'].values
x = StandardScaler().fit_transform(dataset.iloc[:,:-1])


# logic of PCA explained variance calculator
n_comp = 0
expl_variance = 0

while expl_variance < 0.85:
	n_comp += 1

	pca = PCA(n_components = n_comp)
	pc = pca.fit_transform(x)

	expl_variance = pca.explained_variance_ratio_.sum()


print('Required', n_comp, 'principal component to capture', expl_variance, 'variance')
