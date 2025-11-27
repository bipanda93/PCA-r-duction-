#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:22:18 2025

@author: macbook
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style("white")

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 

# Importation du dataset Iris via scikit-learn
from sklearn import datasets 
iris = datasets.load_iris(as_frame=True)
df = iris.frame

# Manipulation pandas sur le DataFrame df
iris2 = df.drop("target", axis=1)
print(iris2.head())

#réaliser un ACP ou PCA
pca = PCA(n_components=4)
pca.fit(scale(iris2))

#appercu des valeurs propre ou pourcentage de variance 
eig=pd.DataFrame(
    {
     "Dimension": ["Dim" + str(x + 1) for x in range(4)],
     "valeur propre": pca.explained_variance_,
     "% variance expliquée": np.round(pca.explained_variance_ratio_ * 100),
     "%cum.var.expliquée": np.round(np.cumsum(pca.explained_variance_ratio_)*100),
     },
    columns = ["Dimension", "valeur propre", "% variance expliquée", "%cum..var.expliquée"]
    )
eig

#visualisation en deux dimension 
iris_pca = pca.transform(scale(iris2))
iris_pca_df = pd.DataFrame({
    "Dim1":iris_pca[:,0],
    "Dim2":iris_pca[:,1],
    "Species": df["target"].map(dict(enumerate(iris.target_names)))
    })
iris_pca_df.head()

coordvar = pca.components_.T * np.sqrt(pca.explained_variance_)
coordvar_df = pd.DataFrame(
    coordvar,
    columns=['PC' + str(i) for i in range(1, 5)],
    index=iris2.columns   # Ceci marche toujours
)
coordvar_df

  
#visualisation des données pour une meilleurs comprehension 
fig, axes = plt.subplots(figsize=(5,5))
fig.suptitle("Cercle des corrélations")
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
axes.axvline(x=0, color='lightblue', linestyle='--', linewidth=1)
axes.axhline(y=0, color='lightblue', linestyle='--', linewidth=1)

for j in range(4):
    axes.text(coordvar_df["PC1"][j], coordvar_df["PC2"][j], coordvar_df.index[j], size=12)
    axes.plot([0, coordvar_df["PC1"][j]], [0, coordvar_df["PC2"][j]], color="blue", linestyle='dashed')

# Ajout du cercle unité
axes.add_artist(plt.Circle((0,0), 1, color='red', fill=False))

plt.show()

g_pca = sn.lmplot(x="Dim1", y="Dim2", hue="Species", data=iris_pca_df, fit_reg=False, height=4, aspect=3)
g_pca.set(xlabel="Dimension 1 (73%)", ylabel="Dimension 2 (23%)")
g_pca.fig.suptitle("Premier plan Factoriel")
plt.show()





















  
    
    
    
    
    
    