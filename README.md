# 🌸 PCA (Principal Component Analysis) - Iris Dataset

## 🎯 Contexte et Objectif

### Problématique
L'Analyse en Composantes Principales (ACP/PCA) est une technique de réduction de dimensionnalité essentielle en Data Science. Face à des datasets avec de nombreuses variables corrélées, la PCA permet de :
- Réduire la complexité des données (de 4 dimensions à 2)
- Visualiser des données multidimensionnelles
- Éliminer la redondance entre variables
- Conserver un maximum d'information avec un minimum de dimensions

### Objectifs du projet
- Appliquer la PCA sur le célèbre dataset Iris (4 variables → 2 dimensions)
- Interpréter les valeurs propres et la variance expliquée
- Visualiser le cercle des corrélations (contribution des variables)
- Projeter les observations sur le plan factoriel
- Identifier la séparabilité des espèces d'iris

## 🏗️ Architecture technique

### Stack technique
- **Langage** : Python 3.x
- **Data Science** : NumPy, Pandas
- **Machine Learning** : scikit-learn (PCA, datasets, preprocessing)
- **Visualisation** : Matplotlib, Seaborn

### Pipeline d'analyse

```
1. IMPORTATION DES DONNÉES
   └─ Dataset Iris (scikit-learn)

2. PRÉPARATION
   ├─ Suppression de la variable cible (target)
   └─ Standardisation (scaling)

3. APPLICATION DE LA PCA
   ├─ Fit sur données standardisées
   └─ Extraction de 4 composantes principales

4. ANALYSE DES RÉSULTATS
   ├─ Valeurs propres (eigenvalues)
   ├─ Variance expliquée par composante
   └─ Variance cumulée

5. VISUALISATIONS
   ├─ Cercle des corrélations (contribution des variables)
   └─ Plan factoriel (projection des observations)
```

## 📊 Dataset Iris

### Description
Le dataset Iris (Fisher, 1936) est un classique du Machine Learning :
- **150 observations** : 50 fleurs de chaque espèce
- **3 espèces** : Setosa, Versicolor, Virginica
- **4 variables numériques** (en cm) :

| Variable | Description | Type |
|----------|-------------|------|
| `sepal length` | Longueur du sépale | Float |
| `sepal width` | Largeur du sépale | Float |
| `petal length` | Longueur du pétale | Float |
| `petal width` | Largeur du pétale | Float |
| `target` | Espèce (0/1/2) | Integer |

### Préparation des données

```python
from sklearn import datasets

# Importation
iris = datasets.load_iris(as_frame=True)
df = iris.frame

# Suppression de la variable cible
iris2 = df.drop("target", axis=1)
```

**Justification** : La PCA est une méthode **non supervisée** → on retire la variable cible pour analyser uniquement la structure des features.

## 🔍 Étapes de l'analyse PCA

### 1. Standardisation des données

```python
from sklearn.preprocessing import scale

# PCA nécessite des données standardisées (moyenne=0, écart-type=1)
iris_scaled = scale(iris2)
```

**Pourquoi standardiser ?**
- Les variables ont des unités/échelles différentes
- Sans standardisation, les variables à grande variance domineraient l'ACP
- La PCA est sensible aux échelles

---

### 2. Application de la PCA

```python
from sklearn.decomposition import PCA

# PCA avec 4 composantes (maximum possible avec 4 variables)
pca = PCA(n_components=4)
pca.fit(scale(iris2))
```

**Paramètres** :
- `n_components=4` : Nombre de composantes principales à extraire (ici, toutes)
- Alternative : `n_components=0.95` → garde assez de composantes pour expliquer 95% de la variance

---

### 3. Analyse des valeurs propres

```python
eig = pd.DataFrame({
    "Dimension": ["Dim" + str(x + 1) for x in range(4)],
    "valeur propre": pca.explained_variance_,
    "% variance expliquée": np.round(pca.explained_variance_ratio_ * 100),
    "%cum.var.expliquée": np.round(np.cumsum(pca.explained_variance_ratio_)*100),
})
```

**Résultats typiques** :

| Dimension | Valeur propre | % variance | % cum. variance |
|-----------|---------------|------------|-----------------|
| Dim1 | 2.91 | 73% | 73% |
| Dim2 | 0.91 | 23% | 96% |
| Dim3 | 0.15 | 4% | 99% |
| Dim4 | 0.02 | 1% | 100% |

**Interprétation** :
- **Dim1 + Dim2** capturent **96% de la variance** → 2 dimensions suffisent !
- **Réduction de dimensionnalité** : 4D → 2D avec seulement 4% de perte d'information
- Les dimensions 3 et 4 sont négligeables (bruit)

**Règle de Kaiser** : Conserver les dimensions avec valeur propre > 1 → ici, Dim1 et Dim2

---

### 4. Projection des observations

```python
# Transformation des données originales dans le nouvel espace
iris_pca = pca.transform(scale(iris2))

# Création d'un DataFrame pour visualisation
iris_pca_df = pd.DataFrame({
    "Dim1": iris_pca[:,0],
    "Dim2": iris_pca[:,1],
    "Species": df["target"].map(dict(enumerate(iris.target_names)))
})
```

**Résultat** : Chaque observation (fleur) est maintenant représentée par 2 coordonnées (Dim1, Dim2) au lieu de 4.

---

### 5. Coordonnées des variables (Cercle des corrélations)

```python
# Calcul des coordonnées des variables sur les composantes
coordvar = pca.components_.T * np.sqrt(pca.explained_variance_)

coordvar_df = pd.DataFrame(
    coordvar,
    columns=['PC' + str(i) for i in range(1, 5)],
    index=iris2.columns
)
```

**Signification** :
- Ces coordonnées indiquent la **contribution** de chaque variable à chaque composante
- Plus la coordonnée est proche de ±1, plus la variable est importante pour cette dimension
- Permet d'interpréter le sens des axes factoriels

---

## 📈 Visualisations et Interprétations

### 1. Cercle des corrélations

```python
fig, axes = plt.subplots(figsize=(5,5))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

# Axes de référence
axes.axvline(x=0, color='lightblue', linestyle='--')
axes.axhline(y=0, color='lightblue', linestyle='--')

# Projection des variables
for j in range(4):
    axes.text(coordvar_df["PC1"][j], coordvar_df["PC2"][j], 
              coordvar_df.index[j], size=12)
    axes.plot([0, coordvar_df["PC1"][j]], 
              [0, coordvar_df["PC2"][j]], 
              color="blue", linestyle='dashed')

# Cercle unité
axes.add_artist(plt.Circle((0,0), 1, color='red', fill=False))
plt.title("Cercle des corrélations")
plt.show()
```

**Interprétation** :

📊 **Règles de lecture** :
- **Longueur de la flèche** : Plus elle est longue (proche du cercle), mieux la variable est représentée sur ce plan
- **Angle entre flèches** :
  - Proche de 0° → Variables fortement corrélées positivement
  - Proche de 90° → Variables indépendantes
  - Proche de 180° → Variables corrélées négativement
- **Position sur l'axe** : Indique quelle dimension capture quelle variable

🌸 **Résultats typiques Iris** :
- `petal length` et `petal width` : Très corrélées (flèches parallèles)
- Ces deux variables contribuent fortement à **Dim1** (axe horizontal)
- `sepal width` : Contribue davantage à **Dim2** (axe vertical)
- **Dim1** = "Dimension de la taille globale de la fleur"
- **Dim2** = "Dimension de la forme (ratio longueur/largeur)"

---

### 2. Plan factoriel (Projection des observations)

```python
g_pca = sn.lmplot(x="Dim1", y="Dim2", hue="Species", 
                  data=iris_pca_df, fit_reg=False, 
                  height=4, aspect=3)
g_pca.set(xlabel="Dimension 1 (73%)", ylabel="Dimension 2 (23%)")
g_pca.fig.suptitle("Premier plan Factoriel")
plt.show()
```

**Interprétation** :

🎨 **Séparabilité des espèces** :
- **Setosa** : Cluster bien distinct, complètement séparé (en bas à gauche généralement)
- **Versicolor** et **Virginica** : Partiellement superposées (centre-droit)
- **Dim1** (73%) : Principale source de séparation
- **Dim2** (23%) : Affine la séparation entre Versicolor et Virginica

💡 **Insights** :
- La PCA révèle une structure naturelle dans les données
- 2 dimensions suffisent pour visualiser 96% de l'information
- Facilite les algorithmes de classification (K-means, SVM, etc.)

---

## 📚 Compétences démontrées

### Pour les recruteurs Data Scientist / ML Engineer

**1. Réduction de dimensionnalité**
- Compréhension théorique de la PCA
- Application pratique avec scikit-learn
- Interprétation des valeurs propres et variance expliquée

**2. Preprocessing**
- Standardisation des données (`scale`)
- Préparation pour algorithmes ML non supervisés

**3. Analyse statistique**
- Interprétation du cercle des corrélations
- Analyse de la structure des données multidimensionnelles
- Identification de la redondance entre variables

**4. Visualisation avancée**
- Cercle des corrélations (matplotlib personnalisé)
- Plan factoriel avec seaborn
- Communication visuelle de résultats complexes

**5. Applications métier**
- **Feature engineering** : Réduction de features avant modélisation
- **Data exploration** : Comprendre les relations entre variables
- **Visualisation** : Représenter des données haute dimension
- **Compression** : Réduire la complexité sans perdre d'information

## 🔧 Applications concrètes de la PCA

### 1. Machine Learning
```python
# Réduire les features avant classification
pca = PCA(n_components=0.95)  # Garde 95% de variance
X_reduced = pca.fit_transform(X_train)

# Entraîner un modèle sur données réduites (plus rapide)
model.fit(X_reduced, y_train)
```

**Avantages** :
- Réduction du temps d'entraînement
- Moins de risque d'overfitting
- Gestion de la multicollinéarité

---

### 2. Compression d'images

```python
# Image = matrice de pixels → appliquer PCA
pca = PCA(n_components=50)  # Garde 50 composantes sur 1000 pixels
image_compressed = pca.fit_transform(image)

# Reconstruction avec perte minimale
image_reconstructed = pca.inverse_transform(image_compressed)
```

**Résultat** : Compression de 95% avec qualité visuelle préservée

---

### 3. Détection d'anomalies

```python
# Projeter sur 2 composantes
X_pca = pca.fit_transform(X)

# Les points éloignés du centre = anomalies
distances = np.linalg.norm(X_pca, axis=1)
anomalies = X[distances > threshold]
```

---

### 4. Analyse de sentiment (NLP)

```python
# 10,000 mots → 100 dimensions PCA
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(texts)

pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_tfidf.toarray())
```

**Utilité** : Visualiser les clusters de documents similaires

---

## 📋 Concepts théoriques clés

### Mathématiques de la PCA

**1. Objectif** : Trouver les axes orthogonaux qui maximisent la variance des données projetées

**2. Étapes mathématiques** :
```
1. Centrer les données (moyenne = 0)
2. Calculer la matrice de covariance
3. Calculer les vecteurs propres (directions) et valeurs propres (importance)
4. Trier par valeurs propres décroissantes
5. Projeter les données sur les k premiers vecteurs propres
```

**3. Propriétés** :
- Les composantes principales sont **orthogonales** (non corrélées)
- La première composante capture le **maximum de variance**
- La PCA est une transformation **linéaire** (limitation)

---

### Quand utiliser la PCA ?

✅ **Cas d'usage appropriés** :
- Variables numériques continues
- Variables corrélées entre elles
- Besoin de visualisation (haute dimension → 2D/3D)
- Réduction de features avant modélisation
- Données bruitées (la PCA filtre le bruit)

❌ **Limitations** :
- **Perte d'interprétabilité** : Les composantes sont des combinaisons linéaires difficiles à nommer
- **Linéarité** : Ne capture pas les relations non-linéaires (alternative : t-SNE, UMAP)
- **Sensible aux outliers** : Les valeurs extrêmes influencent les axes
- **Pas adapté aux variables catégorielles** : Nécessite des données numériques

---

## 🔧 Reproduction du projet

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Exécution

```python
# Lancer le script
python pca_iris_analysis.py
```

### Structure du projet

```
pca-iris-analysis/
├── pca_iris_analysis.py        # Script principal
├── outputs/
│   ├── correlation_circle.png  # Cercle des corrélations
│   └── factorial_plan.png      # Plan factoriel
├── requirements.txt
└── README.md
```

---

## 📖 Contexte

**Réalisé dans le cadre** : Formation personnelle  
**Objectif** : Approfondir les compétences en Machine Learning non supervisé  
**Durée** : Auto-formation  
**Focus** : Maîtriser la réduction de dimensionnalité et l'interprétation de la PCA

---

📧 Contact

Franck Ulrich BIPANDA 

📧 bipanda.franck@icloud.com  
🔗 [LinkedIn](https://linkedin.com/in/franck-bipanda-13392372)  
🌐 [Portfolio](https://datascienceportfol.io/bipandaf)
