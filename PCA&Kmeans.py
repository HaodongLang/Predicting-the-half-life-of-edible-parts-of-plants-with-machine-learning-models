import pandas as pd
import numpy as np
from matplotlib import rcParams
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# Read the data
data = pd.read_csv('./data/pesticide_data.csv')

# Get unique compounds based on SMILES
unique_compounds = data.drop_duplicates(subset=['SMILES'])

# Function to generate ECFP4 fingerprints
def get_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

# Generate fingerprints
fingerprints = []
valid_indices = []
for i, smiles in enumerate(unique_compounds['SMILES']):
    fp = get_ecfp4(smiles)
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(i)

# Convert to numpy array
valid_compounds = unique_compounds.iloc[valid_indices]
X = np.zeros((len(fingerprints), len(fingerprints[0])))
for i, fp in enumerate(fingerprints):
    DataStructs.ConvertToNumpyArray(fp, X[i])

# Perform PCA for 2D and 3D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster info to the dataframe
valid_compounds['Cluster'] = clusters

# Plot 2D clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1',fontsize=25)
plt.ylabel('Principal Component 2',fontsize=25)
plt.tick_params(axis='both', labelsize=0, width=3)
colorbar = plt.colorbar(scatter)
colorbar.ax.tick_params(labelsize=20)
colorbar.ax.set_title('cluster', fontsize=15)
plt.savefig('pesticide_clusters_2d.png')
plt.show()

# Plot 3D clusters
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=clusters, cmap='viridis', alpha=0.7)
ax.set_xlabel('Principal Component 1',fontsize=20)
ax.set_ylabel('Principal Component 2',fontsize=20)
ax.set_zlabel('Principal Component 3',fontsize=20)
plt.tick_params(axis='both', labelsize=0, width=3)
colorbar = plt.colorbar(scatter)
colorbar.ax.tick_params(labelsize=25)
colorbar.ax.set_title('cluster', fontsize=20)


plt.savefig('pesticide_clusters_3d.png')
plt.show()

# Group compounds by cluster
cluster_groups = defaultdict(list)
for idx, row in valid_compounds.iterrows():
    cluster_groups[row['Cluster']].append(row['substance name'])

# Print compounds in each cluster
for cluster_num in sorted(cluster_groups.keys()):
    print(f"\nCluster {cluster_num} compounds:")
    print(", ".join(sorted(set(cluster_groups[cluster_num]))))