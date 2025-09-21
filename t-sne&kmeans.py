import pandas
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.manifold import TSNE
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

# Perform t-SNE for 2D and 3D
print("Running t-SNE for 2D...")
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne_2d.fit_transform(X)

print("Running t-SNE for 3D...")
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
X_3d = tsne_3d.fit_transform(X)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster info to the dataframe
valid_compounds['Cluster'] = clusters

# Plot 2D clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('t-SNE Dimension 1',fontsize=25)
plt.ylabel('t-SNE Dimension 2',fontsize=25)
plt.tick_params(axis='both', labelsize=0, width=3)
colorbar = plt.colorbar(scatter)
colorbar.ax.tick_params(labelsize=20)
colorbar.ax.set_title('cluster', fontsize=15)
plt.savefig('pesticide_clusters_tsne_2d.png')
plt.show()

# Plot 3D clusters
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=clusters, cmap='viridis', alpha=0.7)
ax.set_xlabel('t-SNE Dimension 1',fontsize=25)
ax.set_ylabel('t-SNE Dimension 2',fontsize=25)
ax.set_zlabel('t-SNE Dimension 3',fontsize=25)

plt.tick_params(axis='both', labelsize=0, width=3)
colorbar = plt.colorbar(scatter)
colorbar.ax.tick_params(labelsize=25)
colorbar.ax.set_title('cluster', fontsize=20)
plt.savefig('pesticide_clusters_tsne_3d.png')
plt.show()

# Group compounds by cluster
cluster_groups = defaultdict(list)
for idx, row in valid_compounds.iterrows():
    cluster_groups[row['Cluster']].append(row['substance name'])

# Print compounds in each cluster
print("\nCompounds grouped by clusters:")
for cluster_num in sorted(cluster_groups.keys()):
    print(f"\nCluster {cluster_num} compounds:")
    print(", ".join(sorted(set(cluster_groups[cluster_num]))))
