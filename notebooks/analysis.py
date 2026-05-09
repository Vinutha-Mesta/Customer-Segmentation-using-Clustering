# Customer Segmentation using K-Means Clustering
# Run this notebook cell by cell in Jupyter or Google Colab

# ============================================================
# CELL 1 - Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# ============================================================
# CELL 2 - Load Dataset
# ============================================================
df = pd.read_csv('../dataset/customers.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

# ============================================================
# CELL 3 - Basic Info
# ============================================================
print("Dataset Info:")
print(df.info())
print("\nStatistical Summary:")
df.describe()

# ============================================================
# CELL 4 - Check Missing Values
# ============================================================
print("Missing Values:")
print(df.isnull().sum())

# ============================================================
# CELL 5 - Data Preprocessing
# ============================================================
df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
print("Gender encoded: Male=1, Female=0")
df.head()

# ============================================================
# CELL 6 - Visualize Data Distribution
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Customer Data Distribution', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['Age'], bins=20, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')

axes[0, 1].hist(df['Annual_Income'], bins=20, color='coral', edgecolor='white')
axes[0, 1].set_title('Annual Income Distribution')
axes[0, 1].set_xlabel('Annual Income (₹)')

axes[0, 2].hist(df['Spending_Score'], bins=20, color='mediumseagreen', edgecolor='white')
axes[0, 2].set_title('Spending Score Distribution')
axes[0, 2].set_xlabel('Spending Score (1-100)')

axes[1, 0].bar(df['Gender'].value_counts().index, df['Gender'].value_counts().values, color=['steelblue', 'salmon'])
axes[1, 0].set_title('Gender Distribution')

axes[1, 1].scatter(df['Annual_Income'], df['Spending_Score'], alpha=0.5, color='purple')
axes[1, 1].set_title('Income vs Spending Score')
axes[1, 1].set_xlabel('Annual Income (₹)')
axes[1, 1].set_ylabel('Spending Score')

axes[1, 2].scatter(df['Age'], df['Spending_Score'], alpha=0.5, color='orange')
axes[1, 2].set_title('Age vs Spending Score')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('Spending Score')

plt.tight_layout()
plt.savefig('../dataset/data_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Distribution plots saved!")

# ============================================================
# CELL 7 - Elbow Method to Find Optimal K
# ============================================================
X = df[['Annual_Income', 'Spending_Score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='steelblue', linewidth=2, markersize=8)
plt.axvline(x=5, color='red', linestyle='--', label='Optimal K=5')
plt.title('Elbow Method - Finding Optimal K', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../dataset/elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ WCSS Values: {[round(w, 2) for w in wcss]}")

# ============================================================
# CELL 8 - Apply K-Means with K=5
# ============================================================
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ K-Means applied with K={optimal_k}")
print("\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())

# ============================================================
# CELL 9 - Visualize Clusters
# ============================================================
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
cluster_names = [
    'High Income–High Spenders',
    'Low Income–Low Spenders',
    'Mid Income–Mid Spenders',
    'High Income–Low Spenders',
    'Low Income–High Spenders'
]

plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual_Income'],
        cluster_data['Spending_Score'],
        c=colors[i], label=cluster_names[i],
        s=80, alpha=0.8, edgecolors='white', linewidths=0.5
    )

# Plot centroids
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    c='black', marker='X', s=200, zorder=5, label='Centroids'
)

plt.title('Customer Segmentation – Income vs Spending Score', fontsize=15, fontweight='bold')
plt.xlabel('Annual Income (₹)', fontsize=12)
plt.ylabel('Spending Score (1–100)', fontsize=12)
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('../dataset/cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Cluster visualization saved!")

# ============================================================
# CELL 10 - Cluster Analysis
# ============================================================
print("\n" + "="*60)
print("CLUSTER ANALYSIS REPORT")
print("="*60)
for i in range(optimal_k):
    c = df[df['Cluster'] == i]
    print(f"\n🔵 Cluster {i} — {cluster_names[i]}")
    print(f"   Count       : {len(c)} customers ({len(c)/len(df)*100:.1f}%)")
    print(f"   Avg Age     : {c['Age'].mean():.1f}")
    print(f"   Avg Income  : ₹{c['Annual_Income'].mean():,.0f}")
    print(f"   Avg Spending: {c['Spending_Score'].mean():.1f}/100")
    print(f"   Avg Freq    : {c['Purchase_Frequency'].mean():.1f} purchases")

# ============================================================
# CELL 11 - Heatmap of Cluster Characteristics
# ============================================================
cluster_profile = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency']].mean()

plt.figure(figsize=(10, 5))
sns.heatmap(cluster_profile.T, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=[f'C{i}' for i in range(optimal_k)],
            cbar_kws={'label': 'Value'})
plt.title('Cluster Feature Heatmap', fontsize=14, fontweight='bold')
plt.ylabel('Feature')
plt.xlabel('Cluster')
plt.tight_layout()
plt.savefig('../dataset/cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 12 - Predict New Customer
# ============================================================
def predict_cluster(annual_income, spending_score):
    new_data = np.array([[annual_income, spending_score]])
    new_scaled = scaler.transform(new_data)
    cluster_id = kmeans.predict(new_scaled)[0]
    print(f"\n📊 Prediction Result:")
    print(f"   Annual Income : ₹{annual_income:,}")
    print(f"   Spending Score: {spending_score}/100")
    print(f"   → Cluster {cluster_id}: {cluster_names[cluster_id]}")
    return cluster_id

# Test with a sample customer
predict_cluster(annual_income=85000, spending_score=80)
predict_cluster(annual_income=20000, spending_score=15)
predict_cluster(annual_income=60000, spending_score=50)

print("\n✅ Analysis Complete!")
