from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

DATASET_PATH = os.path.join(os.path.dirname(__file__), '../dataset/customers.csv')

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df.dropna(inplace=True)
    return df

def run_kmeans(df, features, k):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    return labels, kmeans, scaler, X_scaled

def cluster_summary(df, labels, k):
    df_copy = df.copy()
    df_copy['Cluster'] = labels
    
    cluster_labels = {
        0: "💰 High Income – High Spenders",
        1: "💸 Low Income – Low Spenders",
        2: "🎯 Mid Income – Mid Spenders",
        3: "🌟 High Income – Low Spenders",
        4: "🛍️ Low Income – High Spenders"
    }
    
    summary = []
    for i in range(k):
        cluster_df = df_copy[df_copy['Cluster'] == i]
        label = cluster_labels.get(i, f"Cluster {i+1}")
        summary.append({
            "cluster_id": i,
            "label": label,
            "count": int(len(cluster_df)),
            "avg_age": round(float(cluster_df['Age'].mean()), 1),
            "avg_income": round(float(cluster_df['Annual_Income'].mean()), 1),
            "avg_spending": round(float(cluster_df['Spending_Score'].mean()), 1),
            "avg_frequency": round(float(cluster_df['Purchase_Frequency'].mean()), 1),
            "gender_ratio": round(float((cluster_df['Gender'] == 0).sum() / len(cluster_df) * 100), 1)
        })
    return summary

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        df = load_data()
        df_display = pd.read_csv(DATASET_PATH)
        preview = df_display.head(10).to_dict(orient='records')
        stats = {
            "total_customers": int(len(df)),
            "avg_age": round(float(df['Age'].mean()), 1),
            "avg_income": round(float(df['Annual_Income'].mean()), 1),
            "avg_spending": round(float(df['Spending_Score'].mean()), 1),
            "columns": list(pd.read_csv(DATASET_PATH).columns)
        }
        return jsonify({"success": True, "preview": preview, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/elbow', methods=['GET'])
def get_elbow():
    try:
        df = load_data()
        features = ['Annual_Income', 'Spending_Score']
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        wcss = []
        k_range = list(range(1, 11))
        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
            km.fit(X_scaled)
            wcss.append(round(float(km.inertia_), 2))
        
        return jsonify({"success": True, "k_values": k_range, "wcss": wcss})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/cluster', methods=['POST'])
def cluster():
    try:
        body = request.get_json()
        k = int(body.get('k', 5))
        features = body.get('features', ['Annual_Income', 'Spending_Score'])
        
        if k < 2 or k > 10:
            return jsonify({"success": False, "error": "K must be between 2 and 10"}), 400
        
        df = load_data()
        labels, kmeans, scaler, X_scaled = run_kmeans(df, features, k)
        
        df_out = df.copy()
        df_out['Cluster'] = labels
        
        # Scatter data for Income vs Spending
        scatter_data = []
        for idx, row in df_out.iterrows():
            scatter_data.append({
                "x": float(row['Annual_Income']),
                "y": float(row['Spending_Score']),
                "cluster": int(row['Cluster']),
                "age": int(row['Age']),
                "frequency": int(row['Purchase_Frequency'])
            })
        
        # Centroids (inverse transform back to original scale)
        centroids_scaled = kmeans.cluster_centers_
        centroids_orig = scaler.inverse_transform(centroids_scaled)
        centroids = []
        for c in centroids_orig:
            centroids.append({
                "x": round(float(c[features.index('Annual_Income')]) if 'Annual_Income' in features else 0, 1),
                "y": round(float(c[features.index('Spending_Score')]) if 'Spending_Score' in features else 0, 1)
            })
        
        summary = cluster_summary(df, labels, k)
        
        return jsonify({
            "success": True,
            "k": k,
            "features": features,
            "scatter_data": scatter_data,
            "centroids": centroids,
            "summary": summary,
            "inertia": round(float(kmeans.inertia_), 2)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        age = float(body.get('age', 30))
        gender = 1 if body.get('gender', 'Male') == 'Male' else 0
        income = float(body.get('annual_income', 50000))
        spending = float(body.get('spending_score', 50))
        frequency = float(body.get('purchase_frequency', 10))
        k = int(body.get('k', 5))
        
        df = load_data()
        features = ['Annual_Income', 'Spending_Score']
        labels, kmeans, scaler, _ = run_kmeans(df, features, k)
        
        new_customer = np.array([[income, spending]])
        new_scaled = scaler.transform(new_customer)
        cluster_id = int(kmeans.predict(new_scaled)[0])
        
        cluster_labels = {
            0: "💰 High Income – High Spenders",
            1: "💸 Low Income – Low Spenders",
            2: "🎯 Mid Income – Mid Spenders",
            3: "🌟 High Income – Low Spenders",
            4: "🛍️ Low Income – High Spenders"
        }
        
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id + 1}")
        
        return jsonify({
            "success": True,
            "cluster_id": cluster_id,
            "cluster_label": label,
            "input": {
                "age": age, "gender": "Male" if gender == 1 else "Female",
                "annual_income": income, "spending_score": spending,
                "purchase_frequency": frequency
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "Customer Segmentation API running", "version": "1.0"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
