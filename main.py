ml_service.py:

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/correlation', methods=['POST'])
def correlation_analysis():
    try:
        data = request.json
        dataset_url = data.get('dataset_url')
        method = data.get('method', 'pearson')
        
        # Load CSV
        df = pd.read_csv(dataset_url)
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return jsonify({
                "success": False,
                "error": "No numeric columns found"
            }), 400
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        # Calculate VIF scores
        from sklearn.preprocessing import StandardScaler
        vif_scores = {}
        if len(numeric_df.columns) > 1:
            for col in numeric_df.columns:
                try:
                    X = numeric_df.drop(columns=[col])
                    y = numeric_df[col]
                    
                    # Remove rows with NaN
                    mask = ~(X.isna().any(axis=1) | y.isna())
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) > 1:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                        model.fit(X_clean, y_clean)
                        r2 = model.score(X_clean, y_clean)
                        vif = 1 / (1 - r2) if r2 < 0.9999 else 999
                        vif_scores[col] = float(vif)
                except:
                    vif_scores[col] = 1.0
        
        return jsonify({
            "success": True,
            "correlation_matrix": corr_matrix.to_dict(),
            "feature_names": list(corr_matrix.columns),
            "high_correlations": high_corr,
            "multicollinearity_detected": len(high_corr) > 0,
            "vif_scores": vif_scores
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/clustering', methods=['POST'])
def clustering_analysis():
    try:
        data = request.json
        dataset_url = data.get('dataset_url')
        n_clusters = data.get('n_clusters', 3)
        
        # Load CSV
        df = pd.read_csv(dataset_url)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.empty or len(numeric_df) < n_clusters:
            return jsonify({
                "success": False,
                "error": "Not enough numeric data"
            }), 400
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        
        # Clustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        
        # Metrics
        sil_score = float(silhouette_score(X_scaled, labels))
        inertia = float(np.sum([np.sum((X_scaled[labels == i] - X_scaled[labels == i].mean(axis=0))**2) 
                                for i in range(n_clusters)]))
        
        cluster_sizes = {int(i): int(np.sum(labels == i)) for i in range(n_clusters)}
        
        return jsonify({
            "success": True,
            "silhouette_score": sil_score,
            "inertia": inertia,
            "cluster_sizes": cluster_sizes,
            "n_clusters": n_clusters
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
