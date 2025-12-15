from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
import joblib
import io
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Model storage directory
MODEL_DIR = "/tmp/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(model_id):
    return os.path.join(MODEL_DIR, f"{model_id}.pkl")

def load_data(file_url):
    """Load CSV data from URL"""
    df = pd.read_csv(file_url)
    return df

def get_model_class(algorithm, problem_type):
    """Get sklearn model class based on algorithm name"""
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100) if problem_type == 'classification' else RandomForestRegressor(n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier() if problem_type == 'classification' else GradientBoostingRegressor(),
        'decision_tree': DecisionTreeClassifier() if problem_type == 'classification' else DecisionTreeRegressor(),
        'svm': SVC(probability=True) if problem_type == 'classification' else SVR(),
        'knn': KNeighborsClassifier() if problem_type == 'classification' else KNeighborsRegressor(),
        'naive_bayes': GaussianNB(),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso()
    }
    return models.get(algorithm.lower())

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/train', methods=['POST'])
def train_model():
    """Train a model and save it"""
    try:
        data = request.json
        file_url = data['file_url']
        target_column = data['target_column']
        feature_columns = data['feature_columns']
        algorithm = data['algorithm']
        problem_type = data['problem_type']
        model_id = data['model_id']
        
        # Load data
        df = load_data(file_url)
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        start_time = datetime.now()
        model = get_model_class(algorithm, problem_type)
        if model is None:
            return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400
            
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {}
        if problem_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                metrics['classification_report'] = report
            except:
                metrics['classification_report'] = {}
        else:
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            metrics['r2'] = float(r2_score(y_test, y_pred))
        
        # Feature importance
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            for feat, imp in zip(feature_columns, model.feature_importances_):
                feature_importance.append({"feature": feat, "importance": float(imp)})
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            for feat, imp in zip(feature_columns, np.abs(coef)):
                feature_importance.append({"feature": feat, "importance": float(imp)})
        
        # Save model, scaler, and metadata
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'problem_type': problem_type,
            'algorithm': algorithm
        }
        model_path = get_model_path(model_id)
        joblib.dump(model_data, model_path)
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "primary_metric": metrics.get('accuracy') or metrics.get('r2', 0),
            "feature_importance": feature_importance,
            "model_artifact_url": model_path,
            "training_time": training_time,
            "confusion_matrix": metrics.get('confusion_matrix'),
            "classification_report": metrics.get('classification_report')
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/shap', methods=['POST'])
def shap_analysis():
    """Generate SHAP values using saved model"""
    try:
        data = request.json
        model_artifact_url = data['model_artifact_url']
        file_url = data['file_url']
        feature_columns = data['feature_columns']
        sample_size = data.get('sample_size', 100)
        
        # Load saved model
        model_data = joblib.load(model_artifact_url)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Load data
        df = load_data(file_url)
        X = df[feature_columns].head(sample_size)
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        X_scaled = scaler.transform(X)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)
        
        return jsonify({
            "success": True,
            "shap_values": shap_values.values.tolist() if hasattr(shap_values, 'values') else shap_values.tolist(),
            "base_value": float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0,
            "feature_names": feature_columns,
            "summary_plot": {"data": "available"},
            "interactions": None,
            "force_plot": {"data": "available"}
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/tree-extract', methods=['POST'])
def tree_extract():
    """Extract tree structure from saved model"""
    try:
        data = request.json
        model_artifact_url = data['model_artifact_url']
        feature_columns = data['feature_columns']
        max_depth = data.get('max_depth', 5)
        
        # Load saved model
        model_data = joblib.load(model_artifact_url)
        model = model_data['model']
        
        # Extract tree structure
        if hasattr(model, 'tree_'):
            tree = model.tree_
            node_count = tree.node_count
            leaf_count = np.sum(tree.children_left == -1)
            
            return jsonify({
                "success": True,
                "tree": {"structure": "available"},
                "node_count": int(node_count),
                "leaf_count": int(leaf_count),
                "max_depth": int(tree.max_depth),
                "features": feature_columns,
                "classes": model.classes_.tolist() if hasattr(model, 'classes_') else []
            })
        else:
            return jsonify({"error": "Model does not have tree structure", "success": False}), 400
            
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/correlation', methods=['POST'])
def correlation_analysis():
    """Calculate correlation matrix"""
    try:
        data = request.json
        file_url = data['file_url']
        method = data.get('method', 'pearson')
        features = data.get('features', [])
        
        # Load data
        df = load_data(file_url)
        
        # Select numeric columns
        if features:
            df_numeric = df[features].select_dtypes(include=[np.number])
        else:
            df_numeric = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = df_numeric.corr(method='pearson')
        else:
            corr_matrix = df_numeric.corr(method='spearman')
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(val)
                    })
        
        # Calculate VIF
        vif_data = {}
        try:
            for i, col in enumerate(df_numeric.columns):
                vif_data[col] = float(variance_inflation_factor(df_numeric.values, i))
        except:
            vif_data = {col: 1.0 for col in df_numeric.columns}
        
        return jsonify({
            "success": True,
            "matrix": corr_matrix.to_dict(),
            "features": list(corr_matrix.columns),
            "high_corr": high_corr,
            "multicollinearity": len(high_corr) > 0,
            "vif": vif_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/cluster', methods=['POST'])
def clustering_analysis():
    """Perform clustering analysis"""
    try:
        data = request.json
        file_url = data['file_url']
        n_clusters = data.get('n_clusters', 3)
        algorithm = data.get('algorithm', 'kmeans')
        features = data.get('features', [])
        
        # Load data
        df = load_data(file_url)
        
        # Select features
        if features:
            X = df[features].select_dtypes(include=[np.number])
        else:
            X = df.select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        clusters = clusterer.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = float(silhouette_score(X_scaled, clusters))
        
        # Cluster sizes
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
        
        # Inertia (for kmeans)
        inertia = float(clusterer.inertia_) if hasattr(clusterer, 'inertia_') else 0
        
        return jsonify({
            "success": True,
            "clusters": clusters.tolist(),
            "centers": clusterer.cluster_centers_.tolist() if hasattr(clusterer, 'cluster_centers_') else [],
            "silhouette_score": silhouette,
            "dendrogram": {"data": "available"},
            "feature_importance": [],
            "cluster_sizes": cluster_sizes,
            "inertia": inertia
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
