import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             silhouette_score, davies_bouldin_score,
                             accuracy_score, f1_score, precision_score, 
                             recall_score, classification_report)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
RANDOM_STATE = 42
CV_FOLDS = 5


def target_encode_oof(df, col, target_col, n_splits=5):
    """Compute out-of-fold target mean encoding."""
    oof = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in kf.split(df):
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        means = train.groupby(col)[target_col].mean()
        oof.iloc[val_idx] = val[col].map(means).fillna(df[target_col].mean())
    oof.fillna(df[target_col].mean(), inplace=True)
    return oof


class AmesHousingEvaluator:
    """
    Comprehensive Ames Housing evaluation: Regression, Clustering, Classification.
    """
    
    def __init__(self, filepath, target_column='SalePrice'):
        self.filepath = filepath
        self.target_column = target_column
        self.df = None
        self.df_preprocessed = None
        self.selected_features = []
        self.results = {}
        self.preprocessing_applied = False
        
    def load_and_analyze(self):
        self.df = pd.read_csv(self.filepath)
        print(self.df.dtypes.value_counts())
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage': missing_pct})
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\nMissing values detected in {len(missing_df)} features:")
            print(missing_df.head(10))
            print(f"\nTotal missing cells: {missing.sum()} ({missing.sum() / (self.df.shape[0] * self.df.shape[1]) * 100:.2f}% of dataset)")
        
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        
        if self.target_column in self.df.columns:
            print(f"\n--- Target Variable: {self.target_column} ---")
            print(self.df[self.target_column].describe())
            print(f"\nDistribution metrics:")
            print(f"  Skewness: {self.df[self.target_column].skew():.4f}")
            print(f"  Kurtosis: {self.df[self.target_column].kurtosis():.4f}")
            if abs(self.df[self.target_column].skew()) > 0.75:
                print(f"  Note: High skewness detected. Log transformation recommended.")
        
        return self
    
    def preprocess_data(self, handle_outliers=True, cardinality_threshold=10):
        """Apply comprehensive preprocessing."""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        df = self.df.copy()
        
        # Drop ID columns
        id_cols = ['Order', 'PID']
        existing_ids = [c for c in id_cols if c in df.columns]
        if existing_ids:
            df.drop(columns=existing_ids, inplace=True)
            print(f"\nRemoved ID columns: {existing_ids}")
        
        # Feature Engineering
        print("\n--- Feature Engineering ---")
        rename_map = {'1st Flr SF': 'FirstFlrSF', '2nd Flr SF': 'SecondFlrSF', 'Total Bsmt SF': 'TotalBsmtSF'}
        df.rename(columns=rename_map, inplace=True)
        
        area_parts = [c for c in ['FirstFlrSF', 'SecondFlrSF', 'TotalBsmtSF'] if c in df.columns]
        if area_parts:
            df['TotalSF'] = df[area_parts].sum(axis=1)
            print(f"Created TotalSF from {len(area_parts)} area components")
        
        if 'Year Built' in df.columns and 'Yr Sold' in df.columns:
            df['HouseAge'] = df['Yr Sold'] - df['Year Built']
            print("Created HouseAge feature")
            
        if 'Year Remod/Add' in df.columns and 'Yr Sold' in df.columns:
            df['SinceRemod'] = df['Yr Sold'] - df['Year Remod/Add']
            print("Created SinceRemod feature")
        
        binary_features = []
        if 'Pool Area' in df.columns:
            df['HasPool'] = (df['Pool Area'] > 0).astype(int)
            binary_features.append('HasPool')
        if 'Fireplaces' in df.columns:
            df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
            binary_features.append('HasFireplace')
        if 'Garage Type' in df.columns:
            df['HasGarage'] = (~df['Garage Type'].isnull()).astype(int)
            binary_features.append('HasGarage')
        
        if binary_features:
            print(f"Created binary features: {', '.join(binary_features)}")
        if self.target_column in df.columns:
            target_skew = df[self.target_column].skew()
            if abs(target_skew) > 0.75:
                df['SalePrice_log'] = np.log1p(df[self.target_column])
                new_skew = df['SalePrice_log'].skew()
                print(f"\n--- Target Transformation ---")
                print(f"Applied log transformation to {self.target_column}")
                print(f"  Original skewness: {target_skew:.4f}")
                print(f"  Transformed skewness: {new_skew:.4f}")
        
        # Identify features
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for t in [self.target_column, 'SalePrice_log']:
            if t in num_cols:
                num_cols.remove(t)
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        low_card = [c for c in cat_cols if df[c].nunique() <= cardinality_threshold]
        high_card = [c for c in cat_cols if df[c].nunique() > cardinality_threshold]
        
        print(f"\n--- Feature Type Analysis ---")
        print(f"Numeric features: {len(num_cols)}")
        print(f"Low cardinality categorical: {len(low_card)}")
        print(f"High cardinality categorical: {len(high_card)}")
        
        # Missing values
        print(f"\n--- Missing Value Imputation ---")
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            for c in num_cols:
                if df[c].isnull().sum() > 0:
                    df[c] = df[c].fillna(df[c].median())
            for c in cat_cols:
                df[c] = df[c].fillna('Missing')
            
            missing_after = df.isnull().sum().sum()
            print(f"Missing values before: {missing_before}")
            print(f"Missing values after: {missing_after}")
        
        # Outliers
        if handle_outliers:
            outliers_total = 0
            outlier_features = [] 
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_features.append((col, outliers))
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    outliers_total += outliers
            
            if outlier_features:
                outlier_features.sort(key=lambda x: x[1], reverse=True)
                print(f"Features with outliers clipped (top 10):")
                for feat, count in outlier_features[:10]:
                    print(f"  {feat}: {count} outliers ({count/len(df)*100:.2f}%)")
                print(f"\nTotal outliers clipped: {outliers_total}")
        if low_card:
            df = pd.get_dummies(df, columns=low_card, drop_first=True)
            print(f"One-hot encoded {len(low_card)} low cardinality features")
        
        if high_card:
            print(f"Encoding {len(high_card)} high cardinality features:")
            for c in high_card:
                df[f'{c}_freq'] = df[c].map(df[c].value_counts(normalize=True))
                if 'SalePrice_log' in df.columns:
                    df[f'{c}_target_oof'] = target_encode_oof(df, c, 'SalePrice_log', n_splits=5)
                    print(f"  {c}: frequency + OOF target encoding")
                df.drop(columns=[c], inplace=True)
        
        # Remove zero variance
        print(f"\n--- Zero Variance Feature Removal ---")
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [c for c in numeric_features if c not in [self.target_column, 'SalePrice_log']]
        
        zero_var_features = [col for col in numeric_features if df[col].var() < 1e-10]
        if zero_var_features:
            df.drop(columns=zero_var_features, inplace=True)
            print(f"Removed {len(zero_var_features)} zero variance features")
        
        # Remove raw target
        if self.target_column in df.columns:
            df.drop(columns=[self.target_column], inplace=True)
            print(f"\n--- Target Variable Handling ---")
            print(f"Removed raw {self.target_column} from feature set to prevent leakage")
        
        # Standardize
        print(f"\n--- Feature Standardization ---")
        scaler = StandardScaler()
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice_log' in numeric_features:
            numeric_features.remove('SalePrice_log')
        
        if numeric_features:
            df[numeric_features] = scaler.fit_transform(df[numeric_features])
            print(f"Standardized {len(numeric_features)} numeric features (mean=0, std=1)")
        
        self.df_preprocessed = df
        self.preprocessing_applied = True
        
        print("\n" + "=" * 80)
        print(f"PREPROCESSING COMPLETED")
        print(f"Final dataset shape: {df.shape}")
        print("=" * 80)
        
        return self
    
    def select_features_mi(self, k=40):
        """Select top k features using mutual information."""
        print("\n" + "=" * 80)
        print("FEATURE SELECTION - MUTUAL INFORMATION")
        print("=" * 80)
        
        df = self.df_preprocessed.copy()
        X = df.drop(columns=['SalePrice_log'])
        y = df['SalePrice_log']
        
        print(f"\nCalculating mutual information for {X.shape[1]} features...")
        
        mi = mutual_info_regression(X.fillna(0), y, random_state=RANDOM_STATE)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        
        print(f"\nTop 20 features by mutual information:")
        print(f"{'Feature':<35} {'MI Score':<12}")
        print("-" * 50)
        for feat, score in mi_series.head(20).items():
            print(f"{feat:<35} {score:>10.6f}")
        
        top_features = mi_series.head(k).index.tolist()
        self.selected_features = top_features
        
        print(f"\nSelected top {k} features for modeling")
        
        return self
    
    def evaluate_regression(self, n_splits=5):
        """Evaluate regression with cross-validation."""
        print("\n" + "=" * 80)
        print("REGRESSION EVALUATION - CROSS-VALIDATION")
        print("=" * 80)
        
        df = self.df_preprocessed.copy()
        X = df[self.selected_features]
        y = df['SalePrice_log']
        
        print(f"\nFeatures used: {len(self.selected_features)}")
        print(f"Training samples: {len(X)}")
        print(f"Cross-validation folds: {n_splits}")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        rmses, r2s, maes = [], [], []
        
        print(f"\n--- Cross-Validation Results ---")
        print(f"{'Fold':<8} {'RMSE (log)':<15} {'R² Score':<15} {'MAE (log)':<15}")
        print("-" * 60)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, max_depth=30)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            r2 = r2_score(y_val, predictions)
            mae = mean_absolute_error(y_val, predictions)
            
            rmses.append(rmse)
            r2s.append(r2)
            maes.append(mae)
            
            print(f"Fold {fold:<3} {rmse:>13.4f}  {r2:>13.4f}  {mae:>13.4f}")
        
        mean_rmse, mean_r2, mean_mae = np.mean(rmses), np.mean(r2s), np.mean(maes)
        std_rmse, std_r2 = np.std(rmses), np.std(r2s)
        
        print("-" * 60)
        print(f"Mean     {mean_rmse:>13.4f}  {mean_r2:>13.4f}  {mean_mae:>13.4f}")
        print(f"Std Dev  {std_rmse:>13.4f}  {std_r2:>13.4f}  {np.std(maes):>13.4f}")
        
        # Feature importance
        print(f"\n--- Feature Importance Analysis ---")
        final_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        final_model.fit(X, y)
        
        importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        print(f"\nTop 15 most important features:")
        print(f"{'Feature':<35} {'Importance':<12}")
        print("-" * 50)
        for feat, imp in importances.head(15).items():
            print(f"{feat:<35} {imp:>10.6f}")
        
        # Verdict
        print(f"\n--- Performance Assessment ---")
        if mean_r2 > 0.85:
            verdict = "EXCELLENT - Very strong predictive performance"
        elif mean_r2 > 0.75:
            verdict = "VERY GOOD - Strong predictive performance"
        elif mean_r2 > 0.65:
            verdict = "GOOD - Solid predictive performance"
        else:
            verdict = "FAIR - Moderate predictive performance"
        
        print(f"Overall verdict: {verdict}")
        print(f"Mean R² Score: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"Mean RMSE (log scale): {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        self.results['regression'] = {
            'mean_r2': mean_r2, 'std_r2': std_r2,
            'mean_rmse': mean_rmse, 'std_rmse': std_rmse,
            'mean_mae': mean_mae, 'verdict': verdict,
            'feature_importances': importances
        }
        
        return self
    
    def evaluate_clustering(self, method='kmeans', k_range=range(2, 11)):
        """Evaluate clustering."""
        print("\n" + "=" * 80)
        print(f"CLUSTERING EVALUATION - {method.upper()}")
        print("=" * 80)
        
        df = self.df_preprocessed.copy()
        
        # Select features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'SalePrice_log']
        variances = df[numeric_cols].var().sort_values(ascending=False)
        features_for_clustering = variances.head(15).index.tolist()
        
        X_cluster = df[features_for_clustering].copy()
        
        print(f"\nFeatures used: {len(features_for_clustering)}")
        print(f"Samples: {len(X_cluster)}")
        
        # PCA for better clustering
        print(f"\n--- Applying PCA for dimensionality reduction ---")
        pca = PCA(n_components=0.85, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_cluster)
        print(f"Reduced to {X_pca.shape[1]} components (85% variance retained)")
        
        if method == 'kmeans':
            print(f"\n--- K-Means Evaluation ---")
            print(f"{'K':<8} {'Silhouette':<15} {'Davies-Bouldin':<18} {'Inertia':<15}")
            print("-" * 65)
            
            silhouette_scores, db_scores, inertias = [], [], []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                labels = kmeans.fit_predict(X_pca)
                
                silhouette = silhouette_score(X_pca, labels)
                db = davies_bouldin_score(X_pca, labels)
                inertia = kmeans.inertia_
                
                silhouette_scores.append(silhouette)
                db_scores.append(db)
                inertias.append(inertia)
                
                print(f"{k:<8} {silhouette:>13.4f}  {db:>16.4f}  {inertia:>13.2f}")
            
            best_k = k_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
            
            print(f"\n--- Optimal Configuration ---")
            print(f"Best K: {best_k}")
            print(f"Best Silhouette Score: {best_silhouette:.4f}")
            print(f"Davies-Bouldin Score at best K: {db_scores[np.argmax(silhouette_scores)]:.4f}")
            
            verdict = self._clustering_verdict(best_silhouette)
            print(f"\n--- Performance Assessment ---")
            print(f"Overall verdict: {verdict}")
            
            self.results['clustering'] = {
                'method': 'kmeans', 'best_k': best_k,
                'best_silhouette': best_silhouette,
                'verdict': verdict
            }
        
        elif method == 'dbscan':
            print(f"\n--- DBSCAN Evaluation ---")
            best_silhouette = -1
            best_params = None
            
            for eps in [0.5, 1.0, 1.5]:
                for min_samples in [5, 10, 15]:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_pca)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters > 1:
                        silhouette = silhouette_score(X_pca, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_params = {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters}
            
            if best_params:
                print(f"Best parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
                print(f"Number of clusters: {best_params['n_clusters']}")
                print(f"Silhouette Score: {best_silhouette:.4f}")
                
                verdict = self._clustering_verdict(best_silhouette)
                print(f"\n--- Performance Assessment ---")
                print(f"Overall verdict: {verdict}")
                
                self.results['clustering'] = {
                    'method': 'dbscan',
                    'best_params': best_params,
                    'best_silhouette': best_silhouette,
                    'verdict': verdict
                }
        
        return self
    
    def _clustering_verdict(self, silhouette):
        """Helper to determine clustering verdict."""
        if silhouette > 0.50:
            return "EXCELLENT - Strong, well-separated clusters"
        elif silhouette > 0.35:
            return "GOOD - Clear cluster structure present"
        elif silhouette > 0.25:
            return "FAIR - Weak but detectable structure"
        elif silhouette > 0.15:
            return "POOR - Minimal cluster structure"
        else:
            return "VERY POOR - No meaningful cluster structure"
    
    def evaluate_classification(self, n_bins=3, n_splits=5):
        """Evaluate classification."""
        print("\n" + "=" * 80)
        print("CLASSIFICATION EVALUATION - CROSS-VALIDATION")
        print("=" * 80)
        
        df = self.df_preprocessed.copy()
        
        # Create categories
        bins = [df['SalePrice_log'].min() - 1] + \
               list(df['SalePrice_log'].quantile([0.33, 0.67]).values) + \
               [df['SalePrice_log'].max() + 1]
        labels = ['Low', 'Medium', 'High']
        
        df['Price_Category'] = pd.cut(df['SalePrice_log'], bins=bins, labels=labels)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df['Price_Category'])
        X = df[self.selected_features]
        
        print(f"\nFeatures used: {len(self.selected_features)}")
        print(f"Number of classes: {n_bins}")
        print(f"Training samples: {len(X)}")
        
        # Class distribution
        print(f"\n--- Class Distribution ---")
        class_counts = pd.Series(df['Price_Category']).value_counts().sort_index()
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
        
        balance_ratio = class_counts.min() / class_counts.max()
        print(f"\nClass balance ratio: {balance_ratio:.4f}")
        
        # Cross-validation
        print(f"\n--- Cross-Validation Results ---")
        print(f"{'Fold':<8} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 65)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        accuracies, f1_scores_list, precisions, recalls = [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            
            acc = accuracy_score(y_val, predictions)
            f1 = f1_score(y_val, predictions, average='weighted')
            prec = precision_score(y_val, predictions, average='weighted', zero_division=0)
            rec = recall_score(y_val, predictions, average='weighted')
            
            accuracies.append(acc)
            f1_scores_list.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            
            print(f"Fold {fold:<3} {acc:>10.4f}  {f1:>10.4f}  {prec:>10.4f}  {rec:>10.4f}")
        
        mean_acc = np.mean(accuracies)
        mean_f1 = np.mean(f1_scores_list)
        
        print("-" * 65)
        print(f"Mean     {mean_acc:>10.4f}  {mean_f1:>10.4f}  {np.mean(precisions):>10.4f}  {np.mean(recalls):>10.4f}")
        print(f"Std Dev  {np.std(accuracies):>10.4f}  {np.std(f1_scores_list):>10.4f}  {np.std(precisions):>10.4f}  {np.std(recalls):>10.4f}")
        
        baseline_acc = class_counts.max() / len(df)
        improvement = mean_acc - baseline_acc
        # Verdict
        if mean_f1 > 0.85:
            verdict = "EXCELLENT - Very strong classification performance"
        elif mean_f1 > 0.75:
            verdict = "VERY GOOD - Strong classification performance"
        elif mean_f1 > 0.65:
            verdict = "GOOD - Solid classification performance"
        else:
            verdict = "FAIR - Moderate classification performance"
        
        print(f"\n--- Performance Assessment ---")
        print(f"Overall verdict: {verdict}")
        print(f"Mean F1-Score: {mean_f1:.4f} ± {np.std(f1_scores_list):.4f}")
        
        self.results['classification'] = {
            'mean_accuracy': mean_acc, 'mean_f1': mean_f1,
            'baseline_accuracy': baseline_acc,
            'balance_ratio': balance_ratio,
            'verdict': verdict
        }
        
        return self
    
    def generate_summary(self):
        """Generate comprehensive summary."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION SUMMARY - AMES HOUSING")
        print("=" * 80)
        
        print("\n--- Dataset Information ---")
        print(f"Original dimensions: {self.df.shape}")
        print(f"Preprocessed dimensions: {self.df_preprocessed.shape}")
        print(f"Preprocessing applied: {'Yes' if self.preprocessing_applied else 'No'}")        
        print("\n" + "=" * 80)
        print("TASK-SPECIFIC PERFORMANCE RESULTS")
        print("=" * 80)
        
        # Regression results
        if 'regression' in self.results:
            reg = self.results['regression']
            print("\n[1] REGRESSION PERFORMANCE")
            print("-" * 80)
            print(f"Model: Random Forest Regressor (200 trees)")
            print(f"Cross-validation: 5-fold")
            print(f"\nMetrics:")
            print(f"  R² Score: {reg['mean_r2']:.4f} ± {reg['std_r2']:.4f}")
            print(f"  RMSE (log scale): {reg['mean_rmse']:.4f} ± {reg['std_rmse']:.4f}")
            print(f"  MAE (log scale): {reg['mean_mae']:.4f}")
            print(f"\nVerdict: {reg['verdict']}")
            
            print(f"\nTop 5 most important features:")
            for i, (feat, imp) in enumerate(reg['feature_importances'].head(5).items(), 1):
                print(f"  {i}. {feat}: {imp:.4f}")
        
        # Clustering results
        if 'clustering' in self.results:
            clust = self.results['clustering']
            print("\n[2] CLUSTERING PERFORMANCE")
            print("-" * 80)
            print(f"Algorithm: {clust['method'].upper()}")
            if clust['method'] == 'kmeans':
                print(f"  Best K: {clust['best_k']}")
                print(f"  Silhouette Score: {clust['best_silhouette']:.4f}")
            elif clust['method'] == 'dbscan':
                print(f"\nOptimal configuration:")
                print(f"  Parameters: {clust['best_params']}")
                print(f"  Silhouette Score: {clust['best_silhouette']:.4f}")
            print(f"\nVerdict: {clust['verdict']}")
        
        # Classification results
        if 'classification' in self.results:
            clf = self.results['classification']
            print("\n[3] CLASSIFICATION PERFORMANCE")
            print("-" * 80)
            print(f"Model: Random Forest Classifier (200 trees)")
            print(f"Number of classes: 3 (Low, Medium, High)")
            print(f"Cross-validation: 5-fold")
            print(f"Class balance ratio: {clf['balance_ratio']:.4f}")
            print(f"\nMetrics:")
            print(f"  Accuracy: {clf['mean_accuracy']:.4f}")
            print(f"  F1-Score (weighted): {clf['mean_f1']:.4f}")
            print(f"\nBaseline comparison:")
            print(f"  Baseline accuracy: {clf['baseline_accuracy']:.4f}")
            print(f"  Improvement: +{clf['mean_accuracy'] - clf['baseline_accuracy']:.4f}")
            print(f"\nVerdict: {clf['verdict']}")
        
        # Overall recommendation
        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT AND RECOMMENDATIONS")
        print("=" * 80)
        
        suitable_tasks = []
        excellent_tasks = []
        
        if 'regression' in self.results:
            if self.results['regression']['mean_r2'] > 0.85:
                excellent_tasks.append("Regression")
                suitable_tasks.append("Regression")
            elif self.results['regression']['mean_r2'] > 0.65:
                suitable_tasks.append("Regression")
        
        if 'clustering' in self.results:
            if self.results['clustering']['best_silhouette'] > 0.50:
                excellent_tasks.append("Clustering")
                suitable_tasks.append("Clustering")
            elif self.results['clustering']['best_silhouette'] > 0.35:
                suitable_tasks.append("Clustering")
        
        if 'classification' in self.results:
            if self.results['classification']['mean_f1'] > 0.85:
                excellent_tasks.append("Classification")
                suitable_tasks.append("Classification")
            elif self.results['classification']['mean_f1'] > 0.70:
                suitable_tasks.append("Classification")
        
        print("\nDataset suitability:")
        if excellent_tasks:
            print(f"  Excellent performance: {', '.join(excellent_tasks)}")
        if suitable_tasks:
            print(f"  Suitable for: {', '.join(suitable_tasks)}")
        # Key strengths
        print("\nKey strengths:")
        strengths = []
        
        if 'regression' in self.results and self.results['regression']['mean_r2'] > 0.75:
            strengths.append("  - Strong predictive features for regression (high R² score)")
        
        if 'clustering' in self.results and self.results['clustering']['best_silhouette'] > 0.30:
            strengths.append("  - Good feature variance supporting cluster formation")
        
        if 'classification' in self.results:
            if self.results['classification']['balance_ratio'] > 0.8:
                strengths.append("  - Well-balanced class distribution")
            if self.results['classification']['mean_f1'] > 0.70:
                strengths.append("  - Strong class separability")
        
        if strengths:
            for s in strengths:
                print(s)
        return self
    
    def run_full_evaluation(self, n_features=40, clustering_method='kmeans', 
                           classification_bins=3, cv_folds=5):
        print("\n" + "=" * 80)
        print("AMES HOUSING DATASET - COMPREHENSIVE MACHINE LEARNING EVALUATION")
        print("=" * 80)
        
        # Execute pipeline
        self.load_and_analyze()
        self.preprocess_data(handle_outliers=True, cardinality_threshold=10)
        self.select_features_mi(k=n_features)
        
        # Evaluate all three tasks
        print("\n[TASK 1/3] Starting Regression Evaluation...")
        self.evaluate_regression(n_splits=cv_folds)
        
        print("\n[TASK 2/3] Starting Clustering Evaluation...")
        self.evaluate_clustering(method=clustering_method)
        
        print("\n[TASK 3/3] Starting Classification Evaluation...")
        self.evaluate_classification(n_bins=classification_bins, n_splits=cv_folds)
        
        # Generate final summary
        self.generate_summary()
        
        return self


# Main execution
if __name__ == "__main__":
    
    print("=" * 80)
    print("AMES HOUSING - COMPREHENSIVE EVALUATION FRAMEWORK")
    print("Regression, Clustering, and Classification Analysis")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = AmesHousingEvaluator(
        filepath='AmesHousing.csv',
        target_column='SalePrice'
    )
    
    evaluator.run_full_evaluation(
        n_features=40,
        clustering_method='kmeans',  # preferred method
        classification_bins=3,        # Low, Medium, High price categories
        cv_folds=5
    )
    
    print("\n" + "=" * 80)
    print("All evaluations completed successfully")
    print("=" * 80)
    
    # Optional: Print results summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY TABLE")
    print("=" * 80)
    
    if evaluator.results:
        print(f"\n{'Task':<20} {'Metric':<25} {'Value':<15} {'Verdict'}")
        print("-" * 80)
        
        if 'regression' in evaluator.results:
            reg = evaluator.results['regression']
            print(f"{'Regression':<20} {'R² Score':<25} {reg['mean_r2']:.4f}          {reg['verdict'].split('-')[0].strip()}")
            print(f"{'':20} {'RMSE (log)':<25} {reg['mean_rmse']:.4f}")
        
        if 'clustering' in evaluator.results:
            clust = evaluator.results['clustering']
            method_display = f"{clust['method'].upper()}"
            if clust['method'] == 'kmeans':
                print(f"{'Clustering':<20} {method_display + ' (K=' + str(clust['best_k']) + ')':<25} {clust['best_silhouette']:.4f}          {clust['verdict'].split('-')[0].strip()}")
            else:
                print(f"{'Clustering':<20} {method_display:<25} {clust['best_silhouette']:.4f}          {clust['verdict'].split('-')[0].strip()}")
        
        if 'classification' in evaluator.results:
            clf = evaluator.results['classification']
            print(f"{'Classification':<20} {'F1-Score':<25} {clf['mean_f1']:.4f}          {clf['verdict'].split('-')[0].strip()}")
            print(f"{'':20} {'Accuracy':<25} {clf['mean_accuracy']:.4f}")
            print(f"{'':20} {'Balance Ratio':<25} {clf['balance_ratio']:.4f}")
    
    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)