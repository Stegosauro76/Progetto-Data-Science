
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold

import logging
from typing import Optional, List, Tuple

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
logger = logging.getLogger("AmesVisualizer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class AmesVisualizer:
    """
    Improved and critical revision of the original AmesHousingVisualizer.
    Key improvements:
      - explicit outlier handling options (report, winsorize, drop)
      - robust and configurable preprocessing
      - formal normality tests + clear handling of log/log1p
      - targeted correlation heatmaps (top-k), not full unreadable matrix
      - safer clustering pipeline and protected silhouette computations
      - permutation importance with CV for feature importance
    """

    def __init__(self, filepath: str, random_state: int = RANDOM_STATE):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.pca_model: Optional[PCA] = None
        self.df_pca: Optional[np.ndarray] = None
        self.random_state = random_state

    def load(self) -> 'AmesVisualizer':
        self.df = pd.read_csv(self.filepath)
        logger.info(f"Loaded dataset with shape: {self.df.shape}")
        return self
    @staticmethod
    def _freedman_diaconis_bins(x: np.ndarray) -> int:
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return 10
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            return int(np.sqrt(len(x)))
        h = 2 * iqr * (len(x) ** (-1/3))
        if h <= 0:
            return int(np.sqrt(len(x)))
        return max(10, int(np.ceil((x.max() - x.min()) / h)))

    # -----------------------
    # Target distribution / outliers
    # -----------------------
    def analyze_target(self, target='SalePrice', show_plots: bool = True,
                       outlier_method: Optional[str] = None) -> dict:
        """
        Analyzes SalePrice distribution, performs log transforms, runs normality tests,
        and optionally handles outliers.
        outlier_method: None | 'report' | 'winsorize' | 'drop'
        Returns summary dict with statistics and decisions.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        s = self.df[target].dropna()
        summary = {
            'count': int(s.count()),
            'mean': float(s.mean()),
            'median': float(s.median()),
            'std': float(s.std()),
            'skew': float(s.skew()),
            'kurtosis': float(s.kurtosis()),
            'min': float(s.min()),
            'max': float(s.max())
        }

        # Bins via Freedman–Diaconis
        bins = self._freedman_diaconis_bins(s.values)
        if (s <= 0).any():
            logger.warning("Target contains <=0 values; using log1p for safety.")
            log_s = np.log1p(s.clip(lower=0))
            used_transform = 'log1p'
        else:
            log_s = np.log1p(s) 
            used_transform = 'log1p'

        # Normality tests (Anderson-Darling)
        ad_result = stats.anderson(log_s, dist='norm')
        normaltest = None
        try:
            normaltest = stats.normaltest(log_s)  
            normaltest_p = float(normaltest.pvalue)
        except Exception:
            normaltest_p = None

        summary.update({
            'transform_used': used_transform,
            'ad_statistic': float(ad_result.statistic),
            'ad_critical': ad_result.critical_values.tolist(),
            'normaltest_pvalue': normaltest_p
        })

        # Outlier detection (IQR)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        summary['outliers_count'] = int(outliers.count())
        summary['outliers_fraction'] = float(outliers.count() / s.count())
        if outlier_method == 'winsorize':
            s_w = s.clip(lower=lower, upper=upper)
            self.df[target] = self.df[target].where(self.df[target].notna(), s_w)
            logger.info("Applied winsorization to target.")
        elif outlier_method == 'drop':
            idx_to_drop = outliers.index
            self.df = self.df.drop(index=idx_to_drop).reset_index(drop=True)
            logger.info(f"Dropped {len(idx_to_drop)} outlier rows from dataframe.")
        elif outlier_method == 'report' or outlier_method is None:
            
            pass
        else:
            raise ValueError("outlier_method must be one of None|'report'|'winsorize'|'drop'")

        if show_plots:
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()
            axs[0].hist(s, bins=bins, edgecolor='black', alpha=0.7)
            axs[0].set_title(f"{target} distribution (bins={bins})")
            axs[0].axvline(summary['mean'], color='red', linestyle='--', label='mean')
            axs[0].axvline(summary['median'], color='green', linestyle='-.', label='median')
            axs[0].legend()
            axs[1].boxplot(s, vert=False, showfliers=True)
            axs[1].set_title("Boxplot (outliers shown)")
            axs[2].hist(log_s, bins=self._freedman_diaconis_bins(log_s.values), edgecolor='black', alpha=0.7)
            axs[2].set_title(f"{used_transform}({target}) distribution")

            # Q-Q
            stats.probplot(log_s, dist="norm", plot=axs[3])
            axs[3].set_title("Q-Q plot (log-transformed)")

            # KDE + rug
            sns.kdeplot(s, ax=axs[4], fill=True)
            axs[4].set_title("KDE of target")

            # Outlier scatter (index vs value)
            axs[5].scatter(range(len(s)), s, s=6, alpha=0.6)
            if len(outliers) > 0:
                axs[5].scatter(outliers.index, outliers.values, color='red', s=20, label='outliers')
                axs[5].legend()
            axs[5].set_title("Index vs Value (outliers highlighted)")

            plt.tight_layout()
            plt.show()
        return summary
    def show_correlations(self, target='SalePrice', top_k: int = 15, figsize: Tuple[int, int] = (12, 8)):
        """
        Show:
          - top_k features by absolute Pearson correlation with target (table + bar)
          - heatmap of correlations among top_k features (triangular mask)
          - scatter plots (pairgrid) for top few features vs target
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        numeric = self.df.select_dtypes(include=[np.number]).copy()
        if target not in numeric.columns:
            raise ValueError(f"{target} not found among numeric columns.")

        corr_with_target = numeric.corrwith(numeric[target]).abs().sort_values(ascending=False)
        corr_with_target = corr_with_target.drop(target, errors='ignore')
        top_features = corr_with_target.head(top_k).index.tolist()

        # Bar plot of correlations
        fig, ax = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] / 1.6))
        sns.barplot(x=corr_with_target.head(top_k).values, y=corr_with_target.head(top_k).index, ax=ax[0])
        ax[0].set_title(f"Top {top_k} features by |Pearson corr| with {target}")
        ax[0].set_xlabel("Absolute Pearson correlation")

        # Heatmap of top features
        sub_corr = numeric[top_features + [target]].corr()
        mask = np.triu(np.ones_like(sub_corr, dtype=bool))
        sns.heatmap(sub_corr, mask=mask, cmap='coolwarm', annot=False, center=0, ax=ax[1])
        ax[1].set_title("Correlation matrix (top features + target)")

        plt.tight_layout()
        plt.show()

        # Scatter plots for the top 6 features vs target
        top_scatter = top_features[:6]
        if len(top_scatter) > 0:
            sns.pairplot(self.df, x_vars=top_scatter, y_vars=[target], kind='reg', height=3.2)
            plt.suptitle("Top features vs Target (regression line)", y=1.02)
            plt.show()

        # Return numeric summary for programmatic use
        return corr_with_target.head(top_k)

    def missing_value_report(self, top_n: int = 30, figsize: Tuple[int, int] = (12, 8)):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

        if missing_df.empty:
            logger.info("No missing values found.")
            return missing_df

        display_df = missing_df.head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        display_df['missing_pct'].plot(kind='barh', ax=ax)
        ax.set_xlabel("Percent missing (%)")
        ax.set_title(f"Top {min(top_n, len(display_df))} features with missing values")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        total_cells = self.df.size
        total_missing = missing.sum()
        logger.info(f"Total missing cells: {total_missing:,} ({(total_missing/total_cells)*100:.2f}%)")
        return missing_df

    def prepare_numeric(self, drop_cols: Optional[List[str]] = None,
                        impute_strategy: str = 'median',
                        scaler: str = 'standard') -> pd.DataFrame:
        """
        Returns numeric dataframe cleaned:
          - drop ids/target by default
          - impute with median or mean
          - apply scaler: 'standard' or 'robust'
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        numeric = self.df.select_dtypes(include=[np.number]).copy()
        if drop_cols is None:
            drop_cols = ['Order', 'PID', 'SalePrice']

        cols = [c for c in numeric.columns if c not in drop_cols]
        df_num = numeric[cols].copy()

        # Impute
        if impute_strategy not in ('median', 'mean'):
            raise ValueError("impute_strategy must be 'median' or 'mean'")
        for c in df_num.columns:
            if df_num[c].isnull().any():
                if impute_strategy == 'median':
                    df_num[c].fillna(df_num[c].median(), inplace=True)
                else:
                    df_num[c].fillna(df_num[c].mean(), inplace=True)

        # Scale
        if scaler == 'standard':
            sc = StandardScaler()
        elif scaler == 'robust':
            sc = RobustScaler()
        else:
            raise ValueError("scaler must be 'standard' or 'robust'")

        df_scaled = pd.DataFrame(sc.fit_transform(df_num), columns=df_num.columns)
        self.df_clean = df_scaled
        logger.info(f"Prepared numeric data: {df_scaled.shape} (scaler={scaler}, impute={impute_strategy})")
        return df_scaled

    # -----------------------
    # PCA analysis
    # -----------------------
    def run_pca(self, explained_variance: float = 0.85, min_components: int = 2):
        if self.df_clean is None:
            self.prepare_numeric()

        pca = PCA(n_components=explained_variance, random_state=self.random_state)
        X_pca = pca.fit_transform(self.df_clean)
        n_comp = X_pca.shape[1]
        if n_comp < min_components:
            # rerun with explicit components
            pca = PCA(n_components=min_components, random_state=self.random_state)
            X_pca = pca.fit_transform(self.df_clean)

        self.pca_model = pca
        self.df_pca = X_pca
        logger.info(f"PCA produced {X_pca.shape[1]} components, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        return X_pca

    def plot_pca_scree(self):
        if self.pca_model is None:
            raise RuntimeError("PCA not run. Call run_pca() first.")
        evr = self.pca_model.explained_variance_ratio_
        cumsum = np.cumsum(evr)
        plt.figure(figsize=(10, 4))
        plt.bar(range(1, len(evr) + 1), evr, alpha=0.6, label='Individual')
        plt.plot(range(1, len(evr) + 1), cumsum, marker='o', color='black', label='Cumulative')
        plt.axhline(0.85, color='red', linestyle='--', label='85%')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Scree and Cumulative Explained Variance')
        plt.legend()
        plt.show()

    # -----------------------
    # KMeans clustering
    # -----------------------
    def find_optimal_k(self, k_range: range = range(2, 11)) -> int:
        if self.df_pca is None:
            self.run_pca()

        inertias = []
        silhouettes = []
        valid_ks = []
        logger.info("Computing KMeans inertias and silhouette scores...")

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=50)
            labels = kmeans.fit_predict(self.df_pca)
            inertias.append(kmeans.inertia_)
            if len(set(labels)) > 1 and len(labels) > k:
                try:
                    sil = silhouette_score(self.df_pca, labels)
                except Exception:
                    sil = -1
            else:
                sil = -1
            silhouettes.append(sil)
            valid_ks.append(k)
            logger.info(f"K={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil:.4f}")

        # pick best k by silhouette (robust)
        best_idx = int(np.argmax(silhouettes))
        best_k = valid_ks[best_idx]
        # plot
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        axs[0].plot(valid_ks, inertias, marker='o')
        axs[0].set_title("Elbow (inertia)")
        axs[0].set_xlabel("k")
        axs[0].set_ylabel("Inertia")
        axs[1].plot(valid_ks, silhouettes, marker='o', color='orange')
        axs[1].set_title("Silhouette scores")
        axs[1].set_xlabel("k")
        axs[1].set_ylabel("Silhouette")
        plt.tight_layout()
        plt.show()

        logger.info(f"Selected optimal k={best_k} by silhouette (score={silhouettes[best_idx]:.4f})")
        return best_k

    def cluster_and_plot(self, n_clusters: int = 3, show_3d: bool = False):
        if self.df_pca is None:
            self.run_pca()
        if self.df_pca.shape[1] < 2:
            raise RuntimeError("PCA produced <2 components; cannot plot in 2D.")

        kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=self.random_state)
        labels = kmeans.fit_predict(self.df_pca)
        centers = kmeans.cluster_centers_

        # 2D scatter
        plt.figure(figsize=(10, 4))
        ax = plt.subplot(121)
        sc = ax.scatter(self.df_pca[:, 0], self.df_pca[:, 1], c=labels, cmap='tab10', s=20, alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=150, c='red', edgecolor='black')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"KMeans clusters (k={n_clusters})")
        plt.colorbar(sc, ax=ax)

        # Cluster sizes
        ax2 = plt.subplot(122)
        counts = pd.Series(labels).value_counts().sort_index()
        ax2.bar(counts.index.astype(str), counts.values, color='skyblue', edgecolor='black')
        ax2.set_title("Cluster sizes")
        for i, v in enumerate(counts.values):
            ax2.text(i, v + 2, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Silhouette
        if len(set(labels)) > 1:
            sil = silhouette_score(self.df_pca, labels)
            logger.info(f"Silhouette score for k={n_clusters}: {sil:.4f}")
        else:
            logger.warning("Only one cluster found; silhouette undefined.")

        return labels

    # -----------------------
    # DBSCAN parameter grid 
    # -----------------------
    def dbscan_grid(self, eps_range: np.ndarray = None, min_samples_range: range = None,
                    plot: bool = True):
        if self.df_pca is None:
            self.run_pca()

        if eps_range is None:
            eps_range = np.arange(0.5, 3.01, 0.25)
        if min_samples_range is None:
            min_samples_range = range(3, 11)

        results = []
        for eps in eps_range:
            for ms in min_samples_range:
                db = DBSCAN(eps=eps, min_samples=ms)
                labels = db.fit_predict(self.df_pca)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                sil = -1
                if n_clusters > 1:
                    try:
                        sil = silhouette_score(self.df_pca, labels)
                    except Exception:
                        sil = -1
                results.append({'eps': float(eps), 'min_samples': int(ms),
                                'n_clusters': int(n_clusters), 'n_noise': int(n_noise), 'silhouette': float(sil)})
        res_df = pd.DataFrame(results)
        if plot:
            pivot = res_df.pivot(index='min_samples', columns='eps', values='silhouette')
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, cmap='RdYlBu', center=0)
            plt.title("DBSCAN silhouette heatmap")
            plt.show()

        best = res_df[res_df['silhouette'] > 0].sort_values('silhouette', ascending=False)
        if not best.empty:
            top = best.iloc[0]
            logger.info(f"Best DBSCAN (silhouette>0): eps={top.eps}, min_samples={top.min_samples}, "
                        f"clusters={top.n_clusters}, noise={top.n_noise}, silhouette={top.silhouette:.4f}")
            return top
        else:
            logger.warning("No DBSCAN configuration with silhouette>0 found.")
            return res_df

    def feature_importance(self, target: str = 'SalePrice', n_features: int = 15, cv_folds: int = 5,
                           random_state: int = RANDOM_STATE, show_plot: bool = True):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        numeric = self.df.select_dtypes(include=[np.number]).copy()
        if target not in numeric.columns:
            raise ValueError(f"{target} not in numeric columns.")

        X = numeric.drop(columns=[target, 'Order', 'PID'], errors='ignore')
        y = numeric[target].copy()

        # Impute median
        for c in X.columns:
            if X[c].isnull().any():
                X[c].fillna(X[c].median(), inplace=True)

        # Train RandomForest on log1p(y)
        y_log = np.log1p(y.clip(lower=0))
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        rf.fit(X, y_log)

        # Permutation importance 
        logger.info("Computing permutation importance (this may take time)...")
        perm = permutation_importance(rf, X, y_log, n_repeats=10, random_state=random_state, n_jobs=-1)
        perm_idx = np.argsort(perm.importances_mean)[::-1][:n_features]
        perm_df = pd.DataFrame({
            'feature': X.columns[perm_idx],
            'importance_mean': perm.importances_mean[perm_idx],
            'importance_std': perm.importances_std[perm_idx]
        }).reset_index(drop=True)

        if show_plot:
            plt.figure(figsize=(8, max(3, 0.4 * len(perm_df))))
            sns.barplot(x='importance_mean', y='feature', data=perm_df, orient='h')
            plt.title(f"Top {len(perm_df)} features (permutation importance)")
            plt.xlabel("Mean decrease in score (higher -> more important)")
            plt.tight_layout()
            plt.show()

        rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(n_features)
        logger.info("Top features by model (Gini) importance:")
        logger.info(rf_imp.head(n_features).to_string())

        return perm_df, rf_imp

    def run_full_report(self):
        """
        Conservative end-to-end reporting. Does NOT perform destructive operations.
        """
        logger.info("=== STARTING FULL REPORT ===")
        self.analyze_target(show_plots=True, outlier_method='report')
        self.missing_value_report()
        self.prepare_numeric()
        self.run_pca()
        self.plot_pca_scree()
        top_corr = self.show_correlations()
        optimal_k = self.find_optimal_k()
        self.cluster_and_plot(n_clusters=optimal_k)
        self.dbscan_grid()
        self.feature_importance()
        logger.info("=== REPORT COMPLETE ===")


if __name__ == "__main__":
    viz = AmesVisualizer("AmesHousing.csv")
    viz.load()
    viz.run_full_report()
