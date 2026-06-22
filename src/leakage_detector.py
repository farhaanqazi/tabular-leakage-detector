import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

class DataLeakageDetector:
    """
    Standalone leakage-detection module that scores tabular features for suspicious label-predictiveness.
    
    Scope:
    - Reliably catches univariate within-class degeneracy leaks (constant or near-constant values within a specific class).
    - Surfaces suspiciously strong features (high univariate AUC/KS) for human review.
    
    Does NOT detect:
    - Temporal leakage
    - Train/test contamination
    - Multivariate leakage (where no single feature looks suspicious)
    """
    
    def __init__(self, degeneracy_std_threshold=1e-5):
        """
        Args:
            degeneracy_std_threshold (float): Threshold for within-class standard deviation 
                                              below which a feature is flagged as degenerate.
        """
        self.degeneracy_std_threshold = degeneracy_std_threshold
        self.feature_scores_ = None
        
    def fit(self, X, y):
        """
        Scores all features in X against binary label y.
        
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Binary labels (0 or 1).
            
        Returns:
            self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            
        if isinstance(y, pd.Series):
            y = y.values
            
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Leakage detector currently only supports binary classification.")
            
        class_0, class_1 = classes[0], classes[1]
        mask_0 = (y == class_0)
        mask_1 = (y == class_1)
        
        scores = []
        
        for col in X.columns:
            feat_vals = X[col].values
            
            # Handle NaN if any
            valid_idx = ~pd.isna(feat_vals)
            v_feat = feat_vals[valid_idx]
            v_y = y[valid_idx]
            
            if len(np.unique(v_y)) < 2:
                continue # Can't score if only one class present
                
            # 1. Univariate AUC (direction-agnostic)
            try:
                auc_raw = roc_auc_score(v_y, v_feat)
                auc_score = max(auc_raw, 1.0 - auc_raw)
            except ValueError:
                auc_score = 0.5
                
            # 2. KS Statistic
            vals_0 = v_feat[v_y == class_0]
            vals_1 = v_feat[v_y == class_1]
            
            if len(vals_0) > 0 and len(vals_1) > 0:
                ks_stat, _ = ks_2samp(vals_0, vals_1)
            else:
                ks_stat = 0.0
                
            # 3. Degeneracy Check
            std_0 = np.std(vals_0) if len(vals_0) > 0 else np.inf
            std_1 = np.std(vals_1) if len(vals_1) > 0 else np.inf
            
            is_degenerate_0 = std_0 <= self.degeneracy_std_threshold
            is_degenerate_1 = std_1 <= self.degeneracy_std_threshold
            
            # Degeneracy flag indicates one or both classes collapsed to a near-constant value
            degenerate_flag = bool(is_degenerate_0 or is_degenerate_1)
            
            # Avoid flagging standard binary categorical features as degenerate
            unique_vals = np.unique(v_feat)
            if len(unique_vals) <= 2:
                degenerate_flag = False
            
            scores.append({
                'feature': col,
                'auc': auc_score,
                'ks_stat': ks_stat,
                'degenerate_flag': degenerate_flag,
                'std_class_0': std_0,
                'std_class_1': std_1
            })
            
        self.feature_scores_ = pd.DataFrame(scores)
        # Rank primarily by KS stat and AUC (as an ensemble 'predictiveness' score)
        self.feature_scores_['predictiveness_score'] = (self.feature_scores_['auc'] + self.feature_scores_['ks_stat']) / 2.0
        
        # Sort such that degenerate flags and highest predictiveness come first
        self.feature_scores_ = self.feature_scores_.sort_values(
            by=['degenerate_flag', 'predictiveness_score'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        return self

    def get_scores(self):
        """Returns the ranked DataFrame of feature scores."""
        if self.feature_scores_ is None:
            raise ValueError("Detector has not been fitted yet.")
        return self.feature_scores_

    def predict_leaks(self, threshold_score=0.9):
        """
        Given a threshold, returns flags for each feature. 
        Note: The degeneracy flag is a hard flag independent of the continuous threshold.
        
        Args:
            threshold_score: Float between 0.5 and 1.0 on the predictiveness score.
        """
        if self.feature_scores_ is None:
            raise ValueError("Detector has not been fitted yet.")
            
        df = self.feature_scores_.copy()
        df['is_leak_pred'] = df['degenerate_flag'] | (df['predictiveness_score'] >= threshold_score)
        return df[['feature', 'is_leak_pred', 'degenerate_flag', 'predictiveness_score', 'auc', 'ks_stat']]
