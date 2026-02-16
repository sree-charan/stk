"""Ensemble model combining XGBoost, LightGBM, and LSTM predictions with classification."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .xgboost_models import XGBoostShort, XGBoostMedium, XGBoostLong
from .lgbm_models import LGBMShort, LGBMMedium, LGBMLong
from .lstm_model import LSTMModel
from .explain import compute_shap_explanation, get_conviction_tier, compute_volatility_zscore


class EnsembleModel:
    """Weighted ensemble of all prediction models with classification support."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path('backend/models/saved')
        self.xgb_short = XGBoostShort()
        self.xgb_medium = XGBoostMedium()
        self.xgb_long = XGBoostLong()
        self.lgbm_short = LGBMShort()
        self.lgbm_medium = LGBMMedium()
        self.lgbm_long = LGBMLong()
        self.lstm = LSTMModel()
        self.selected_features: Optional[List[int]] = None  # adaptive feature selection
        self.loaded: bool = False

        self.weights = {
            'short': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
            'medium': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
            'long': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
        }

    def set_weights_from_accuracy(self, xgb_acc: dict, lgbm_acc: dict):
        for h in ('short', 'medium', 'long'):
            xa = xgb_acc.get(h, 0.5)
            la = lgbm_acc.get(h, 0.5)
            total = xa + la
            if total > 0:
                self.weights[h]['xgb'] = xa / total
                self.weights[h]['lgbm'] = la / total

    def _get_models(self, horizon: str):
        if horizon == 'short':
            return self.xgb_short, self.lgbm_short
        elif horizon == 'medium':
            return self.xgb_medium, self.lgbm_medium
        else:
            return self.xgb_long, self.lgbm_long

    def train_all(self, X: np.ndarray, y_short: np.ndarray, y_medium: np.ndarray, y_long: np.ndarray) -> Dict[str, float]:
        scores = {}
        scores['xgb_short'] = self.xgb_short.train(X, y_short)
        scores['xgb_medium'] = self.xgb_medium.train(X, y_medium)
        scores['xgb_long'] = self.xgb_long.train(X, y_long)
        scores['lgbm_short'] = self.lgbm_short.train(X, y_short)
        scores['lgbm_medium'] = self.lgbm_medium.train(X, y_medium)
        scores['lgbm_long'] = self.lgbm_long.train(X, y_long)
        scores['lstm'] = self.lstm.train(X, y_short)
        return scores

    def train_classifiers(self, X: np.ndarray, y_short: np.ndarray, y_medium: np.ndarray,
                          y_long: np.ndarray, sample_weight: np.ndarray = None):
        """Train classifiers for direction prediction."""
        y_short_cls = (y_short > 0).astype(int)
        y_medium_cls = (y_medium > 0).astype(int)
        y_long_cls = (y_long > 0).astype(int)
        self.xgb_short.train_classifier(X, y_short_cls, sample_weight)
        self.xgb_medium.train_classifier(X, y_medium_cls, sample_weight)
        self.xgb_long.train_classifier(X, y_long_cls, sample_weight)
        self.lgbm_short.train_classifier(X, y_short_cls, sample_weight)
        self.lgbm_medium.train_classifier(X, y_medium_cls, sample_weight)
        self.lgbm_long.train_classifier(X, y_long_cls, sample_weight)

    def predict(self, X: np.ndarray, horizon: str = 'short') -> Tuple[float, float, Dict]:
        xgb_m, lgbm_m = self._get_models(horizon)
        w = self.weights[horizon]

        xgb_pred, xgb_conf = xgb_m.predict_proba(X)
        lgbm_pred, lgbm_conf = lgbm_m.predict_proba(X)
        lstm_pred, lstm_conf = self.lstm.predict_proba(X)

        ensemble_pred = w['xgb'] * xgb_pred + w['lgbm'] * lgbm_pred + w['lstm'] * lstm_pred
        ensemble_conf = w['xgb'] * xgb_conf + w['lgbm'] * lgbm_conf + w['lstm'] * lstm_conf

        breakdown = {
            'xgb': {'prediction': xgb_pred, 'confidence': xgb_conf, 'weight': w['xgb']},
            'lgbm': {'prediction': lgbm_pred, 'confidence': lgbm_conf, 'weight': w['lgbm']},
            'lstm': {'prediction': lstm_pred, 'confidence': lstm_conf, 'weight': w['lstm']},
        }
        return ensemble_pred, ensemble_conf, breakdown

    def predict_direction_ensemble(self, X: np.ndarray, horizon: str = 'short') -> Tuple[str, float, float]:
        """Predict direction using classifier ensemble. Returns (direction, prob, expected_return)."""
        xgb_m, lgbm_m = self._get_models(horizon)
        w = self.weights[horizon]

        xgb_dir, xgb_prob = xgb_m.predict_direction(X)
        lgbm_dir, lgbm_prob = lgbm_m.predict_direction(X)

        # Weighted average probability of UP
        prob_up = w['xgb'] * xgb_prob + w['lgbm'] * lgbm_prob
        direction = 'bullish' if prob_up > 0.5 else 'bearish'
        confidence = max(prob_up, 1 - prob_up)

        # Get expected return magnitude from regressors
        xgb_pred = xgb_m.predict(X)
        lgbm_pred = lgbm_m.predict(X)
        expected_return = float(w['xgb'] * xgb_pred[0] + w['lgbm'] * lgbm_pred[0])

        # Fix direction/magnitude contradiction: sign must match classifier direction
        if direction == 'bearish' and expected_return > 0:
            expected_return = -abs(expected_return)
        elif direction == 'bullish' and expected_return < 0:
            expected_return = abs(expected_return)

        return direction, confidence, expected_return

    def predict_all_horizons(self, X: np.ndarray, feature_names: list = None,
                             historical_std: float = None) -> Dict[str, Dict]:
        """Get predictions for all time horizons with classification, SHAP, and conviction."""
        results = {}
        for horizon in ['short', 'medium', 'long']:
            direction, confidence, expected_return = self.predict_direction_ensemble(X, horizon)
            tier, tier_label, tier_emoji = get_conviction_tier(confidence)

            result = {
                'prediction': expected_return,
                'confidence': confidence,
                'direction': direction,
                'conviction_tier': tier,
                'conviction_label': tier_label,
                'prob_up': confidence if direction == 'bullish' else 1 - confidence,
            }

            # Volatility-adjusted prediction
            if historical_std is not None and historical_std > 0:
                z, z_desc = compute_volatility_zscore(expected_return, historical_std)
                result['vol_zscore'] = z
                result['vol_zscore_desc'] = z_desc

            # SHAP explanation
            if feature_names:
                xgb_m, _ = self._get_models(horizon)
                model_for_shap = xgb_m._classifier if xgb_m._cls_trained else (xgb_m._model if xgb_m._trained else None)
                if model_for_shap is not None:
                    result['shap'] = compute_shap_explanation(model_for_shap, X, feature_names)

            # Breakdown
            _, _, breakdown = self.predict(X, horizon)
            result['breakdown'] = breakdown

            results[horizon] = result
        return results

    def save(self):
        self.xgb_short.save(self.model_dir / 'xgb_short.pkl')
        self.xgb_medium.save(self.model_dir / 'xgb_medium.pkl')
        self.xgb_long.save(self.model_dir / 'xgb_long.pkl')
        self.lgbm_short.save(self.model_dir / 'lgbm_short.pkl')
        self.lgbm_medium.save(self.model_dir / 'lgbm_medium.pkl')
        self.lgbm_long.save(self.model_dir / 'lgbm_long.pkl')
        self.lstm.save(self.model_dir / 'lstm.pkl')
        # Save selected features
        if self.selected_features is not None:
            import json
            with open(self.model_dir / 'selected_features.json', 'w') as f:
                json.dump(self.selected_features, f)

    def load(self) -> bool:
        base = all([
            self.xgb_short.load(self.model_dir / 'xgb_short.pkl'),
            self.xgb_medium.load(self.model_dir / 'xgb_medium.pkl'),
            self.xgb_long.load(self.model_dir / 'xgb_long.pkl'),
            self.lstm.load(self.model_dir / 'lstm.pkl'),
        ])
        lgbm_loaded = all([
            self.lgbm_short.load(self.model_dir / 'lgbm_short.pkl'),
            self.lgbm_medium.load(self.model_dir / 'lgbm_medium.pkl'),
            self.lgbm_long.load(self.model_dir / 'lgbm_long.pkl'),
        ])
        if not lgbm_loaded:
            for h in self.weights:
                self.weights[h] = {'xgb': 1.0, 'lgbm': 0.0, 'lstm': 0.0}
        # Load selected features
        sf_path = self.model_dir / 'selected_features.json'
        if sf_path.exists():
            import json
            with open(sf_path) as f:
                self.selected_features = json.load(f)
        self.loaded = base
        return base
