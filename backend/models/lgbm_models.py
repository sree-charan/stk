"""LightGBM models for short/medium/long term predictions — regression + classification."""
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional


class BaseLGBM:
    """Base LightGBM model with both regressor and classifier."""

    def __init__(self, horizon: str, params: dict | None = None):
        import lightgbm as lgb
        self.horizon = horizon
        self._params = params or self._default_params()
        self.model = lgb.LGBMRegressor(**self._params)
        cls_params = dict(self._params)
        cls_params['objective'] = 'binary'
        self._classifier = lgb.LGBMClassifier(**cls_params)
        self._trained = False
        self._cls_trained = False

    def _default_params(self) -> dict:
        return {
            'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.5, 'reg_alpha': 0.1,
            'reg_lambda': 1.0, 'num_leaves': 8, 'min_child_weight': 5.0,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        self.model.fit(X, y, sample_weight=sample_weight)
        self._trained = True
        return self.model.score(X, y)

    def train_classifier(self, X: np.ndarray, y_cls: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        if len(np.unique(y_cls)) < 2:
            self._cls_trained = False
            return 0.5
        self._classifier.fit(X, y_cls, sample_weight=sample_weight)
        self._cls_trained = True
        return self._classifier.score(X, y_cls)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            return np.zeros(len(X) if X.ndim > 1 else 1)
        return self.model.predict(X.reshape(1, -1) if X.ndim == 1 else X)

    def predict_direction(self, X: np.ndarray) -> Tuple[int, float]:
        if not self._cls_trained:
            return 1, 0.5
        x = X.reshape(1, -1) if X.ndim == 1 else X
        prob = self._classifier.predict_proba(x)[0]
        prob_up = float(prob[1]) if len(prob) > 1 else float(prob[0])
        return (1 if prob_up > 0.5 else 0), prob_up

    def predict_proba(self, X: np.ndarray) -> Tuple[float, float]:
        pred = self.predict(X)
        val = pred[0] if isinstance(pred, np.ndarray) else pred
        if self._cls_trained:
            _, prob_up = self.predict_direction(X)
            confidence = max(prob_up, 1 - prob_up)
        else:
            confidence = min(0.95, 0.5 + 0.4 * np.tanh(abs(val) * 100))
        return float(val), confidence

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model, 'trained': self._trained,
                'classifier': self._classifier, 'cls_trained': self._cls_trained,
            }, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self._trained = data['trained']
                self._classifier = data.get('classifier')
                self._cls_trained = data.get('cls_trained', False)
            return True
        except Exception:
            return False

    @property
    def feature_importance(self) -> Optional[np.ndarray]:
        return self.model.feature_importances_ if self._trained else None


_DEFAULT_SHORT = {'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1, 'num_leaves': 4,
                  'subsample': 0.8, 'colsample_bytree': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                  'min_child_weight': 5.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
_DEFAULT_MEDIUM = {'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.1, 'num_leaves': 8,
                   'subsample': 0.8, 'colsample_bytree': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                   'min_child_weight': 5.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
_DEFAULT_LONG = {'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.1, 'num_leaves': 8,
                 'subsample': 0.8, 'colsample_bytree': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                 'min_child_weight': 5.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}


class LGBMShort(BaseLGBM):
    def __init__(self, params=None):
        super().__init__('1h', params or _DEFAULT_SHORT)


class LGBMMedium(BaseLGBM):
    def __init__(self, params=None):
        super().__init__('5d', params or _DEFAULT_MEDIUM)


class LGBMLong(BaseLGBM):
    def __init__(self, params=None):
        super().__init__('60d', params or _DEFAULT_LONG)
