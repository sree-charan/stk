"""XGBoost models for short/medium/long term predictions — regression + classification."""
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple


def _get_xgb():
    import xgboost as xgb
    return xgb


class BaseXGBoost:
    """Base XGBoost model with both regressor and classifier."""

    def __init__(self, horizon: str, n_estimators: int = 100, max_depth: int = 4,
                 learning_rate: float = 0.05, subsample: float = 0.8):
        self.horizon = horizon
        self._reg_params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=5, gamma=0.0,
            objective='reg:squarederror', random_state=42, n_jobs=-1,
        )
        self._cls_params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=5, gamma=0.0,
            eval_metric='logloss',
            random_state=42, n_jobs=-1,
        )
        self._model = None  # regressor
        self._classifier = None
        self._trained = False
        self._cls_trained = False

    @property
    def model(self):
        if self._model is None:
            xgb = _get_xgb()
            self._model = xgb.XGBRegressor(**self._reg_params)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def classifier(self):
        if self._classifier is None:
            xgb = _get_xgb()
            self._classifier = xgb.XGBClassifier(**self._cls_params)
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """Train regressor."""
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        self._trained = True
        return self.model.score(X, y)

    def train_classifier(self, X: np.ndarray, y_cls: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """Train classifier (y_cls: 1=up, 0=down)."""
        if len(np.unique(y_cls)) < 2:
            self._cls_trained = False
            return 0.5
        if sample_weight is not None:
            self.classifier.fit(X, y_cls, sample_weight=sample_weight)
        else:
            self.classifier.fit(X, y_cls)
        self._cls_trained = True
        return self.classifier.score(X, y_cls)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns (regression)."""
        if not self._trained:
            return np.zeros(len(X) if X.ndim > 1 else 1)
        return self.model.predict(X.reshape(1, -1) if X.ndim == 1 else X)

    def predict_direction(self, X: np.ndarray) -> Tuple[int, float]:
        """Predict direction using classifier. Returns (direction, probability_of_up)."""
        if not self._cls_trained:
            return 1, 0.5
        x = X.reshape(1, -1) if X.ndim == 1 else X
        prob = self.classifier.predict_proba(x)[0]
        prob_up = float(prob[1]) if len(prob) > 1 else float(prob[0])
        direction = 1 if prob_up > 0.5 else 0
        return direction, prob_up

    def predict_proba(self, X: np.ndarray) -> Tuple[float, float]:
        """Return prediction and confidence. Uses classifier if available."""
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
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self._trained = data['trained']
            self._classifier = data.get('classifier')
            self._cls_trained = data.get('cls_trained', False)
        return True

    @property
    def feature_importance(self) -> Optional[np.ndarray]:
        return self.model.feature_importances_ if self._trained else None


class XGBoostShort(BaseXGBoost):
    def __init__(self):
        super().__init__('1h', n_estimators=30, max_depth=2, learning_rate=0.1, subsample=0.8)
        self._model = None
        self._classifier = None


class XGBoostMedium(BaseXGBoost):
    def __init__(self):
        super().__init__('5d', n_estimators=40, max_depth=3, learning_rate=0.1, subsample=0.8)


class XGBoostLong(BaseXGBoost):
    def __init__(self):
        super().__init__('60d', n_estimators=40, max_depth=3, learning_rate=0.1, subsample=0.8)
