"""Analysis engine - bridges CLI to backend."""
import sys
import json
import time
from collections.abc import Callable
from pathlib import Path
from datetime import datetime

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from backend.data.real_providers import generate_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data
from backend.features.feature_store import FeatureStore
from backend.models.ensemble import EnsembleModel
from backend.utils.invalidation import calculate_invalidation
from cli.errors import InvalidTickerError, NetworkError, RateLimitError

_store = FeatureStore()
_ensemble = EnsembleModel()
_loaded = False
_sentiment_warned = False

# Model version: bump when training pipeline changes materially
MODEL_VERSION = "2.0.0"


def _get_sentiment_safe(ticker: str):
    """Fetch sentiment, showing a single clean warning on failure."""
    global _sentiment_warned
    try:
        return get_sentiment(ticker)
    except Exception:
        if not _sentiment_warned:
            import logging
            logger = logging.getLogger('cli.engine')
            logger.debug("Sentiment API failed for %s", ticker)
            # Show clean one-liner in normal mode (not a WARNING log, just user info)
            print("⚠ Sentiment data unavailable (API limit reached)", file=sys.stderr)
            _sentiment_warned = True
        return None

# --- Cached yfinance .info ---
_info_cache: dict[str, tuple[float, dict]] = {}
_INFO_TTL = 60  # seconds


def _get_yf_info(ticker: str) -> dict:
    """Get yfinance .info with per-request caching."""
    ticker = ticker.upper()
    now = time.time()
    if ticker in _info_cache:
        ts, info = _info_cache[ticker]
        if now - ts < _INFO_TTL:
            return info
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    _info_cache[ticker] = (now, info)
    return info


def _get_current_price(ticker: str, fallback_close: float | None = None) -> float:
    """Best available price: postMarket > preMarket > regularMarket > last close."""
    info = _get_yf_info(ticker)
    for key in ('postMarketPrice', 'preMarketPrice', 'regularMarketPrice'):
        val = info.get(key)
        if val is not None and val > 0:
            return float(val)
    if fallback_close is not None and fallback_close > 0:
        return fallback_close
    # Last resort: fetch close
    df = generate_ohlcv(ticker.upper(), "daily", 5)
    return float(df['close'].iloc[-1])


# --- Per-ticker model management ---
_MODELS_DIR = Path.home() / '.stk' / 'models'
_ticker_ensembles: dict[str, EnsembleModel] = {}
_ticker_meta: dict[str, dict] = {}
_hourly_models: dict[str, dict] = {}


def _ticker_model_dir(ticker: str) -> Path:
    return _MODELS_DIR / ticker.upper()


def _save_feature_stability(ticker: str, stability: dict) -> None:
    """Save feature stability data for use in next retrain. Keeps only top 200 entries."""
    d = _ticker_model_dir(ticker.upper())
    d.mkdir(parents=True, exist_ok=True)
    # Limit to top 200 by frequency to prevent unbounded growth
    if len(stability) > 200:
        sorted_items = sorted(stability.items(), key=lambda x: x[1], reverse=True)[:200]
        stability = dict(sorted_items)
    (d / 'feature_stability.json').write_text(json.dumps(stability))


def _load_feature_stability(ticker: str) -> dict | None:
    """Load feature stability from previous retrain."""
    p = _ticker_model_dir(ticker.upper()) / 'feature_stability.json'
    if p.exists():
        return json.loads(p.read_text())
    return None

def _load_hourly_model(ticker: str) -> dict | None:
    """Load saved hourly short-term model for a ticker."""
    ticker = ticker.upper()
    if ticker in _hourly_models:
        return _hourly_models[ticker]
    path = _ticker_model_dir(ticker) / 'hourly_short.pkl'
    if path.exists():
        import pickle
        with open(path, 'rb') as fp:
            model = pickle.load(fp)
        _hourly_models[ticker] = model
        return model
    return None


def _predict_hourly_short(ticker: str) -> tuple[float, float] | None:
    """Use hourly model for short-term prediction. Returns (prediction, confidence) or None."""
    model = _load_hourly_model(ticker)
    if model is None:
        return None
    try:
        from backend.features.hourly import fetch_hourly_data, compute_hourly_features
        hourly_df = fetch_hourly_data(ticker)
        if hourly_df is None or len(hourly_df) < 20:
            return None
        feats = compute_hourly_features(hourly_df)
        row = feats.iloc[-1:].values
        xgb_pred = model['xgb'].predict(row)[0]
        lgbm_pred = model['lgbm'].predict(row)[0]
        pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
        conf = min(0.95, 0.5 + 0.4 * np.tanh(abs(pred) * 100))
        return float(pred), float(conf)
    except Exception:
        return None


def _load_ticker_meta(ticker: str) -> dict | None:
    ticker = ticker.upper()
    if ticker in _ticker_meta:
        return _ticker_meta[ticker]
    meta_path = _ticker_model_dir(ticker) / 'meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        _ticker_meta[ticker] = meta
        return meta
    return None


def _save_ticker_meta(ticker: str, meta: dict) -> None:
    ticker = ticker.upper()
    d = _ticker_model_dir(ticker)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'meta.json').write_text(json.dumps(meta, indent=2))
    _ticker_meta[ticker] = meta


def _get_ticker_ensemble(ticker: str) -> EnsembleModel:
    """Get or load per-ticker ensemble. Returns fallback if no per-ticker model."""
    ticker = ticker.upper()
    if ticker in _ticker_ensembles:
        return _ticker_ensembles[ticker]
    d = _ticker_model_dir(ticker)
    try:
        ens = EnsembleModel(model_dir=d)
        if ens.load():
            _ticker_ensembles[ticker] = ens
            return ens
    except Exception:
        pass  # corrupted model files — fall through to fallback
    return _ensemble  # fallback


def _fetch_spy_sector_data(ticker: str, days: int):
    """Fetch SPY and sector ETF data for sector-relative features."""
    from backend.features.sector import get_sector_etf
    spy_df = None
    sector_df = None
    try:
        spy_df = generate_ohlcv('SPY', 'daily', days)
    except Exception:
        pass
    sector_etf = get_sector_etf(ticker)
    if sector_etf != 'SPY':
        try:
            sector_df = generate_ohlcv(sector_etf, 'daily', days)
        except Exception:
            pass
    return spy_df, sector_df


def _adaptive_feature_selection(X, y_short, y_medium, y_long, feature_names, verbose=False,
                                max_features=150, ticker=None):
    """Drop features with low importance. Returns selected indices and names.
    Uses only the first 80% of data to avoid leaking future info.
    Horizon-weighted: medium/long get more weight since short-term is noisy.
    Optionally boosts features that were stable across WF windows in previous retrain."""
    import xgboost as xgb
    n_sel = int(len(X) * 0.8)
    X_sel = X[:n_sel]
    y_cls_s = (y_short[:n_sel] > 0).astype(int)
    y_cls_m = (y_medium[:n_sel] > 0).astype(int)
    y_cls_l = (y_long[:n_sel] > 0).astype(int)
    importances = np.zeros(X.shape[1])
    horizon_weights = [(y_cls_s, 0.2), (y_cls_m, 0.4), (y_cls_l, 0.4)]
    n_valid = 0
    for y_cls, w in horizon_weights:
        if len(np.unique(y_cls)) < 2:
            continue
        # Use multiple seeds for more robust importance estimation
        for seed in (42, 123):
            m = xgb.XGBClassifier(n_estimators=40, max_depth=3, random_state=seed, n_jobs=-1,
                                   eval_metric='logloss', subsample=0.8, colsample_bytree=0.5,
                                   reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5)
            m.fit(X_sel, y_cls)
            importances += m.feature_importances_ * w * 0.5
        n_valid += 1
    if n_valid == 0:
        selected = list(range(min(X.shape[1], max_features)))
        sel_names = [feature_names[i] for i in selected] if feature_names else []
        if verbose:
            print(f"  Using {len(selected)}/{X.shape[1]} features (single-class target)")
        return selected, sel_names

    # Boost features that were stable in previous WF (if available)
    stability_boost = 0
    if ticker:
        stability = _load_feature_stability(ticker)
        if stability and len(stability) > 0:
            for idx_str, freq in stability.items():
                idx = int(idx_str)
                if idx < len(importances):
                    importances[idx] *= (1.0 + 0.3 * freq)
            stability_boost = len(stability)

    n_keep = max(50, min(max_features, X.shape[1]))
    top_idx = np.argsort(importances)[-n_keep:]
    selected = sorted(top_idx.tolist())
    sel_names = [feature_names[i] for i in selected] if feature_names else []
    if verbose:
        boost_msg = f", stability boost from {stability_boost} features" if stability_boost else ""
        print(f"  Using {len(selected)}/{X.shape[1]} features (dropped {X.shape[1] - len(selected)} noise features{boost_msg})")
    return selected, sel_names


def _calibrate_probabilities(wf_results: dict, X, y_short, y_medium, y_long, train_window, test_window):
    """Build calibrators from WF out-of-sample predictions.
    Uses isotonic regression when enough data (>=50), Platt scaling as fallback (>=20).
    Returns dict of {horizon: calibrator} or empty dict if insufficient data."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return {}

    oos_data = wf_results.get('oos_data', {})
    if not oos_data:
        return {}

    calibrators = {}
    for h in ('short', 'medium', 'long'):
        probs = np.array(oos_data.get(h, {}).get('probs', []))
        actuals = np.array(oos_data.get(h, {}).get('actuals', []))
        if len(probs) >= 50:
            ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
            ir.fit(probs, actuals)
            calibrators[h] = ir
        elif len(probs) >= 20:
            # Platt scaling fallback for smaller datasets
            from backend.models.explain import calibrate_platt
            platt = calibrate_platt(probs, actuals)
            if platt is not None:
                calibrators[h] = platt
    return calibrators


def _train_classifiers_with_bagging(ens, X_sel, y_short, y_medium, y_long, weights):
    """Train classifiers with adaptive dead zone filtering and bagged ensembles."""
    import xgboost as xgb_lib
    bagged_classifiers = {}
    for name, xgb_m, lgbm_m, y in [('short', ens.xgb_short, ens.lgbm_short, y_short),
                                     ('medium', ens.xgb_medium, ens.lgbm_medium, y_medium),
                                     ('long', ens.xgb_long, ens.lgbm_long, y_long)]:
        abs_y = np.abs(y)
        dz = float(np.percentile(abs_y[abs_y > 0], 10)) if np.any(abs_y > 0) else 0.001
        mask = abs_y > dz
        if np.sum(mask) < 50:
            mask = np.ones(len(y), dtype=bool)
        y_cls = (y[mask] > 0).astype(int)
        w_m = weights[mask] if weights is not None else None
        if len(np.unique(y_cls)) < 2:
            continue
        xgb_m.train_classifier(X_sel[mask], y_cls, w_m)
        lgbm_m.train_classifier(X_sel[mask], y_cls, w_m)
        bag_models = []
        for seed in (123, 7):
            p = xgb_m._cls_params.copy()
            p['random_state'] = seed
            bx = xgb_lib.XGBClassifier(**p)
            bx.fit(X_sel[mask], y_cls, sample_weight=w_m)
            bag_models.append(bx)
        bagged_classifiers[name] = bag_models
    return bagged_classifiers


def _compute_val_accuracy(ens, X_sel, y_short, y_medium, y_long, n, test_window):
    """Compute validation accuracy on last test_window for ensemble weighting."""
    split = max(0, n - test_window)
    X_val = X_sel[split:]
    xgb_acc, lgbm_acc, accuracies = {}, {}, {}
    for name, xgb_m, lgbm_m, y in [('short', ens.xgb_short, ens.lgbm_short, y_short),
                                     ('medium', ens.xgb_medium, ens.lgbm_medium, y_medium),
                                     ('long', ens.xgb_long, ens.lgbm_long, y_long)]:
        actual = y[split:]
        actual_cls = (actual > 0).astype(int)
        xgb_dir_preds = np.array([xgb_m.predict_direction(X_val[i])[0] for i in range(len(X_val))])
        lgbm_dir_preds = np.array([lgbm_m.predict_direction(X_val[i])[0] for i in range(len(X_val))])
        xgb_a = float(np.sum(xgb_dir_preds == actual_cls) / len(actual)) if len(actual) > 0 else 0.5
        lgbm_a = float(np.sum(lgbm_dir_preds == actual_cls) / len(actual)) if len(actual) > 0 else 0.5
        xgb_acc[name] = round(xgb_a, 4)
        lgbm_acc[name] = round(lgbm_a, 4)
        total = xgb_a + lgbm_a
        w_xgb = xgb_a / total if total > 0 else 0.5
        w_lgbm = lgbm_a / total if total > 0 else 0.5
        xgb_probs = np.array([xgb_m.predict_direction(X_val[i])[1] for i in range(len(X_val))])
        lgbm_probs = np.array([lgbm_m.predict_direction(X_val[i])[1] for i in range(len(X_val))])
        ens_probs = w_xgb * xgb_probs + w_lgbm * lgbm_probs
        ens_preds = (ens_probs > 0.5).astype(int)
        ens_correct = np.sum(ens_preds == actual_cls)
        accuracies[name] = round(float(ens_correct / len(actual)) if len(actual) > 0 else 0.5, 4)
    return xgb_acc, lgbm_acc, accuracies


def _train_mtf_models(X_sel, y_short, y_medium, y_long, n):
    """Train multi-timeframe voting classifiers with different lookback windows."""
    import xgboost as xgb
    mtf_models = {}
    mtf_hp = {
        'short': {'n_est': 30, 'depth': 2, 'alpha': 0.1, 'lam': 1.0, 'mcw': 5, 'colsample': 0.5, 'gamma': 0.0},
        'medium': {'n_est': 40, 'depth': 3, 'alpha': 0.1, 'lam': 1.0, 'mcw': 5, 'colsample': 0.5, 'gamma': 0.0},
        'long': {'n_est': 40, 'depth': 3, 'alpha': 0.1, 'lam': 1.0, 'mcw': 5, 'colsample': 0.5, 'gamma': 0.0},
    }
    for win in [30, 90, 252]:
        if n < win + 10:
            continue
        X_win = X_sel[-win:]
        mtf_models[win] = {}
        for name, y in [('short', y_short), ('medium', y_medium), ('long', y_long)]:
            y_cls = (y[-win:] > 0).astype(int)
            if len(np.unique(y_cls)) < 2:
                continue
            p = mtf_hp[name]
            xm = xgb.XGBClassifier(n_estimators=p['n_est'], max_depth=p['depth'], eval_metric='logloss',
                                    reg_alpha=p['alpha'], reg_lambda=p['lam'], min_child_weight=p['mcw'],
                                    subsample=0.5, colsample_bytree=p['colsample'], gamma=p['gamma'],
                                    random_state=42, n_jobs=-1)
            xm.fit(X_win, y_cls)
            mtf_models[win][name] = xm
    return mtf_models


def _save_training_artifacts(d, bagged_classifiers, calibrators, wf_results, X_sel):
    """Save pickle artifacts: bagged classifiers, calibrators, residuals, feature stats."""
    import pickle
    d.mkdir(parents=True, exist_ok=True)
    for name, data in [('bagged_classifiers.pkl', bagged_classifiers),
                       ('calibrators.pkl', calibrators),
                       ('wf_residuals.pkl', wf_results.get('wf_residuals', {}))]:
        if data:
            with open(d / name, 'wb') as fp:
                pickle.dump(data, fp)
    with open(d / 'feature_stats.pkl', 'wb') as fp:
        pickle.dump({'means': np.nanmean(X_sel, axis=0), 'stds': np.nanstd(X_sel, axis=0)}, fp)


def _train_hourly_model(ticker, d, verbose):
    """Train hourly short-term model if data available. Returns accuracy or None."""
    try:
        from backend.features.hourly import fetch_hourly_data, compute_hourly_features
        import xgboost as xgb_lib
        import lightgbm as lgb_lib
        hourly_df = fetch_hourly_data(ticker)
        if hourly_df is None or len(hourly_df) < 100:
            return None
        h_feats = compute_hourly_features(hourly_df)
        h_close = hourly_df['close'].values.astype(float)
        h_n = len(h_close)
        h_y = np.zeros(h_n)
        h_y[:-1] = (h_close[1:] - h_close[:-1]) / np.maximum(h_close[:-1], 1e-8)
        h_X = h_feats.values

        h_train_w, h_test_w = 168, 42
        h_wf_accs = []
        start = 0
        while start + h_train_w + h_test_w <= h_n:
            te = start + h_train_w
            hx = xgb_lib.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)
            hl = lgb_lib.LGBMRegressor(n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
            hx.fit(h_X[start:te], h_y[start:te])
            hl.fit(h_X[start:te], h_y[start:te])
            hp = 0.5 * hx.predict(h_X[te:te + h_test_w]) + 0.5 * hl.predict(h_X[te:te + h_test_w])
            ha = h_y[te:te + h_test_w]
            h_wf_accs.append(float(np.sum((hp > 0) == (ha > 0)) / len(ha)) if len(ha) > 0 else 0.5)
            start += h_test_w

        h_xgb = xgb_lib.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
        h_lgb = lgb_lib.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        h_xgb.fit(h_X, h_y)
        h_lgb.fit(h_X, h_y)

        hourly_acc = round(float(np.mean(h_wf_accs)), 4) if h_wf_accs else 0.5
        import pickle
        hourly_path = d / 'hourly_short.pkl'
        hourly_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hourly_path, 'wb') as fp:
            pickle.dump({'xgb': h_xgb, 'lgbm': h_lgb, 'feature_count': h_X.shape[1]}, fp)
        if verbose:
            print(f"  Hourly short-term model: {hourly_acc*100:.1f}% accuracy ({h_n} bars, {len(h_wf_accs)} windows)")
        return hourly_acc
    except Exception as e:
        if verbose:
            print(f"  Hourly data unavailable: {e}")
        return None


def _build_training_meta(ticker, n, history_days, feature_names, selected_features, sel_names,
                         accuracies, xgb_acc, lgbm_acc, wf_results, ens, calibrators,
                         regime, y_short, hourly_acc, best_xgb_params, best_lgbm_params):
    """Build the meta dict for a trained model."""
    from backend.models.explain import compute_calibration_curve, compute_model_health, save_health_trend, save_feature_changelog
    meta = {
        'model_version': MODEL_VERSION,
        'trained_at': datetime.now().isoformat(),
        'samples': n,
        'history_days': history_days,
        'feature_count': len(feature_names),
        'selected_feature_count': len(selected_features),
        'include_sequence': True,
        'accuracy': accuracies,
        'xgb_accuracy': xgb_acc,
        'lgbm_accuracy': lgbm_acc,
        'walk_forward': {k: v for k, v in wf_results.items() if k not in ('oos_data', 'feature_stability', 'wf_residuals')},
        'ensemble_weights': ens.weights,
        'selected_features': selected_features,
        'feature_names': sel_names,
        'historical_daily_std': float(np.std(y_short[y_short != 0])) if np.any(y_short != 0) else 0.01,
        'regime': {k: v for k, v in regime.items() if isinstance(v, str)},
        'has_calibrators': bool(calibrators),
    }
    ens_div = wf_results.get('ensemble_diversity', {})
    if ens_div:
        meta['ensemble_diversity'] = ens_div

    oos = wf_results.get('oos_data', {})
    brier_scores = {}
    for h in ('short', 'medium', 'long'):
        probs = np.array(oos.get(h, {}).get('probs', []))
        actuals = np.array(oos.get(h, {}).get('actuals', []))
        if len(probs) >= 20:
            brier_scores[h] = {'raw': round(float(np.mean((probs - actuals) ** 2)), 4)}
            cal = calibrators.get(h)
            if cal:
                cal_probs = cal.predict_proba(probs.reshape(-1, 1))[:, 1] if hasattr(cal, 'predict_proba') else cal.predict(probs)
                brier_scores[h]['calibrated'] = round(float(np.mean((cal_probs - actuals) ** 2)), 4)
    if brier_scores:
        meta['brier_scores'] = brier_scores

    cal_curves = {}
    for h in ('short', 'medium', 'long'):
        curve = compute_calibration_curve(
            np.array(oos.get(h, {}).get('probs', [])),
            np.array(oos.get(h, {}).get('actuals', [])))
        if curve:
            cal_curves[h] = curve
    if cal_curves:
        meta['calibration_curves'] = cal_curves

    health = compute_model_health(wf_results.get('average', {}), brier_scores, bool(calibrators))
    meta['model_health'] = health
    save_health_trend(_ticker_model_dir(ticker), health)
    save_feature_changelog(_ticker_model_dir(ticker), sel_names[:20])

    if hourly_acc is not None:
        meta['hourly_accuracy'] = hourly_acc
    if best_xgb_params:
        meta['best_xgb_params'] = best_xgb_params
    if best_lgbm_params:
        meta['best_lgbm_params'] = best_lgbm_params
    return meta


def _print_train_summary(ticker, accuracies, wf_results, selected_features,
                         feature_names, sel_names, meta, prev_meta):
    """Print verbose training summary."""
    from backend.models.explain import (format_model_health, format_health_trend,
                                        format_feature_changelog, _readable_name)
    print(f"Model trained: {accuracies['short']*100:.1f}% short, "
          f"{accuracies['medium']*100:.1f}% medium, {accuracies['long']*100:.1f}% long")

    wf_avg = wf_results.get('average', {})
    wf_std = wf_results.get('std', {})
    wf_ranges = wf_results.get('ranges', {})
    wf_trend = wf_results.get('trend', {})
    if wf_avg:
        print("\n  Walk-forward summary:")
        for h in ('short', 'medium', 'long'):
            if h in wf_avg:
                lo, hi = wf_ranges.get(h, (0, 0))
                t = wf_trend.get(h, '')
                t_icon = '📈' if t == 'improving' else ('📉' if t == 'degrading' else '➡️')
                print(f"    {h.capitalize()}: {wf_avg[h]*100:.1f}% ± {wf_std.get(h, 0)*100:.1f}% "
                      f"(range: {lo*100:.0f}-{hi*100:.0f}%) {t_icon}")
    print(f"  Feature selection: {len(selected_features)}/{len(feature_names)} features")

    fs_data = wf_results.get('feature_stability', {})
    if fs_data:
        stable_count = sum(1 for v in fs_data.values() if v >= 0.5)
        print(f"  Feature stability: {stable_count} features appear in top-20 across ≥50% of WF windows")
        sorted_feats = sorted(fs_data.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_feats:
            def _idx_to_name(idx):
                i = int(idx)
                return sel_names[i] if i < len(sel_names) else str(idx)
            stable_str = ', '.join(f"{_readable_name(_idx_to_name(f))}({v:.0%})" for f, v in sorted_feats)
            print(f"    Most stable: {stable_str}")

    ens_div = wf_results.get('ensemble_diversity', {})
    for h, d in ens_div.items():
        print(f"  Ensemble diversity ({h}): {d['diversity']*100:.1f}% disagreement ({d['description']})")

    health = meta.get('model_health', {})
    print(f"  {format_model_health(health)}")
    trend_text = format_health_trend(_ticker_model_dir(ticker))
    if trend_text:
        print(f"  {trend_text}")
    changelog = format_feature_changelog(_ticker_model_dir(ticker))
    if changelog:
        print(f"  {changelog}")

    if prev_meta:
        _print_changes_vs_previous(prev_meta, meta, wf_avg, selected_features)


def _print_changes_vs_previous(prev_meta, meta, wf_avg, selected_features):
    """Print what changed compared to previous model."""
    prev_wf = prev_meta.get('walk_forward', {}).get('average', {})
    changes = []
    for h in ('short', 'medium', 'long'):
        old_v, new_v = prev_wf.get(h), wf_avg.get(h)
        if old_v is not None and new_v is not None:
            diff = (new_v - old_v) * 100
            if abs(diff) >= 1.0:
                icon = '📈' if diff > 0 else '📉'
                changes.append(f"{h}: {old_v*100:.1f}→{new_v*100:.1f}% ({diff:+.1f}%) {icon}")
    prev_sel = prev_meta.get('selected_feature_count')
    if prev_sel and prev_sel != len(selected_features):
        changes.append(f"features: {prev_sel}→{len(selected_features)}")
    for h in ('short', 'medium', 'long'):
        ob = prev_meta.get('brier_scores', {}).get(h, {})
        nb = meta.get('brier_scores', {}).get(h, {})
        old_val = ob.get('calibrated', ob.get('raw')) if isinstance(ob, dict) else ob
        new_val = nb.get('calibrated', nb.get('raw')) if isinstance(nb, dict) else nb
        if old_val is not None and new_val is not None and abs(new_val - old_val) >= 0.01:
            icon = '📈' if new_val < old_val else '📉'
            changes.append(f"brier_{h}: {old_val:.3f}→{new_val:.3f} {icon}")
    prev_health = prev_meta.get('model_health', {}).get('grade', '')
    new_health = meta.get('model_health', {}).get('grade', '')
    if prev_health and new_health and prev_health != new_health:
        changes.append(f"health: {prev_health}→{new_health}")
    if changes:
        print(f"  Changes vs previous: {', '.join(changes)}")


def train_ticker_model(ticker: str, verbose: bool = True, tune: bool = False) -> dict:
    """Train per-ticker model with classification, walk-forward validation, and adaptive feature selection."""
    ticker = ticker.upper()
    prev_meta = _load_ticker_meta(ticker)  # for "what changed" comparison
    if verbose:
        print(f"Training model for {ticker}...")

    # Fetch max history
    import yfinance as yf
    import io
    import contextlib
    t = yf.Ticker(ticker)
    f = io.StringIO()
    with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        hist = t.history(period="10y", interval="1d")
    if hist.empty:
        raise ValueError(f"No data for {ticker}")
    if len(hist) < 50:
        raise ValueError(f"Insufficient history for {ticker}: {len(hist)} days (need 50+)")

    df = hist.reset_index()
    col_map = {"Date": "date", "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["symbol"] = ticker
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    history_days = len(df)
    spot = float(df['close'].iloc[-1])
    opts = get_options_chain(ticker, spot)
    fund = get_fundamentals(ticker)
    sent = _get_sentiment_safe(ticker)
    macro = get_macro_data()

    # Fetch SPY/sector data for sector-relative features
    spy_df, sector_df = _fetch_spy_sector_data(ticker, history_days)

    feats = _store.compute_all_features(df, opts, fund, sent, macro, spy_df=spy_df, sector_df=sector_df,
                                         include_sequence=True)
    feature_names = list(feats.columns)

    if verbose:
        print(f"  Features: {len(feature_names)} ({history_days} days of history)")

    from backend.features.regime import compute_regime_features
    regime = compute_regime_features(df, macro)

    close = df['close'].values
    n = len(close)

    # Build targets
    y_short = np.zeros(n)
    y_short[:-1] = (close[1:] - close[:-1]) / close[:-1]
    y_medium = np.zeros(n)
    for i in range(n - 5):
        y_medium[i] = (close[i + 5] - close[i]) / close[i]
    y_long = np.zeros(n)
    for i in range(n - 20):
        y_long[i] = (close[i + 20] - close[i]) / close[i]

    X = feats.values

    # Adaptive feature selection
    selected_features, sel_names = _adaptive_feature_selection(
        X, y_short, y_medium, y_long, feature_names, verbose, max_features=150, ticker=ticker)
    X_sel = X[:, selected_features]

    # Walk-forward validation (on selected features)
    train_window = 504   # 2 years training
    test_window = 63     # ~3 months test
    wf_results = _walk_forward_validate(X_sel, y_short, y_medium, y_long, train_window, test_window, verbose)

    # Save feature stability for next retrain's feature selection
    fs_data = wf_results.get('feature_stability', {})
    if fs_data:
        # Map selected feature indices back to full feature space
        full_stability = {str(selected_features[int(k)]): v for k, v in fs_data.items()
                          if int(k) < len(selected_features)}
        _save_feature_stability(ticker, full_stability)

    # Build probability calibrators from WF out-of-sample predictions
    calibrators = _calibrate_probabilities(wf_results, X_sel, y_short, y_medium, y_long, train_window, test_window)
    # Show Brier score comparison (raw vs calibrated)
    if verbose:
        oos = wf_results.get('oos_data', {})
        for h in ('short', 'medium', 'long'):
            probs = np.array(oos.get(h, {}).get('probs', []))
            actuals = np.array(oos.get(h, {}).get('actuals', []))
            if len(probs) >= 20:
                raw_brier = float(np.mean((probs - actuals) ** 2))
                cal = calibrators.get(h)
                if cal:
                    cal_probs = cal.predict(probs)
                    cal_brier = float(np.mean((cal_probs - actuals) ** 2))
                    print(f"  {h.capitalize()} Brier: {raw_brier:.4f} → {cal_brier:.4f} (calibrated)")
                else:
                    print(f"  {h.capitalize()} Brier: {raw_brier:.4f}")
        if calibrators:
            print(f"  Probability calibration: {', '.join(calibrators.keys())} horizons")

    # Optuna tuning if requested
    best_xgb_params = None
    best_lgbm_params = None
    if tune:
        if verbose:
            print("Running Optuna hyperparameter tuning (50 trials per model)...")
        best_xgb_params, best_lgbm_params = _optuna_tune(X_sel, y_short, y_medium, y_long, train_window, test_window, verbose)

    # Train final models on ALL data (selected features)
    d = _ticker_model_dir(ticker)
    ens = EnsembleModel(model_dir=d)
    ens.selected_features = selected_features

    # Apply tuned params if available
    if best_xgb_params:
        for model in (ens.xgb_short, ens.xgb_medium, ens.xgb_long):
            model.model.set_params(**best_xgb_params)
    if best_lgbm_params:
        for model in (ens.lgbm_short, ens.lgbm_medium, ens.lgbm_long):
            model.model.set_params(**best_lgbm_params)

    # Exponential decay sample weights
    decay = 0.999
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.mean()

    # Train regressors (for magnitude)
    ens.xgb_short.train(X_sel, y_short, sample_weight=weights)
    ens.xgb_medium.train(X_sel, y_medium, sample_weight=weights)
    ens.xgb_long.train(X_sel, y_long, sample_weight=weights)

    # Train LightGBM
    ens.lgbm_short.train(X_sel, y_short, sample_weight=weights)
    ens.lgbm_medium.train(X_sel, y_medium, sample_weight=weights)
    ens.lgbm_long.train(X_sel, y_long, sample_weight=weights)

    # Train classifiers (for direction) with adaptive dead zone filtering + bagging
    bagged_classifiers = _train_classifiers_with_bagging(ens, X_sel, y_short, y_medium, y_long, weights)

    # Train LSTM
    ens.lstm.train(X_sel, y_short)

    # Compute validation accuracy on last test_window for weighting
    xgb_acc, lgbm_acc, accuracies = _compute_val_accuracy(ens, X_sel, y_short, y_medium, y_long, n, test_window)

    ens.set_weights_from_accuracy(xgb_acc, lgbm_acc)
    ens.save()
    _ticker_ensembles[ticker] = ens

    # Multi-timeframe voting: train classifiers with different lookback windows
    mtf_models = _train_mtf_models(X_sel, y_short, y_medium, y_long, n)
    if mtf_models:
        import pickle
        mtf_path = d / 'mtf_models.pkl'
        mtf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mtf_path, 'wb') as fp:
            pickle.dump(mtf_models, fp)
    if verbose and mtf_models:
        print(f"  Multi-timeframe models: {list(mtf_models.keys())} windows")

    # Save training artifacts (bagged classifiers, calibrators, residuals, feature stats)
    _save_training_artifacts(d, bagged_classifiers, calibrators, wf_results, X_sel)

    # Train hourly short-term model if data available
    hourly_acc = _train_hourly_model(ticker, d, verbose)

    meta = _build_training_meta(
        ticker, n, history_days, feature_names, selected_features, sel_names,
        accuracies, xgb_acc, lgbm_acc, wf_results, ens, calibrators,
        regime, y_short, hourly_acc, best_xgb_params, best_lgbm_params)
    _save_ticker_meta(ticker, meta)

    if verbose:
        _print_train_summary(ticker, accuracies, wf_results, selected_features,
                             feature_names, sel_names, meta, prev_meta)

    return accuracies


def _wf_train_horizon(name, y, X, test_start, test_window, hp, oos_data, wf_residuals, diversity_samples):
    """Train and evaluate one horizon within a walk-forward window. Returns (accuracy, feature_importance_or_None)."""
    import xgboost as xgb
    import lightgbm as lgb

    purge = {'short': 1, 'medium': 5, 'long': 20}
    gap = purge[name]
    train_end = test_start - gap
    if train_end < 100:
        return 0.5, None

    X_train_full = X[0:train_end]
    X_test = X[test_start:test_start + test_window]
    y_tr_full = y[0:train_end]
    actual = y[test_start:test_start + test_window]

    if name in ('medium', 'long'):
        valid_mask = actual != 0
        if np.sum(valid_mask) < 5:
            return 0.5, None
        X_test_valid, actual_valid = X_test[valid_mask], actual[valid_mask]
    else:
        X_test_valid, actual_valid = X_test, actual

    p = hp[name]
    abs_rets = np.abs(y_tr_full)
    dz = float(np.percentile(abs_rets[abs_rets > 0], p['dz_pct'])) if np.any(abs_rets > 0) else 0.001
    mask = abs_rets > dz
    if np.sum(mask) < 50:
        mask = np.ones(len(y_tr_full), dtype=bool)
    X_train, y_tr = X_train_full[mask], y_tr_full[mask]

    y_cls_tr = (y_tr > 0).astype(int)
    actual_cls = (actual_valid > 0).astype(int)
    if len(np.unique(y_cls_tr)) < 2:
        return 0.5, None

    n_train = len(X_train)
    w_train = np.array([0.999 ** (n_train - 1 - i) for i in range(n_train)])
    w_train /= w_train.mean()

    n_pos = y_cls_tr.sum()
    n_neg = len(y_cls_tr) - n_pos
    spw = n_neg / max(n_pos, 1) if n_pos < n_neg * 0.7 or n_neg < n_pos * 0.7 else 1.0

    bag_probs = []
    window_imp = None
    for seed in (42, 123, 7):
        xgb_m = xgb.XGBClassifier(
            n_estimators=p['n_est'], max_depth=p['depth'], learning_rate=p['lr'],
            subsample=p['subsample'], colsample_bytree=p['colsample'],
            reg_alpha=p['alpha'], reg_lambda=p['lam'], min_child_weight=p['mcw'],
            gamma=p['gamma'], scale_pos_weight=spw,
            eval_metric='logloss', random_state=seed, n_jobs=-1)
        xgb_m.fit(X_train, y_cls_tr, sample_weight=w_train)
        lgb_m = lgb.LGBMClassifier(
            n_estimators=p['n_est'], max_depth=p['depth'], learning_rate=p['lr'],
            subsample=p['subsample'], colsample_bytree=p['colsample'],
            reg_alpha=p['alpha'], reg_lambda=p['lam'], min_child_weight=p['mcw'],
            num_leaves=p['leaves'], scale_pos_weight=spw,
            random_state=seed, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_train, y_cls_tr, sample_weight=w_train)
        bag_probs.append(0.5 * xgb_m.predict_proba(X_test_valid)[:, 1] +
                         0.5 * lgb_m.predict_proba(X_test_valid)[:, 1])
        if seed == 42 and name == 'medium':
            window_imp = xgb_m.feature_importances_.copy()

    ens_prob = np.mean(bag_probs, axis=0)
    base_rate = float(n_pos / len(y_cls_tr)) if len(y_cls_tr) > 0 else 0.5
    threshold = 1.0 - base_rate
    clip_ranges = {'short': (0.42, 0.52), 'medium': (0.35, 0.52), 'long': (0.30, 0.52)}
    lo, hi = clip_ranges.get(name, (0.42, 0.52))
    threshold = np.clip(threshold, lo, hi)
    ens_pred = (ens_prob > threshold).astype(int)

    if len(bag_probs) >= 2:
        diversity_samples[name].append(bag_probs.copy())
    oos_data[name]['probs'].extend(ens_prob.tolist())
    oos_data[name]['actuals'].extend(actual_cls.tolist())

    xgb_reg = xgb.XGBRegressor(
        n_estimators=p['n_est'], max_depth=p['depth'], learning_rate=p['lr'],
        subsample=p['subsample'], colsample_bytree=p['colsample'],
        reg_alpha=p['alpha'], reg_lambda=p['lam'], min_child_weight=p['mcw'],
        random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train_full, y_tr_full)
    pred_returns = xgb_reg.predict(X_test_valid)
    wf_residuals[name].extend((actual_valid - pred_returns).tolist())

    correct = np.sum(ens_pred == actual_cls)
    acc = round(float(correct / len(actual_valid)) if len(actual_valid) > 0 else 0.5, 4)
    return acc, window_imp


def _wf_aggregate_results(windows, importance_list, X, oos_data, wf_residuals, diversity_samples, verbose):
    """Aggregate walk-forward window results into final metrics."""
    from backend.models.explain import compute_ensemble_diversity

    avg, std_devs, ranges, trend = {}, {}, {}, {}
    if windows:
        for h in ('short', 'medium', 'long'):
            vals = [w.get(h, 0.5) for w in windows]
            avg[h] = round(float(np.mean(vals)), 4)
            std_devs[h] = round(float(np.std(vals)), 4)
            ranges[h] = (round(float(min(vals)), 4), round(float(max(vals)), 4))
            if len(vals) >= 4:
                mid = len(vals) // 2
                diff = float(np.mean(vals[mid:])) - float(np.mean(vals[:mid]))
                trend[h] = 'improving' if diff > 0.02 else ('degrading' if diff < -0.02 else 'stable')
            else:
                trend[h] = 'insufficient_data'
        if verbose:
            print(f"\n  Walk-forward ({len(windows)} windows):")
            for h in ('short', 'medium', 'long'):
                lo, hi = ranges[h]
                print(f"    {h.capitalize()}: {avg[h]*100:.1f}% ± {std_devs[h]*100:.1f}% (range: {lo*100:.0f}-{hi*100:.0f}%)")
            degrading = [h for h, t in trend.items() if t == 'degrading']
            if degrading:
                print(f"  ⚠ Accuracy degrading for: {', '.join(degrading)}")

    feature_stability = {}
    if windows and importance_list:
        n_feat = X.shape[1]
        counts = np.zeros(n_feat)
        for imp in importance_list:
            if len(imp) == n_feat:
                top20 = np.argsort(imp)[-20:]
                counts[top20] += 1
        if len(importance_list) > 0:
            feature_stability = {int(i): round(float(counts[i] / len(importance_list)), 3)
                                 for i in range(n_feat) if counts[i] > 0}

    result = {'windows': windows, 'average': avg, 'std': std_devs, 'ranges': ranges, 'trend': trend}

    conf_filtered = {}
    for h in ('short', 'medium', 'long'):
        probs = np.array(oos_data.get(h, {}).get('probs', []))
        actuals = np.array(oos_data.get(h, {}).get('actuals', []))
        if len(probs) >= 20:
            conf_mask = np.abs(probs - 0.5) > 0.05
            if np.sum(conf_mask) >= 10:
                preds = (probs[conf_mask] > 0.5).astype(int)
                correct = np.sum(preds == actuals[conf_mask])
                conf_filtered[h] = {
                    'accuracy': round(float(correct / len(preds)), 4),
                    'count': int(np.sum(conf_mask)),
                    'total': len(probs),
                }
    if conf_filtered:
        result['confidence_filtered'] = conf_filtered
        if verbose:
            print("  Confidence-filtered accuracy (predictions with >55% confidence):")
            for h, cf in conf_filtered.items():
                print(f"    {h.capitalize()}: {cf['accuracy']*100:.1f}% on {cf['count']}/{cf['total']} predictions")

    if feature_stability:
        result['feature_stability'] = feature_stability
    result['oos_data'] = oos_data
    result['wf_residuals'] = {h: v for h, v in wf_residuals.items() if v}

    ensemble_div = {}
    for h in ('short', 'medium', 'long'):
        all_bags = diversity_samples.get(h, [])
        if all_bags:
            concat = [np.concatenate([b[i] for b in all_bags]) for i in range(len(all_bags[0]))]
            ensemble_div[h] = compute_ensemble_diversity(concat)
    if ensemble_div:
        result['ensemble_diversity'] = ensemble_div
        if verbose:
            print("  Ensemble diversity:")
            for h, d in ensemble_div.items():
                print(f"    {h.capitalize()}: {d['diversity']*100:.1f}% disagreement, "
                      f"r={d['avg_correlation']:.2f} ({d['description']})")

    return result


def _walk_forward_validate(X, y_short, y_medium, y_long, train_window, test_window, verbose=False):
    """Walk-forward validation with purge gap, bagged ensemble, and adaptive complexity."""
    n = len(X)
    max_wf_samples = min(n, 1260)
    wf_start = n - max_wf_samples

    daily_vol = float(np.std(y_short[y_short != 0])) if np.any(y_short != 0) else 0.02
    vol_adj = min(5, max(0, int((daily_vol - 0.01) / 0.005 * 2)))
    hp = {
        'short': {'n_est': 50, 'depth': 2, 'lr': 0.05, 'subsample': 0.7,
                  'colsample': 0.4, 'alpha': 0.2, 'lam': 2.0, 'mcw': 10, 'gamma': 0.1,
                  'leaves': 4, 'dz_pct': 3 + vol_adj},
        'medium': {'n_est': 80, 'depth': 3, 'lr': 0.04, 'subsample': 0.7,
                   'colsample': 0.5, 'alpha': 0.15, 'lam': 1.5, 'mcw': 8, 'gamma': 0.05,
                   'leaves': 8, 'dz_pct': 2},
        'long': {'n_est': 80, 'depth': 3, 'lr': 0.06, 'subsample': 0.8,
                 'colsample': 0.5, 'alpha': 0.1, 'lam': 1.0, 'mcw': 5, 'gamma': 0.0,
                 'leaves': 8, 'dz_pct': 0},
    }

    windows = []
    importance_list = []
    oos_data = {h: {'probs': [], 'actuals': []} for h in ('short', 'medium', 'long')}
    wf_residuals = {h: [] for h in ('short', 'medium', 'long')}
    diversity_samples = {h: [] for h in ('short', 'medium', 'long')}
    targets = {'short': y_short, 'medium': y_medium, 'long': y_long}

    test_start = wf_start + train_window
    while test_start + test_window <= n:
        window_acc = {}
        window_imp = None
        for name in ('short', 'medium', 'long'):
            acc, imp = _wf_train_horizon(name, targets[name], X, test_start, test_window,
                                          hp, oos_data, wf_residuals, diversity_samples)
            window_acc[name] = acc
            if imp is not None:
                window_imp = imp

        windows.append(window_acc)
        if window_imp is not None:
            importance_list.append(window_imp)
        if verbose and len(windows) <= 30:
            print(f"  Window {len(windows)}: {window_acc.get('short', 0.5)*100:.1f}% | "
                  f"{window_acc.get('medium', 0.5)*100:.1f}% | {window_acc.get('long', 0.5)*100:.1f}%")
        test_start += test_window

    if verbose and len(windows) > 30:
        print(f"  ... ({len(windows)} total windows)")

    return _wf_aggregate_results(windows, importance_list, X, oos_data, wf_residuals, diversity_samples, verbose)


def _optuna_tune(X, y_short, y_medium, y_long, train_window, test_window, verbose=False):
    """Run Optuna hyperparameter tuning for XGBoost and LightGBM."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(X)

    def _wf_score(model_cls, params, y_target):
        """Walk-forward accuracy for a given model and params."""
        accs = []
        start = 0
        while start + train_window + test_window <= n:
            train_end = start + train_window
            test_end = train_end + test_window
            m = model_cls(**params)
            m.fit(X[start:train_end], y_target[start:train_end])
            preds = m.predict(X[train_end:test_end])
            actual = y_target[train_end:test_end]
            correct = np.sum((preds > 0) == (actual > 0))
            accs.append(correct / len(actual) if len(actual) > 0 else 0.5)
            start += test_window
        return np.mean(accs) if accs else 0.5

    # XGBoost tuning
    import xgboost as xgb

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
            'random_state': 42, 'n_jobs': -1,
        }
        # Average across all horizons
        scores = [_wf_score(xgb.XGBRegressor, params, y) for y in (y_short, y_medium, y_long)]
        return np.mean(scores)

    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
    best_xgb = {k: v for k, v in xgb_study.best_params.items()}
    best_xgb['random_state'] = 42
    best_xgb['n_jobs'] = -1
    if verbose:
        print(f"  Best XGBoost score: {xgb_study.best_value*100:.1f}%")

    # LightGBM tuning
    import lightgbm as lgb

    def lgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 10, 60),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        scores = [_wf_score(lgb.LGBMRegressor, params, y) for y in (y_short, y_medium, y_long)]
        return np.mean(scores)

    lgbm_study = optuna.create_study(direction='maximize')
    lgbm_study.optimize(lgbm_objective, n_trials=50, show_progress_bar=False)
    best_lgbm = {k: v for k, v in lgbm_study.best_params.items()}
    best_lgbm['random_state'] = 42
    best_lgbm['n_jobs'] = -1
    best_lgbm['verbose'] = -1
    if verbose:
        print(f"  Best LightGBM score: {lgbm_study.best_value*100:.1f}%")

    return best_xgb, best_lgbm


def _ensure_ticker_model(ticker: str) -> EnsembleModel:
    """Ensure a per-ticker model exists, training if needed. Falls back to generic."""
    ticker = ticker.upper()
    if ticker in _ticker_ensembles:
        return _ticker_ensembles[ticker]
    d = _ticker_model_dir(ticker)
    try:
        ens = EnsembleModel(model_dir=d)
        if ens.load():
            meta = _load_ticker_meta(ticker)
            if meta:
                # Version compatibility check
                meta_ver = meta.get('model_version', '0.0.0')
                if meta_ver != MODEL_VERSION:
                    import logging
                    logging.getLogger('cli.engine').warning(
                        "Model for %s was trained with v%s (current v%s), consider retraining",
                        ticker, meta_ver, MODEL_VERSION)
                # Staleness check: warn if trained > 30 days ago
                trained_at = meta.get('trained_at')
                if trained_at:
                    try:
                        age = (datetime.now() - datetime.fromisoformat(trained_at)).days
                        if age > 30:
                            import logging
                            logging.getLogger('cli.engine').warning(
                                "Model for %s is %d days old, consider retraining", ticker, age)
                    except (ValueError, TypeError):
                        pass
                if 'ensemble_weights' in meta:
                    ens.weights = meta['ensemble_weights']
            _ticker_ensembles[ticker] = ens
            return ens
    except Exception:
        pass  # corrupted model files
    # Try to train
    try:
        train_ticker_model(ticker)
        return _ticker_ensembles[ticker]
    except Exception:
        # Fall back to generic
        _ensure_models()
        return _ensemble


def retrain_ticker(ticker: str, tune: bool = False) -> dict:
    """Force retrain a specific ticker's model."""
    ticker = ticker.upper()
    _ticker_ensembles.pop(ticker, None)
    _ticker_meta.pop(ticker, None)
    return train_ticker_model(ticker, verbose=True, tune=tune)


def retrain_all() -> dict[str, dict]:
    """Retrain all tickers that have saved models."""
    results = {}
    if _MODELS_DIR.exists():
        for d in _MODELS_DIR.iterdir():
            if d.is_dir() and (d / 'meta.json').exists():
                ticker = d.name
                try:
                    results[ticker] = retrain_ticker(ticker)
                except Exception as e:
                    results[ticker] = {'error': str(e)}
    return results


def global_feature_importance(top_n: int = 20) -> dict:
    """Aggregate feature importance across all trained tickers.

    Returns features ranked by how often they appear in top-20 across tickers,
    weighted by walk-forward accuracy.
    """
    if not _MODELS_DIR.exists():
        return {'features': [], 'ticker_count': 0}

    all_names = FeatureStore.all_feature_names()
    feature_scores: dict[str, float] = {}
    feature_counts: dict[str, int] = {}
    ticker_count = 0

    for d in sorted(_MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / 'meta.json').exists():
            continue
        meta = _load_ticker_meta(d.name)
        if not meta:
            continue
        ticker_count += 1
        sel_features = meta.get('selected_features', [])
        sel_names = meta.get('feature_names', [])
        if not sel_names:
            sel_names = [all_names[i] for i in sel_features if i < len(all_names)]

        # Weight by average WF accuracy (better models count more)
        wf = meta.get('walk_forward', {}).get('average', {})
        avg_wf = float(np.mean([v for v in wf.values() if isinstance(v, (int, float))])) if wf else 0.5
        weight = max(0.1, avg_wf - 0.45)  # 0.55 acc → 0.1 weight, 0.65 → 0.2

        # Load feature stability if available (keys are string indices in full feature space)
        stability = _load_feature_stability(d.name)
        # Build name→stability mapping (keys may be string indices or feature names)
        name_stability = {}
        if stability:
            for key, freq in stability.items():
                try:
                    idx = int(key)
                    if idx < len(all_names):
                        name_stability[all_names[idx]] = freq
                except ValueError:
                    name_stability[key] = freq
        for name in sel_names[:top_n]:
            stab = name_stability.get(name, 0.3)
            score = weight * stab
            feature_scores[name] = feature_scores.get(name, 0) + score
            feature_counts[name] = feature_counts.get(name, 0) + 1

    from backend.models.explain import _readable_name
    ranked = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [
        {
            'name': _readable_name(name),
            'raw_name': name,
            'score': round(score, 4),
            'ticker_count': feature_counts.get(name, 0),
            'pct_tickers': round(feature_counts.get(name, 0) / max(ticker_count, 1) * 100, 0),
        }
        for name, score in ranked
    ]
    return {'features': features, 'ticker_count': ticker_count}


def model_status() -> list[dict]:
    """Get health/accuracy summary for all trained models."""
    statuses = []
    if not _MODELS_DIR.exists():
        return statuses
    for d in sorted(_MODELS_DIR.iterdir()):
        if not d.is_dir() or not (d / 'meta.json').exists():
            continue
        meta = _load_ticker_meta(d.name)
        if not meta:
            continue
        wf = meta.get('walk_forward', {}).get('average', {})
        health = meta.get('model_health', {})
        age = None
        if meta.get('trained_at'):
            try:
                age = (datetime.now() - datetime.fromisoformat(meta['trained_at'])).days
            except (ValueError, TypeError):
                pass
        brier = meta.get('brier_scores', {})
        # Average calibrated Brier across horizons (lower = better calibration)
        brier_vals = [b.get('calibrated', b.get('raw')) for b in brier.values()
                      if isinstance(b, dict) and (b.get('calibrated') or b.get('raw'))]
        avg_brier = round(float(np.mean(brier_vals)), 4) if brier_vals else None
        statuses.append({
            'ticker': d.name,
            'health_grade': health.get('grade', '?'),
            'health_score': health.get('score', 0),
            'wf_short': wf.get('short'),
            'wf_medium': wf.get('medium'),
            'wf_long': wf.get('long'),
            'features': meta.get('selected_feature_count') or meta.get('feature_count'),
            'total_features': meta.get('feature_count'),
            'age_days': age,
            'calibrated': meta.get('has_calibrators', False),
            'avg_brier': avg_brier,
            'best_horizon': max(wf, key=wf.get) if wf else None,
        })
    return statuses


def get_model_explanation(ticker: str) -> dict:
    """Get detailed model explanation for a ticker: meta, health, features, SHAP, calibration."""
    ticker = ticker.upper()
    meta = _load_ticker_meta(ticker)
    if not meta:
        return {'error': f'No trained model for {ticker}'}
    wf = meta.get('walk_forward', {})
    health = meta.get('model_health', {})
    brier = meta.get('brier_scores', {})
    result = {
        'ticker': ticker,
        'trained_at': meta.get('trained_at'),
        'model_version': meta.get('model_version'),
        'samples': meta.get('samples'),
        'feature_count': meta.get('feature_count'),
        'selected_feature_count': meta.get('selected_feature_count'),
        'wf_accuracy': wf.get('average', {}),
        'wf_std': wf.get('std', {}),
        'wf_ranges': wf.get('ranges', {}),
        'wf_trend': wf.get('trend', {}),
        'health': health,
        'brier_scores': brier,
        'calibrated': meta.get('has_calibrators', False),
        'regime': meta.get('regime', {}),
        'ensemble_weights': meta.get('ensemble_weights', {}),
        'ensemble_diversity': meta.get('ensemble_diversity', {}),
        'top_features': (meta.get('feature_names') or [])[:20],
    }
    # Load health trend
    trend_path = _ticker_model_dir(ticker) / 'health_trend.json'
    if trend_path.exists():
        import json as _json
        try:
            trend = _json.loads(trend_path.read_text())
            result['health_trend'] = trend[-5:]
        except Exception:
            pass
    # Load feature changelog
    changelog_path = _ticker_model_dir(ticker) / 'feature_changelog.json'
    if changelog_path.exists():
        import json as _json
        try:
            cl = _json.loads(changelog_path.read_text())
            result['feature_changelog'] = cl[-3:]  # last 3 entries
        except Exception:
            pass
    # Load feature stability
    stability = _load_feature_stability(ticker)
    if stability:
        sorted_s = sorted(stability.items(), key=lambda x: x[1], reverse=True)
        result['feature_stability_top'] = sorted_s[:10]
        result['stable_feature_count'] = sum(1 for v in stability.values() if v >= 0.5)
    # Conformal interval summary from WF residuals
    res_path = _ticker_model_dir(ticker) / 'wf_residuals.pkl'
    if res_path.exists():
        import pickle
        try:
            with open(res_path, 'rb') as fp:
                wf_residuals = pickle.load(fp)
            conf_summary = {}
            for h in ('short', 'medium', 'long'):
                residuals = wf_residuals.get(h, [])
                if residuals:
                    abs_res = [abs(r) for r in residuals]
                    q90 = float(np.percentile(abs_res, 90))
                    conf_summary[h] = {
                        'n_residuals': len(residuals),
                        'interval_width_90': round(q90 * 2 * 100, 2),
                    }
            if conf_summary:
                result['conformal_summary'] = conf_summary
        except Exception:
            pass
    return result


def feature_sensitivity(ticker: str, top_n: int = 5) -> dict:
    """Show how predictions change when top features are perturbed ±1 std."""
    ticker = ticker.upper()
    meta = _load_ticker_meta(ticker)
    if not meta:
        return {'error': f'No trained model for {ticker}'}
    ens = _get_ticker_ensemble(ticker)
    if not ens.loaded:
        return {'error': f'No trained model for {ticker}'}

    # Get current features using same pattern as get_analysis
    try:
        price_df = generate_ohlcv(ticker, "daily", 365)
    except Exception:
        return {'error': f'Cannot fetch data for {ticker}'}
    spot = float(price_df['close'].iloc[-1])
    opts = get_options_chain(ticker, spot)
    fund = get_fundamentals(ticker)
    sent = _get_sentiment_safe(ticker)
    macro = get_macro_data()
    spy_df, sector_df = _fetch_spy_sector_data(ticker, len(price_df))
    use_seq = meta.get('include_sequence', False)
    feats = _store.compute_all_features(price_df, opts, fund, sent, macro,
                                         spy_df=spy_df, sector_df=sector_df,
                                         include_sequence=use_seq)
    if feats.empty:
        return {'error': 'No feature data'}

    feature_names = list(feats.columns)
    selected = None
    if ens.selected_features is not None:
        valid_idx = [i for i in ens.selected_features if i < len(feature_names)]
        if valid_idx:
            selected = valid_idx
            feature_names = [feature_names[i] for i in valid_idx]

    row = feats.values[-1]
    if selected is not None:
        X_base = row[selected].astype(np.float64)
    else:
        X_base = row.astype(np.float64)
    X_base = np.nan_to_num(X_base, nan=0.0)

    # Base prediction
    base_pred = ens.predict(X_base)
    base_prob = base_pred.get('short', {}).get('probability', 0.5)

    # Pick top features by importance
    importances = meta.get('feature_importances', {})
    if importances:
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_feats = [f for f, _ in sorted_imp if f in feature_names][:top_n]
    else:
        top_feats = feature_names[:top_n]

    # Compute std from feature columns
    sensitivities = []
    from backend.models.explain import _readable_name
    for feat in top_feats:
        if feat not in feature_names:
            continue
        idx = feature_names.index(feat)
        col = feats.get(feat)
        std = float(col.std()) if col is not None and len(col) > 5 and col.std() > 1e-8 else 1.0

        X_up = X_base.copy()
        X_up[idx] += std
        prob_up = ens.predict(X_up).get('short', {}).get('probability', 0.5)

        X_down = X_base.copy()
        X_down[idx] -= std
        prob_down = ens.predict(X_down).get('short', {}).get('probability', 0.5)

        sensitivities.append({
            'feature': _readable_name(feat),
            'raw_name': feat,
            'current_value': round(float(X_base[idx]), 4),
            'std': round(std, 4),
            'base_prob': round(base_prob, 4),
            'prob_plus_1std': round(prob_up, 4),
            'prob_minus_1std': round(prob_down, 4),
            'sensitivity': round(abs(prob_up - prob_down), 4),
        })

    sensitivities.sort(key=lambda x: x['sensitivity'], reverse=True)
    return {
        'ticker': ticker,
        'base_probability': round(base_prob, 4),
        'sensitivities': sensitivities,
    }


# Feature group definitions for ablation study
_FEATURE_GROUPS = {
    'price_volume': lambda n: not any(x in n for x in ['_lag', '_roc5', '_mean5', '_std5', 'sector_', 'spy_',
                                                         'return_vs_', 'correlation_', 'consecutive_', 'drawdown_',
                                                         'rally_', 'days_since_', 'volatility_expansion',
                                                         'price_acceleration', 'volume_trend_']) and
                    any(n.startswith(p) for p in ['open_', 'high_', 'low_', 'close_', 'volume_', 'vwap', 'atr',
                                                   'daily_return', 'log_return', 'range_pct', 'close_in_range',
                                                   'gap_pct', 'body_pct', 'upper_shadow', 'lower_shadow',
                                                   'vol_ratio', 'price_vs_vwap', 'obv', 'mfi', 'vpt',
                                                   'dollar_volume', 'avg_volume']),
    'technical': lambda n: not any(x in n for x in ['_lag', '_roc5', '_mean5', '_std5']) and
                 any(n.startswith(p) for p in ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'stoch_', 'adx',
                                                'cci', 'willr', 'roc_', 'ppo', 'trix', 'dpo', 'cmo',
                                                'aroon', 'ichimoku', 'keltner', 'donchian', 'pivot',
                                                'fib_', 'hurst', 'fractal']),
    'lagged': lambda n: any(x in n for x in ['_lag', '_roc5', '_mean5', '_std5']),
    'price_derived': lambda n: n in ['consecutive_up_days', 'consecutive_down_days', 'drawdown_from_20d_high',
                                      'rally_from_20d_low', 'days_since_last_5pct_drop', 'volatility_expansion',
                                      'price_acceleration', 'volume_trend_5d'] or
                     any(n.startswith(p) for p in ['consecutive_', 'drawdown_', 'rally_', 'days_since_',
                                                    'volatility_expansion', 'price_acceleration', 'volume_trend_']),
    'sector_relative': lambda n: any(n.startswith(p) for p in ['return_vs_', 'spy_', 'correlation_with_spy',
                                                                 'sector_momentum']),
    'options': lambda n: any(n.startswith(p) for p in ['put_call', 'iv_', 'skew', 'term_', 'oi_', 'max_pain',
                                                        'gamma_', 'delta_', 'vanna', 'charm', 'speed',
                                                        'options_', 'call_', 'put_']),
    'fundamentals': lambda n: any(n.startswith(p) for p in ['pe_', 'pb_', 'ps_', 'ev_', 'roe', 'roa',
                                                              'debt_', 'current_ratio', 'quick_ratio',
                                                              'gross_margin', 'operating_margin', 'net_margin',
                                                              'revenue_', 'earnings_', 'fcf_', 'dividend_']),
    'sentiment': lambda n: any(n.startswith(p) for p in ['news_', 'social_', 'insider_', 'analyst_',
                                                           'short_interest', 'sentiment_', 'filing_',
                                                           'earnings_call_']),
    'macro': lambda n: any(n.startswith(p) for p in ['vix', 'yield_', 'spread_', 'fed_', 'gdp_', 'cpi_',
                                                       'unemployment_', 'pmi_', 'dollar_index', 'oil_',
                                                       'gold_', 'copper_']),
    'regime': lambda n: any(n.startswith(p) for p in ['regime_', 'vol_regime_', 'trend_', 'mom_', 'momentum_roc',
                                                        'adx_value']),
}


def feature_ablation(ticker: str) -> dict:
    """Run ablation study: measure accuracy drop when each feature group is removed."""
    ticker = ticker.upper()
    meta = _load_ticker_meta(ticker)
    if not meta:
        return {'error': f'No trained model for {ticker}. Run: stk retrain {ticker}'}

    ens = _ensure_ticker_model(ticker)
    sel_features = meta.get('selected_features', [])
    all_names = FeatureStore.all_feature_names()
    sel_names = [all_names[i] for i in sel_features if i < len(all_names)]

    # Classify selected features into groups
    group_indices = {}  # group_name -> list of indices within selected feature array
    for idx, name in enumerate(sel_names):
        for gname, matcher in _FEATURE_GROUPS.items():
            if matcher(name):
                group_indices.setdefault(gname, []).append(idx)
                break

    # Get current prediction as baseline
    import yfinance as yf
    import io
    import contextlib
    t = yf.Ticker(ticker)
    f = io.StringIO()
    with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        hist = t.history(period="1y", interval="1d")
    if hist.empty:
        return {'error': f'No price data for {ticker}'}

    df = hist.reset_index()
    col_map = {"Date": "date", "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["symbol"] = ticker
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    spy_df, sector_df = _fetch_spy_sector_data(ticker, len(df))
    feats = _store.compute_all_features(df, spy_df=spy_df, sector_df=sector_df)
    X_full = feats.values
    X_sel = X_full[:, sel_features] if sel_features else X_full
    X_today = X_sel[-1:]

    # Baseline predictions
    baseline = {}
    for h in ('short', 'medium', 'long'):
        _, conf, _ = ens.predict_direction_ensemble(X_today, h)
        baseline[h] = conf

    # Ablate each group: zero out features and measure confidence change
    results = []
    for gname, indices in sorted(group_indices.items()):
        X_ablated = X_today.copy()
        X_ablated[0, indices] = 0.0
        drops = {}
        for h in ('short', 'medium', 'long'):
            _, conf_abl, _ = ens.predict_direction_ensemble(X_ablated, h)
            drops[h] = round(baseline[h] - conf_abl, 4)
        avg_drop = round(sum(drops.values()) / len(drops), 4)
        results.append({
            'group': gname,
            'feature_count': len(indices),
            'confidence_drop': drops,
            'avg_drop': avg_drop,
        })

    results.sort(key=lambda x: abs(x['avg_drop']), reverse=True)
    return {
        'ticker': ticker,
        'baseline_confidence': baseline,
        'groups': results,
        'total_selected': len(sel_names),
    }


def prediction_streak(ticker: str) -> dict:
    """Compute prediction streak stats: consecutive correct/wrong predictions."""
    ticker = ticker.upper()
    from cli.db import get_predictions
    preds = get_predictions(ticker, limit=200)
    if not preds:
        return {'ticker': ticker, 'total': 0}

    try:
        cur_price = get_price(ticker)['price']
    except Exception:
        return {'ticker': ticker, 'total': len(preds), 'evaluated': 0}

    horizon_days = {'short': 1, 'medium': 5, 'long': 20}
    now = datetime.now()

    # Evaluate each prediction
    evaluated = []
    for p in preds:
        price_at = p.get('price_at', 0)
        if price_at <= 0:
            continue
        h = p.get('horizon', 'short')
        days = horizon_days.get(h, 1)
        try:
            created = datetime.fromisoformat(p['created_at'])
            if (now - created).days < days:
                continue
        except (ValueError, KeyError, TypeError):
            continue
        actual_return = (cur_price - price_at) / price_at
        is_correct = (p['direction'] == 'bullish' and actual_return > 0) or \
                     (p['direction'] == 'bearish' and actual_return < 0)
        evaluated.append({
            'date': p.get('created_at', '')[:10],
            'horizon': h,
            'direction': p.get('direction'),
            'conviction': p.get('conviction_tier', ''),
            'correct': is_correct,
        })

    if not evaluated:
        return {'ticker': ticker, 'total': len(preds), 'evaluated': 0}

    # Current streak
    current_streak = 0
    streak_type = None
    for e in evaluated:
        if streak_type is None:
            streak_type = 'correct' if e['correct'] else 'wrong'
            current_streak = 1
        elif e['correct'] == (streak_type == 'correct'):
            current_streak += 1
        else:
            break

    # Best/worst streaks
    best_streak = worst_streak = 0
    cur_good = cur_bad = 0
    for e in evaluated:
        if e['correct']:
            cur_good += 1
            cur_bad = 0
        else:
            cur_bad += 1
            cur_good = 0
        best_streak = max(best_streak, cur_good)
        worst_streak = max(worst_streak, cur_bad)

    return {
        'ticker': ticker,
        'total': len(preds),
        'evaluated': len(evaluated),
        'current_streak': current_streak,
        'streak_type': streak_type or 'none',
        'best_correct_streak': best_streak,
        'worst_wrong_streak': worst_streak,
        'recent': evaluated[:10],
    }


def _ensure_models() -> None:
    global _loaded
    if not _loaded:
        _loaded = _ensemble.load()
        if not _loaded:
            df = generate_ohlcv("SPY", "daily", 365)
            opts = get_options_chain("SPY", df['close'].iloc[-1])
            fund = get_fundamentals("SPY")
            sent = _get_sentiment_safe("SPY")
            macro = get_macro_data()
            feats = _store.compute_all_features(df, opts, fund, sent, macro)
            prices = df['close'].values
            y = np.diff(prices) / prices[:-1]
            y = np.append(y, 0)
            y_m = np.zeros(len(prices))
            y_l = np.zeros(len(prices))
            for i in range(len(prices) - 20):
                y_m[i] = (prices[min(i + 5, len(prices) - 1)] / prices[i]) - 1
                y_l[i] = (prices[min(i + 20, len(prices) - 1)] / prices[i]) - 1
            X = feats.values
            _ensemble.train_all(X, y, y_m, y_l)
            _ensemble.train_classifiers(X, y, y_m, y_l)
            _loaded = True


def _classify_error(e: Exception, ticker: str) -> None:
    """Convert generic exceptions to specific error types."""
    msg = str(e).lower()
    if "no price data" in msg or "no data" in msg or "not found" in msg:
        raise InvalidTickerError(ticker) from e
    if "rate limit" in msg or "429" in msg or "too many" in msg:
        raise RateLimitError("Yahoo Finance") from e
    if "connection" in msg or "timeout" in msg or "network" in msg:
        raise NetworkError("Yahoo Finance", str(e)) from e
    raise e


def _gf(feats: pd.DataFrame, name: str) -> float | None:
    """Get feature value from DataFrame, or None if missing."""
    return float(feats[name].iloc[-1]) if name in feats.columns else None


def _build_horizons(raw: dict, spot: float, atr: float, feats: pd.DataFrame) -> dict:
    """Build horizon predictions with entry/exit levels."""
    horizons = {}
    sma50 = _gf(feats, 'sma_50')
    for h in ('short', 'medium', 'long'):
        p = raw[h]
        inv = calculate_invalidation(spot, p['direction'], h, atr)
        entry_lo = round(spot - atr * 0.3, 2)
        entry_hi = round(spot + atr * 0.3, 2)
        support = round(sma50, 2) if sma50 else round(spot - atr * 3, 2)
        resistance = round(spot + atr * 4, 2)
        horizons[h] = {
            'prediction': float(p['prediction']),
            'confidence': float(p['confidence']),
            'direction': p['direction'],
            'stop': inv['stop_loss'], 'target': inv['take_profit'],
            'entry_lo': entry_lo, 'entry_hi': entry_hi,
            'support': support, 'resistance': resistance,
        }
    return horizons


def _get_ticker_info(ticker: str) -> tuple[str, int, str, float]:
    """Fetch company name, market cap, sector, PE from yfinance."""
    info = _get_yf_info(ticker)
    if info:
        return (info.get('shortName', ticker), info.get('marketCap', 0),
                info.get('sector', 'N/A'), info.get('trailingPE', 0) or 0)
    return ticker, 0, 'N/A', 0


# Signal rules: (feature_name, bullish_condition, bearish_condition, bullish_msg, bearish_msg)
# Each rule is a tuple: (feature, bull_test, bear_test, bull_fmt, bear_fmt)
# bull_test/bear_test are lambdas taking the feature value, returning bool
# fmt functions take the value and return the message string

def _check_threshold(val: float | None, bull_thresh: float | None, bear_thresh: float | None, bull_fmt: Callable[[float], str], bear_fmt: Callable[[float], str], bullish: list[str], bearish: list[str]) -> None:
    """Check a value against bull/bear thresholds and append signals."""
    if val is None:
        return
    if bull_thresh is not None and val < bull_thresh:
        bullish.append(bull_fmt(val))
    elif bear_thresh is not None and val > bear_thresh:
        bearish.append(bear_fmt(val))


def _signals_momentum(gf, bullish, bearish):
    """Extract momentum-based signals (RSI, MACD, Stochastic, ROC, etc.)."""
    _check_threshold(gf('rsi_14'), 30, 70, lambda v: f"RSI oversold ({v:.1f})", lambda v: f"RSI overbought ({v:.1f})", bullish, bearish)
    macd_hist = gf('macd_hist')
    if macd_hist is not None:
        if macd_hist > 0.5: bullish.append(f"MACD positive ({macd_hist:.2f})")
        elif macd_hist < -0.5: bearish.append(f"MACD negative ({macd_hist:.2f})")
    _check_threshold(gf('williams_r'), -80, -20, lambda v: f"Williams %R oversold ({v:.1f})", lambda v: f"Williams %R overbought ({v:.1f})", bullish, bearish)
    _check_threshold(gf('mfi_14'), 30, 70, lambda v: f"MFI oversold ({v:.1f})", lambda v: f"MFI overbought ({v:.1f})", bullish, bearish)
    stoch_k, stoch_d = gf('stoch_k'), gf('stoch_d')
    if stoch_k is not None:
        _check_threshold(stoch_k, 20, 80, lambda v: f"Stochastic oversold ({v:.1f})", lambda v: f"Stochastic overbought ({v:.1f})", bullish, bearish)
        if stoch_k and stoch_d and stoch_k > stoch_d and stoch_k < 30:
            bullish.append("Stochastic bullish crossover")
    for feat, thresh, label in [('roc_5', 3, '5d'), ('roc_60', 10, '60d')]:
        v = gf(feat)
        if v is not None:
            if v > thresh: bullish.append(f"Strong {label} momentum ({v:+.1f}%)")
            elif v < -thresh: bearish.append(f"Weak {label} momentum ({v:+.1f}%)")
    _check_threshold(gf('williams_r_7'), -80, -20, lambda v: f"Williams %R(7) oversold ({v:.1f})", lambda v: f"Williams %R(7) overbought ({v:.1f})", bullish, bearish)
    _check_threshold(gf('cmo'), -50, 50, lambda v: f"CMO oversold ({v:.1f})", lambda v: f"CMO overbought ({v:.1f})", bullish, bearish)
    _check_threshold(gf('connors_rsi'), 20, 80, lambda v: f"Connors RSI oversold ({v:.1f})", lambda v: f"Connors RSI overbought ({v:.1f})", bullish, bearish)


def _signals_volume(gf, bullish, bearish):
    """Extract volume-based signals (volume ratio, OBV, CMF, force index, etc.)."""
    vol_ratio = gf('volume_ratio')
    if vol_ratio and vol_ratio > 1.5: bullish.append(f"Volume surge {vol_ratio:.1f}x")
    elif vol_ratio and vol_ratio < 0.7: bearish.append(f"Low volume {vol_ratio:.1f}x")
    obv_slope = gf('obv_slope')
    if obv_slope is not None:
        (bullish if obv_slope > 0 else bearish).append(f"{'Positive' if obv_slope > 0 else 'Negative'} OBV trend")
    cmf = gf('cmf')
    if cmf is not None:
        if cmf > 0.1: bullish.append(f"Strong money flow ({cmf:.2f})")
        elif cmf < -0.1: bearish.append(f"Weak money flow ({cmf:.2f})")
    fi = gf('force_index')
    if fi is not None:
        if fi > 0.01: bullish.append(f"Positive force index ({fi:.3f})")
        elif fi < -0.01: bearish.append(f"Negative force index ({fi:.3f})")
    adl = gf('adl_slope')
    if adl is not None:
        (bullish if adl > 0.01 else bearish if adl < -0.01 else []).append(
            "A/D line rising" if adl > 0.01 else "A/D line falling")
    emv = gf('emv')
    if emv is not None:
        if emv > 0.5: bullish.append("Positive ease of movement")
        elif emv < -0.5: bearish.append("Negative ease of movement")
    kvo = gf('kvo')
    if kvo is not None:
        if kvo > 0.5: bullish.append(f"Klinger positive ({kvo:.2f})")
        elif kvo < -0.5: bearish.append(f"Klinger negative ({kvo:.2f})")


def _signals_trend(gf, spot, bullish, bearish):
    """Extract trend-based signals (ADX, EMA, SMA, Ichimoku, Aroon, etc.)."""
    adx, ret5 = gf('adx_14'), gf('return_5d')
    if adx and adx > 20 and ret5:
        (bullish if ret5 > 0 else bearish).append(f"Strong {'up' if ret5 > 0 else 'down'}trend (ADX {adx:.1f})")
    sma50 = gf('sma_50')
    if sma50 and spot > 0:
        pct = (spot - sma50) / sma50 * 100
        if pct > 5: bullish.append(f"{pct:.1f}% above 50-SMA")
        elif pct < -5: bearish.append(f"{abs(pct):.1f}% below 50-SMA")
    ema5, ema20 = gf('ema_5'), gf('ema_20')
    if ema5 and ema20:
        if ema5 > ema20 * 1.01: bullish.append("EMA5 above EMA20")
        elif ema5 < ema20 * 0.99: bearish.append("EMA5 below EMA20")
    ichi_cloud = gf('ichimoku_cloud_pos')
    if ichi_cloud is not None:
        if ichi_cloud > 0.01: bullish.append("Above Ichimoku cloud")
        elif ichi_cloud < -0.01: bearish.append("Below Ichimoku cloud")
    ichi_tk = gf('ichimoku_tk_cross')
    if ichi_tk is not None:
        if ichi_tk > 0.005: bullish.append("Ichimoku TK bullish cross")
        elif ichi_tk < -0.005: bearish.append("Ichimoku TK bearish cross")
    aroon_up, aroon_down = gf('aroon_up'), gf('aroon_down')
    if aroon_up is not None and aroon_down is not None:
        if aroon_up > 70 and aroon_down < 30: bullish.append(f"Aroon bullish ({aroon_up:.0f}/{aroon_down:.0f})")
        elif aroon_down > 70 and aroon_up < 30: bearish.append(f"Aroon bearish ({aroon_up:.0f}/{aroon_down:.0f})")
    psar = gf('psar_dist')
    if psar is not None:
        if psar > 0.01: bullish.append("Above Parabolic SAR (uptrend)")
        elif psar < -0.01: bearish.append("Below Parabolic SAR (downtrend)")
    supertrend = gf('supertrend')
    if supertrend is not None:
        if supertrend > 0.01: bullish.append(f"Supertrend bullish ({supertrend:.3f})")
        elif supertrend < -0.01: bearish.append(f"Supertrend bearish ({supertrend:.3f})")
    hma = gf('hma_dist')
    if hma is not None:
        if hma > 0.02: bullish.append(f"Above Hull MA ({hma:.3f})")
        elif hma < -0.02: bearish.append(f"Below Hull MA ({hma:.3f})")


def _signals_volatility_bands(gf, spot, bullish, bearish):
    """Extract volatility and band-based signals (BB, Keltner, VWAP, VIX, etc.)."""
    _check_threshold(gf('bb_position'), 0.2, 0.8, lambda v: f"Near lower BB ({v:.2f})", lambda v: f"Near upper BB ({v:.2f})", bullish, bearish)
    _check_threshold(gf('keltner_position'), 0.1, 0.9, lambda v: "Near Keltner lower band", lambda v: "Near Keltner upper band", bullish, bearish)
    vwap, vwap_d = gf('vwap'), gf('vwap_distance')
    if vwap_d and vwap_d > 0.005: bullish.append(f"Above VWAP (${vwap:.2f})")
    elif vwap_d and vwap_d < -0.005: bearish.append(f"Below VWAP (${vwap:.2f})")
    if gf('vwap_lower') is not None and gf('vwap_lower') < 0.005: bullish.append("Near VWAP lower band")
    if gf('vwap_upper') is not None and gf('vwap_upper') < 0.005: bearish.append("Near VWAP upper band")
    vix_val = gf('vix')
    if vix_val:
        if vix_val > 25: bearish.append(f"High VIX ({vix_val:.1f})")
        elif vix_val < 15: bullish.append(f"Low VIX ({vix_val:.1f})")
    atrp = gf('atrp')
    if atrp is not None and atrp > 4: bearish.append(f"High volatility (ATRP {atrp:.1f}%)")
    don_brk = gf('donchian_breakout')
    if don_brk is not None and don_brk > 0: bullish.append("Donchian channel breakout")
    iv_pct = gf('iv_percentile')
    if iv_pct and iv_pct > 0.7: bearish.append(f"Elevated IV ({iv_pct:.0%})")
    elif iv_pct and iv_pct < 0.3: bullish.append(f"Low IV ({iv_pct:.0%})")


def _signals_advanced(gf, spot, bullish, bearish):
    """Extract advanced oscillator signals (DPO, KST, Elder Ray, STC, etc.)."""
    sent_avg = gf('news_sentiment_avg')
    if sent_avg and sent_avg > 0.1: bullish.append(f"Positive sentiment ({sent_avg:.2f})")
    elif sent_avg and sent_avg < -0.1: bearish.append(f"Negative sentiment ({sent_avg:.2f})")
    fib_382, fib_618 = gf('fib_382'), gf('fib_618')
    if fib_382 is not None and fib_618 is not None:
        if abs(fib_382) < 0.03: bullish.append("Near Fib 38.2% support")
        if abs(fib_618) < 0.03: bullish.append("Near Fib 61.8% support")
        if fib_382 > 0.1: bearish.append("Above Fib 38.2% resistance")
    pivot_dist = gf('pivot_distance')
    if pivot_dist is not None:
        if pivot_dist > 0.01: bullish.append("Above daily pivot")
        elif pivot_dist < -0.01: bearish.append("Below daily pivot")
    dpo = gf('dpo')
    if dpo is not None and spot > 0:
        dpo_pct = dpo / spot * 100
        if dpo_pct > 2: bullish.append(f"DPO positive ({dpo_pct:+.1f}%)")
        elif dpo_pct < -2: bearish.append(f"DPO negative ({dpo_pct:+.1f}%)")
    mass_idx = gf('mass_index')
    if mass_idx is not None and mass_idx > 27: bearish.append(f"Mass Index reversal bulge ({mass_idx:.1f})")
    kst = gf('kst')
    if kst is not None:
        if kst > 5: bullish.append(f"KST positive ({kst:.1f})")
        elif kst < -5: bearish.append(f"KST negative ({kst:.1f})")
    chop = gf('choppiness')
    if chop is not None:
        if chop > 61.8: bearish.append(f"Choppy market ({chop:.1f})")
        elif chop < 38.2: bullish.append(f"Strong trend ({chop:.1f})")
    coppock = gf('coppock_curve')
    if coppock is not None:
        if coppock > 0: bullish.append(f"Coppock Curve positive ({coppock:.1f})")
        elif coppock < -5: bearish.append(f"Coppock Curve negative ({coppock:.1f})")
    elder_bull, elder_bear = gf('elder_bull'), gf('elder_bear')
    if elder_bull is not None and elder_bear is not None:
        if elder_bull > 0.02 and elder_bear > -0.01: bullish.append("Elder Ray bullish (strong bulls)")
        elif elder_bear < -0.02 and elder_bull < 0.01: bearish.append("Elder Ray bearish (strong bears)")
    rvi_val = gf('rvi')
    if rvi_val is not None:
        if rvi_val > 0.3: bullish.append(f"RVI positive ({rvi_val:.2f})")
        elif rvi_val < -0.3: bearish.append(f"RVI negative ({rvi_val:.2f})")
    squeeze = gf('squeeze_mom')
    if squeeze is not None:
        if squeeze > 1: bullish.append(f"Squeeze momentum up ({squeeze:.1f})")
        elif squeeze < -1: bearish.append(f"Squeeze momentum down ({squeeze:.1f})")
    stc = gf('stc')
    if stc is not None:
        if stc > 75: bullish.append(f"STC bullish ({stc:.1f})")
        elif stc < 25: bearish.append(f"STC bearish ({stc:.1f})")


def _extract_signals(feats: pd.DataFrame, spot: float) -> tuple[list[str], list[str]]:
    """Extract bullish/bearish signals from computed features."""
    gf = lambda name: _gf(feats, name)
    bullish, bearish = [], []
    _signals_momentum(gf, bullish, bearish)
    _signals_volume(gf, bullish, bearish)
    _signals_trend(gf, spot, bullish, bearish)
    _signals_volatility_bands(gf, spot, bullish, bearish)
    _signals_advanced(gf, spot, bullish, bearish)
    return bullish, bearish


def _feature_importance_signals(ens: EnsembleModel, feats: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Generate signals from XGBoost feature importance."""
    from backend.models.explain import _readable_name
    bullish, bearish = [], []
    cols = list(feats.columns)
    imp = ens.xgb_medium.feature_importance
    if imp is None or len(imp) != len(cols):
        return bullish, bearish
    ranked = sorted(zip(cols, imp), key=lambda x: x[1], reverse=True)
    for name, importance in ranked[:20]:
        if importance < 0.005:
            continue
        val = _gf(feats, name)
        if val is None:
            continue
        label = _readable_name(name)
        pct = f" ({importance*100:.1f}% imp)"
        if val > 0:
            bullish.append(f"{label}: {val:.3f}{pct}")
        elif val < 0:
            bearish.append(f"{label}: {val:.3f}{pct}")
    return bullish, bearish


def _get_top_feature_importances(ens: EnsembleModel, feats: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Get top N features by importance with their current values."""
    from backend.models.explain import _readable_name
    cols = list(feats.columns)
    imp = ens.xgb_medium.feature_importance
    if imp is None or len(imp) != len(cols):
        return []
    ranked = sorted(zip(cols, imp), key=lambda x: x[1], reverse=True)
    result = []
    for name, importance in ranked[:top_n]:
        if importance < 0.001:
            break
        val = _gf(feats, name)
        result.append({'name': _readable_name(name), 'importance': float(importance),
                       'value': float(val) if val is not None else None})
    return result


def _calibrate_confidence(raw_conf: float, ticker: str) -> float:
    """Cap confidence by model's walk-forward validation accuracy for this ticker."""
    meta = _load_ticker_meta(ticker)
    if not meta:
        return raw_conf
    # Prefer walk-forward accuracy (honest out-of-sample) over training accuracy (overfitted)
    wf = meta.get('walk_forward', {}).get('average', {})
    if wf:
        accs = wf
    elif 'accuracy' in meta:
        accs = meta['accuracy']
    else:
        return raw_conf
    avg_acc = sum(accs.values()) / len(accs) if accs else 0.6
    ceiling = min(0.95, avg_acc + 0.05)
    return min(raw_conf, ceiling)


def _update_raw_conviction(raw, h, prob):
    """Update direction, confidence, and conviction tier for a horizon in raw predictions."""
    from backend.models.explain import get_conviction_tier
    raw[h]['prob_up'] = prob
    raw[h]['direction'] = 'bullish' if prob > 0.5 else 'bearish'
    raw[h]['confidence'] = max(prob, 1 - prob)
    tier, label, emoji = get_conviction_tier(raw[h]['confidence'])
    raw[h]['conviction_tier'] = tier
    raw[h]['conviction_label'] = label


def _apply_bagged_predictions(raw, model_dir, X_row):
    """Refine predictions with bagged classifiers for stability."""
    bag_path = model_dir / 'bagged_classifiers.pkl'
    if not bag_path.exists():
        return
    try:
        import pickle
        with open(bag_path, 'rb') as fp:
            bagged_cls = pickle.load(fp)
        x_2d = X_row.reshape(1, -1)
        for h in ('short', 'medium', 'long'):
            if h in bagged_cls and h in raw:
                bag_probs = [m.predict_proba(x_2d)[0, 1] for m in bagged_cls[h]]
                main_prob = raw[h].get('prob_up', 0.5)
                avg_prob = (main_prob + sum(bag_probs)) / (1 + len(bag_probs))
                _update_raw_conviction(raw, h, avg_prob)
    except Exception:
        pass


def _apply_calibration(raw, model_dir):
    """Apply probability calibration if available."""
    cal_path = model_dir / 'calibrators.pkl'
    if not cal_path.exists():
        return
    try:
        import pickle
        with open(cal_path, 'rb') as fp:
            calibrators = pickle.load(fp)
        for h in ('short', 'medium', 'long'):
            if h in calibrators and h in raw:
                prob_up = raw[h].get('prob_up', 0.5)
                cal = calibrators[h]
                if hasattr(cal, 'predict_proba'):
                    from backend.models.explain import apply_platt
                    cal_prob = apply_platt(cal, prob_up)
                else:
                    cal_prob = float(cal.predict([prob_up])[0])
                cal_prob = float(np.clip(cal_prob, 0.15, 0.85))
                _update_raw_conviction(raw, h, cal_prob)
                raw[h]['calibrated'] = True
    except Exception:
        pass


def _apply_regime_adjustment(raw, regime):
    """Discount confidence in high-volatility regimes."""
    if regime.get('vol_regime_high', 0) <= 0.5:
        return
    for h in raw:
        if isinstance(raw[h], dict) and 'prob_up' in raw[h]:
            p = raw[h]['prob_up']
            _update_raw_conviction(raw, h, p * 0.9 + 0.5 * 0.1)
            raw[h]['regime_adjusted'] = True


def _apply_mtf_voting(horizons, model_dir, X_row):
    """Apply multi-timeframe voting to horizons."""
    mtf_path = model_dir / 'mtf_models.pkl'
    if not mtf_path.exists():
        return
    import pickle
    with open(mtf_path, 'rb') as fp:
        mtf_models = pickle.load(fp)
    x_2d = X_row.reshape(1, -1)
    for h in horizons:
        votes, probs = {}, {}
        for win, models in sorted(mtf_models.items()):
            if h in models:
                prob = models[h].predict_proba(x_2d)[0, 1]
                votes[win] = 'BUY' if prob > 0.5 else 'SELL'
                probs[win] = prob
        if not votes:
            continue
        buy_count = sum(1 for v in votes.values() if v == 'BUY')
        total = len(votes)
        avg_prob = float(np.mean(list(probs.values())))
        agreement = max(buy_count, total - buy_count) / total
        horizons[h]['mtf_votes'] = votes
        horizons[h]['mtf_agreement'] = f"{max(buy_count, total - buy_count)}/{total}"
        horizons[h]['mtf_confidence'] = round(agreement * avg_prob, 4)
        vote_str = ', '.join(f"{w}d: {v}" for w, v in sorted(votes.items()))
        majority = 'BUY' if buy_count > total / 2 else 'SELL'
        horizons[h]['mtf_verdict'] = f"{max(buy_count, total - buy_count)}/{total} models say {majority} ({vote_str})"


def _compute_predictability(wf_avg, horizons):
    """Compute predictability assessment from walk-forward accuracy."""
    if not wf_avg:
        return None
    for h in ('short', 'medium', 'long'):
        v = wf_avg.get(h)
        if isinstance(v, (int, float)) and h in horizons:
            if v >= 0.60:
                horizons[h]['predictability'], horizons[h]['predictability_icon'] = 'HIGH', '🟢'
            elif v >= 0.55:
                horizons[h]['predictability'], horizons[h]['predictability_icon'] = 'MODERATE', '🟡'
            else:
                horizons[h]['predictability'], horizons[h]['predictability_icon'] = 'LOW', '⚪'
    avg_wf = np.mean([v for v in wf_avg.values() if isinstance(v, (int, float))])
    if avg_wf >= 0.60:
        return ('HIGH', '🟢', 'Model finds reliable patterns')
    elif avg_wf >= 0.55:
        return ('MODERATE', '🟡', 'Some predictable patterns')
    return ('LOW', '⚪', 'Hard to predict — treat signals with caution')


def _get_model_age(meta):
    """Get model age in days from meta."""
    if not meta or not meta.get('trained_at'):
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(meta['trained_at'])).days
    except (ValueError, TypeError):
        return None


def _apply_conformal_intervals(horizons, model_dir, wf_avg):
    """Apply conformal prediction intervals and re-compute conviction tiers."""
    res_path = model_dir / 'wf_residuals.pkl'
    if res_path.exists():
        import pickle
        try:
            with open(res_path, 'rb') as fp:
                wf_residuals = pickle.load(fp)
            from backend.models.explain import compute_conformal_interval, format_conformal_interval, uncertainty_label
            for h in horizons:
                residuals = wf_residuals.get(h, [])
                if residuals and 'prediction' in horizons[h]:
                    lo, hi = compute_conformal_interval(horizons[h]['prediction'], residuals)
                    horizons[h]['conf_interval'] = (lo, hi)
                    horizons[h]['conf_interval_text'] = format_conformal_interval(lo, hi)
                    horizons[h]['uncertainty'] = uncertainty_label(lo, hi)
        except Exception:
            pass

    from backend.models.explain import get_conviction_tier, format_conviction_verdict
    for h in horizons:
        ci = horizons[h].get('conf_interval')
        ci_width = (ci[1] - ci[0]) if ci else None
        prob = horizons[h].get('confidence', 0.5)
        tier, label, emoji = get_conviction_tier(prob, ci_width)
        horizons[h]['conviction_tier'] = tier
        horizons[h]['conviction_label'] = label
        horizons[h]['conviction_verdict'] = format_conviction_verdict(
            horizons[h]['direction'], prob, wf_avg.get(h))


def _detect_drift(X_row, feature_names, model_dir):
    """Detect feature drift from training statistics."""
    stats_path = model_dir / 'feature_stats.pkl'
    if not stats_path.exists():
        return []
    try:
        import pickle
        with open(stats_path, 'rb') as fp:
            feat_stats = pickle.load(fp)
        from backend.models.explain import detect_feature_drift
        return detect_feature_drift(X_row, feat_stats['means'], feat_stats['stds'], feature_names)
    except Exception:
        return []


def _get_model_health(meta, wf_avg, model_age_days, model_dir):
    """Compute model health, trend, and degradation."""
    if not meta:
        return None, None, None
    from backend.models.explain import compute_model_health, format_health_trend, check_health_degradation
    health = compute_model_health(
        wf_avg, meta.get('brier_scores'), meta.get('has_calibrators', False), model_age_days)
    trend = format_health_trend(model_dir)
    degradation = check_health_degradation(model_dir)
    return health, trend, degradation


def get_analysis(ticker: str) -> dict:
    """Full analysis for a ticker. Returns dict with price, predictions, signals, SHAP, conviction."""
    ticker = ticker.upper()
    ens = _ensure_ticker_model(ticker)
    meta = _load_ticker_meta(ticker)

    try:
        price_df = generate_ohlcv(ticker, "daily", 365)
    except Exception as e:
        _classify_error(e, ticker)
    last_close = float(price_df['close'].iloc[-1])
    spot = _get_current_price(ticker, fallback_close=last_close)
    prev = float(price_df['close'].iloc[-2]) if len(price_df) > 1 else spot
    change_pct = (spot - prev) / prev * 100
    vol = int(price_df['volume'].iloc[-1])
    avg_vol = int(price_df['volume'].tail(20).mean())

    opts = get_options_chain(ticker, spot)
    fund = get_fundamentals(ticker)
    sent = _get_sentiment_safe(ticker)
    macro = get_macro_data()

    # Fetch SPY/sector data
    spy_df, sector_df = _fetch_spy_sector_data(ticker, len(price_df))

    # Check if model was trained with sequence features
    use_seq = meta.get('include_sequence', False) if meta else False
    feats = _store.compute_all_features(price_df, opts, fund, sent, macro, spy_df=spy_df, sector_df=sector_df,
                                         include_sequence=use_seq)

    from backend.features.regime import compute_regime_features, format_regime_display
    regime = compute_regime_features(price_df, macro)
    regime_display = format_regime_display(regime)

    # Apply feature selection if model has it
    X_row = feats.values[-1]
    feature_names = list(feats.columns)
    if ens.selected_features is not None:
        valid_idx = [i for i in ens.selected_features if i < len(X_row)]
        if valid_idx:
            X_row = X_row[valid_idx]
            feature_names = [feature_names[i] for i in valid_idx]
        # If model expects more features than we have, pad with zeros
        expected = len(ens.selected_features)
        if len(X_row) < expected:
            X_row = np.pad(X_row, (0, expected - len(X_row)), constant_values=0)
            feature_names += [f'pad_{i}' for i in range(expected - len(feature_names))]

    # Get historical volatility for z-score
    hist_std = None
    if meta:
        hist_std = meta.get('historical_daily_std')

    raw = ens.predict_all_horizons(X_row, feature_names=feature_names, historical_std=hist_std)

    # Refine predictions with bagged classifiers, calibration, and regime adjustment
    _apply_bagged_predictions(raw, _ticker_model_dir(ticker), X_row)
    _apply_calibration(raw, _ticker_model_dir(ticker))
    _apply_regime_adjustment(raw, regime)

    atr = _gf(feats, 'atr_14') or spot * 0.02

    # Get walk-forward accuracy for conviction display
    wf_avg = meta.get('walk_forward', {}).get('average', {}) if meta else {}
    if not wf_avg and meta:
        wf_avg = meta.get('accuracy', {})

    horizons = _build_horizons(raw, spot, atr, feats)
    # Override short-term with hourly model if available
    hourly_result = _predict_hourly_short(ticker)
    if hourly_result is not None:
        h_pred, h_conf = hourly_result
        horizons['short']['prediction'] = h_pred
        horizons['short']['confidence'] = h_conf
        horizons['short']['direction'] = 'bullish' if h_pred > 0 else 'bearish'

    # Add conviction tiers and SHAP to horizons
    from backend.models.explain import format_conviction_verdict, format_shap_explanation
    for h in horizons:
        r = raw.get(h, {})
        horizons[h]['conviction_tier'] = r.get('conviction_tier', 'LOW')
        horizons[h]['conviction_label'] = r.get('conviction_label', 'low conviction')
        horizons[h]['prob_up'] = r.get('prob_up', 0.5)
        horizons[h]['wf_accuracy'] = wf_avg.get(h)
        horizons[h]['conviction_verdict'] = format_conviction_verdict(
            horizons[h]['direction'], horizons[h]['confidence'], wf_avg.get(h))
        if r.get('shap'):
            horizons[h]['shap'] = r['shap']
            horizons[h]['shap_text'] = format_shap_explanation(r['shap'], horizons[h]['direction'])
        if r.get('vol_zscore') is not None:
            horizons[h]['vol_zscore'] = r['vol_zscore']
            horizons[h]['vol_zscore_desc'] = r.get('vol_zscore_desc', '')
        if r.get('calibrated'):
            horizons[h]['calibrated'] = True
        if r.get('regime_adjusted'):
            horizons[h]['regime_adjusted'] = True

    # Multi-timeframe voting
    _apply_mtf_voting(horizons, _ticker_model_dir(ticker), X_row)

    # Rule-based signals
    rule_bullish, rule_bearish = _extract_signals(feats, spot)
    # Feature importance signals
    fi_bullish, fi_bearish = _feature_importance_signals(ens, feats)

    # Merge: rule-based first, then feature importance (deduplicated)
    all_bullish = rule_bullish + [s for s in fi_bullish if s not in rule_bullish]
    all_bearish = rule_bearish + [s for s in fi_bearish if s not in rule_bearish]

    name, market_cap, sector, pe_ratio = _get_ticker_info(ticker)

    # Save prediction to history
    try:
        from cli.db import save_prediction
        for h in ('short', 'medium', 'long'):
            top_reason = None
            shap_data = horizons[h].get('shap', {})
            bullish_feats = shap_data.get('bullish', [])
            bearish_feats = shap_data.get('bearish', [])
            if horizons[h]['direction'] == 'bullish' and bullish_feats:
                top_reason = bullish_feats[0].get('name', '')
            elif bearish_feats:
                top_reason = bearish_feats[0].get('name', '')
            save_prediction(ticker, horizons[h]['direction'], horizons[h]['confidence'], spot, h,
                            conviction_tier=horizons[h].get('conviction_tier'),
                            top_reason=top_reason)
    except Exception:
        pass

    # Evaluate past prediction accuracy
    prediction_accuracy = None
    try:
        from cli.db import evaluate_predictions
        prediction_accuracy = evaluate_predictions(ticker, spot) or None
    except Exception:
        pass

    # Top feature importances for verbose display
    top_features = _get_top_feature_importances(ens, feats)

    # Predictability assessment
    predictability = _compute_predictability(wf_avg, horizons)

    # Model age
    model_age_days = _get_model_age(meta)

    # Conformal prediction intervals + re-compute conviction tiers
    _apply_conformal_intervals(horizons, _ticker_model_dir(ticker), wf_avg)

    # Feature drift detection
    drift_warnings = _detect_drift(X_row, feature_names, _ticker_model_dir(ticker))

    # Model health
    model_health, health_trend_text, health_degradation = _get_model_health(meta, wf_avg, model_age_days, _ticker_model_dir(ticker))

    # Best horizon recommendation
    from backend.models.explain import best_horizon_recommendation, signal_quality_assessment
    best_horizon = best_horizon_recommendation(wf_avg)

    # Per-horizon signal quality
    for h in horizons:
        drift_count = len(drift_warnings) if drift_warnings else 0
        sq = signal_quality_assessment(
            horizons[h].get('prob_up', 0.5),
            wf_avg.get(h),
            horizons[h].get('conviction_tier', 'LOW'),
            drift_count)
        horizons[h]['signal_quality'] = sq

    return {
        'ticker': ticker, 'name': name, 'price': spot,
        'change_pct': change_pct, 'volume': vol, 'avg_volume': avg_vol,
        'horizons': horizons,
        'bullish': all_bullish[:5] or ['Momentum positive'],
        'bearish': all_bearish[:5] or ['Caution advised'],
        'all_bullish': all_bullish or ['Momentum positive'],
        'all_bearish': all_bearish or ['Caution advised'],
        'regime_display': regime_display,
        'top_features': top_features,
        'predictability': predictability,
        'best_horizon': best_horizon,
        'model_age_days': model_age_days,
        'model_health': model_health,
        'health_trend': health_trend_text,
        'health_degradation': health_degradation,
        'drift_warnings': drift_warnings,
        'brier_scores': meta.get('brier_scores') if meta else None,
        'calibration_curves': meta.get('calibration_curves') if meta else None,
        'prediction_accuracy': prediction_accuracy,
        'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_cap': market_cap, 'sector': sector, 'pe_ratio': pe_ratio,
    }


def get_features(ticker: str) -> dict:
    """Get raw feature values for verbose output."""
    ticker = ticker.upper()
    price_df = generate_ohlcv(ticker, "daily", 365)
    spot = float(price_df['close'].iloc[-1])
    opts = get_options_chain(ticker, spot)
    fund = get_fundamentals(ticker)
    sent = _get_sentiment_safe(ticker)
    macro = get_macro_data()
    feats = _store.compute_all_features(price_df, opts, fund, sent, macro)
    return {col: float(feats[col].iloc[-1]) for col in feats.columns if not feats[col].empty}


def get_price(ticker: str) -> dict:
    """Quick price check, preferring post/pre-market price when available."""
    ticker = ticker.upper()
    df = generate_ohlcv(ticker, "daily", 30)
    last_close = float(df['close'].iloc[-1])
    spot = _get_current_price(ticker, fallback_close=last_close)
    prev = float(df['close'].iloc[-2]) if len(df) > 1 else spot
    return {
        'ticker': ticker, 'price': spot,
        'change': spot - prev, 'change_pct': (spot - prev) / prev * 100,
        'volume': int(df['volume'].iloc[-1]),
        'high': float(df['high'].iloc[-1]), 'low': float(df['low'].iloc[-1]),
    }


def get_price_history(ticker: str, days: int = 30) -> list:
    """Get closing prices for sparkline charts."""
    ticker = ticker.upper()
    df = generate_ohlcv(ticker, "daily", days)
    return [float(x) for x in df['close'].values]


def get_news(ticker: str) -> list:
    """Get recent news with sentiment."""
    sent = _get_sentiment_safe(ticker.upper())
    if sent is None:
        return []
    news_df = sent.get('news')
    if news_df is None or news_df.empty:
        return []
    return news_df.head(10).to_dict('records')


def get_earnings(ticker: str) -> dict:
    """Get earnings info."""
    fund = get_fundamentals(ticker.upper())
    q = fund.get('quarterly')
    val = fund.get('valuation', {})
    latest = fund.get('latest_quarter', {})
    return {'latest': latest, 'valuation': val,
            'quarters': q.head(4).to_dict('records') if q is not None and not q.empty else []}


def chat_query(message: str) -> str:
    """Process natural language query."""
    from backend.llm.intent_parser import IntentParser
    from backend.llm.response_generator import ResponseGenerator
    parser = IntentParser()
    gen = ResponseGenerator()
    intent = parser.parse(message)
    if intent.action == 'help':
        return gen.generate(intent)
    if not intent.ticker:
        return "Please include a stock ticker in your question."
    _ensure_models()
    a = get_analysis(intent.ticker)
    preds = {
        'symbol': a['ticker'], 'current_price': a['price'],
        'horizons': [],
        'bullish_signals': a['bullish'], 'bearish_signals': a['bearish'],
        'predictability': a.get('predictability'),
    }
    for h_key, h_name in [('short', '1-Hour'), ('medium', '5-Day'), ('long', '60-Day')]:
        h = a['horizons'][h_key]
        entry = {
            'name': h_name, 'direction': h['direction'].upper(),
            'confidence': int(h['confidence'] * 100),
            'expected_return': round(h['prediction'] * 100, 2),
            'invalidation': f"Stop ${h['stop']:.2f}",
            'conviction_tier': h.get('conviction_tier', 'LOW'),
            'conviction_verdict': h.get('conviction_verdict', ''),
        }
        if h.get('shap_text'):
            entry['explanation'] = h['shap_text']
        if h.get('vol_zscore') is not None:
            entry['vol_zscore'] = h['vol_zscore']
            entry['vol_zscore_desc'] = h.get('vol_zscore_desc', '')
        if h.get('mtf_verdict'):
            entry['mtf_verdict'] = h['mtf_verdict']
        if h.get('predictability'):
            entry['predictability'] = h['predictability']
        if h.get('conf_interval_text'):
            entry['conf_interval'] = h['conf_interval_text']
        if h.get('uncertainty'):
            entry['uncertainty'] = h['uncertainty']
        sq = h.get('signal_quality')
        if sq:
            entry['signal_quality'] = f"{sq['icon']} {sq['quality']}"
        preds['horizons'].append(entry)
    # Add model health to chat context
    if a.get('model_health'):
        preds['model_health'] = a['model_health']['grade']
    if a.get('health_trend'):
        preds['health_trend'] = a['health_trend']
    if a.get('drift_warnings'):
        preds['drift_warnings'] = len(a['drift_warnings'])
    if a.get('health_degradation'):
        preds['health_degradation'] = a['health_degradation']
    # Add best horizon recommendation
    if a.get('best_horizon'):
        bh = a['best_horizon']
        preds['best_horizon'] = f"{bh['label']} ({bh['accuracy']*100:.0f}% accuracy) — {bh['reason']}"
    # Add past prediction accuracy
    if a.get('prediction_accuracy'):
        parts = []
        for h in ('short', 'medium', 'long'):
            pa = a['prediction_accuracy'].get(h)
            if pa:
                parts.append(f"{h}: {pa['accuracy']*100:.0f}% ({pa['correct']}/{pa['total']})")
        if parts:
            preds['past_accuracy'] = ', '.join(parts)
    return gen.generate(intent, preds)


def screen_tickers(tickers: list, criteria: str = 'oversold') -> list:
    """Screen tickers against criteria. Returns list of matches with details."""
    results = []
    for t in tickers:
        try:
            price_df = generate_ohlcv(t.upper(), "daily", 60)
            from backend.features.tier2_technical import Tier2Technical
            feats = Tier2Technical.compute(price_df)
            last = feats.iloc[-1]
            spot = float(price_df['close'].iloc[-1])
            prev = float(price_df['close'].iloc[-2]) if len(price_df) > 1 else spot
            vol = int(price_df['volume'].iloc[-1])
            avg_vol = int(price_df['volume'].tail(20).mean())
            vol_ratio = vol / max(1, avg_vol)
            rsi = float(last.get('rsi_14', 50))
            stoch = float(last.get('stoch_k', 50))

            match = False
            reason = []
            if criteria == 'oversold':
                if rsi < 30:
                    match, reason = True, reason + [f"RSI {rsi:.1f}"]
                if stoch < 20:
                    match, reason = True, reason + [f"Stoch {stoch:.1f}"]
            elif criteria == 'overbought':
                if rsi > 70:
                    match, reason = True, reason + [f"RSI {rsi:.1f}"]
                if stoch > 80:
                    match, reason = True, reason + [f"Stoch {stoch:.1f}"]
            elif criteria == 'volume':
                if vol_ratio > 2.0:
                    match, reason = True, [f"Vol {vol_ratio:.1f}x avg"]

            if match:
                results.append({
                    'ticker': t.upper(), 'price': spot,
                    'change_pct': (spot - prev) / prev * 100,
                    'rsi': rsi, 'stoch_k': stoch, 'vol_ratio': vol_ratio,
                    'reason': ', '.join(reason),
                })
        except Exception:
            continue
    return results


def get_prediction_history(ticker: str) -> list:
    """Get prediction history for a ticker."""
    from cli.db import get_predictions
    return get_predictions(ticker.upper())


def get_prediction_journal(ticker: str, limit: int = 50) -> dict:
    """Get prediction journal with outcomes for a ticker.

    Returns timeline of predictions with whether they were correct,
    plus summary statistics (accuracy by horizon, by conviction tier,
    confidence distribution histogram).
    """
    from cli.db import get_predictions
    ticker = ticker.upper()
    preds = get_predictions(ticker, limit=limit)
    if not preds:
        return {'ticker': ticker, 'entries': [], 'stats': {}}

    # Try to get current price for evaluation
    current_price = None
    try:
        price_df = generate_ohlcv(ticker, "daily", 5)
        current_price = float(price_df['close'].iloc[-1])
    except Exception:
        pass

    horizon_days = {'short': 1, 'medium': 5, 'long': 20}
    now = datetime.now()
    entries = []
    for p in preds:
        entry = {
            'date': p.get('created_at', ''),
            'horizon': p.get('horizon', ''),
            'direction': p.get('direction', ''),
            'confidence': p.get('confidence', 0),
            'price_at': p.get('price_at', 0),
            'conviction_tier': p.get('conviction_tier', ''),
            'top_reason': p.get('top_reason', ''),
            'outcome': None,
        }
        # Evaluate if enough time has passed
        if current_price and entry['price_at'] > 0:
            try:
                created = datetime.fromisoformat(entry['date'])
                days_needed = horizon_days.get(entry['horizon'], 1)
                if (now - created).days >= days_needed:
                    actual_return = (current_price - entry['price_at']) / entry['price_at']
                    predicted_up = entry['direction'] == 'bullish'
                    actual_up = actual_return > 0
                    entry['outcome'] = 'correct' if predicted_up == actual_up else 'wrong'
                    entry['actual_return'] = round(actual_return * 100, 2)
                else:
                    entry['outcome'] = 'pending'
            except (ValueError, TypeError):
                pass
        entries.append(entry)

    # Summary stats
    stats = {}
    for h in ('short', 'medium', 'long'):
        evaluated = [e for e in entries if e['horizon'] == h and e['outcome'] in ('correct', 'wrong')]
        if evaluated:
            correct = sum(1 for e in evaluated if e['outcome'] == 'correct')
            stats[h] = {'correct': correct, 'total': len(evaluated),
                        'accuracy': round(correct / len(evaluated), 4)}

    # Confidence distribution (histogram buckets)
    conf_values = [e['confidence'] for e in entries if e['confidence']]
    conf_hist = {}
    for bucket_lo in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        bucket_hi = bucket_lo + 0.05
        label = f"{bucket_lo:.0%}-{bucket_hi:.0%}"
        count = sum(1 for c in conf_values if bucket_lo <= c < bucket_hi)
        if count > 0:
            conf_hist[label] = count
    # 80%+ bucket
    high_count = sum(1 for c in conf_values if c >= 0.8)
    if high_count > 0:
        conf_hist['80%+'] = high_count
    stats['confidence_distribution'] = conf_hist

    # Accuracy by conviction tier
    tier_stats = {}
    for tier in ('HIGH', 'MODERATE', 'LOW'):
        evaluated = [e for e in entries if e.get('conviction_tier') == tier and e['outcome'] in ('correct', 'wrong')]
        if evaluated:
            correct = sum(1 for e in evaluated if e['outcome'] == 'correct')
            tier_stats[tier] = {'correct': correct, 'total': len(evaluated),
                                'accuracy': round(correct / len(evaluated), 4)}
    if tier_stats:
        stats['by_conviction'] = tier_stats

    return {'ticker': ticker, 'entries': entries, 'stats': stats}


def correlate_tickers(ticker1: str, ticker2: str, days: int = 180) -> dict:
    """Compute correlation between two tickers."""
    t1, t2 = ticker1.upper(), ticker2.upper()
    df1 = generate_ohlcv(t1, "daily", days)
    df2 = generate_ohlcv(t2, "daily", days)
    r1 = df1['close'].pct_change().dropna()
    r2 = df2['close'].pct_change().dropna()
    # Align by index length (take min)
    n = min(len(r1), len(r2))
    r1, r2 = r1.tail(n).values, r2.tail(n).values
    corr = float(np.corrcoef(r1, r2)[0, 1])
    # Rolling 30-day correlation
    import pandas as pd
    s1 = pd.Series(r1)
    s2 = pd.Series(r2)
    rolling = s1.rolling(30).corr(s2).dropna()
    return {
        'ticker1': t1, 'ticker2': t2, 'days': days,
        'correlation': round(corr, 4),
        'rolling_min': round(float(rolling.min()), 4) if len(rolling) > 0 else corr,
        'rolling_max': round(float(rolling.max()), 4) if len(rolling) > 0 else corr,
        'rolling_current': round(float(rolling.iloc[-1]), 4) if len(rolling) > 0 else corr,
    }


SECTOR_TICKERS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'CRM', 'ADBE'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'AXP'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'COST'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
}


def sector_analysis(sector: str) -> list:
    """Analyze all tickers in a sector. Returns list of analysis summaries."""
    tickers = SECTOR_TICKERS.get(sector, [])
    if not tickers:
        return []
    results = []
    for t in tickers:
        try:
            p = get_price(t)
            from backend.features.tier2_technical import Tier2Technical
            df = generate_ohlcv(t, "daily", 60)
            feats = Tier2Technical.compute(df)
            last = feats.iloc[-1]
            results.append({
                'ticker': t, 'price': p['price'], 'change_pct': p['change_pct'],
                'rsi': float(last.get('rsi_14', 50)),
                'macd_hist': float(last.get('macd_hist', 0)),
                'vol_ratio': p['volume'] / max(1, int(df['volume'].tail(20).mean())),
            })
        except Exception:
            continue
    return results


def prediction_accuracy(ticker: str) -> dict:
    """Compute prediction accuracy stats for a ticker, including by conviction tier."""
    from cli.db import get_predictions
    preds = get_predictions(ticker.upper(), limit=100)
    if not preds:
        return {'total': 0}
    try:
        cur = get_price(ticker)['price']
    except Exception:
        return {'total': len(preds), 'evaluated': 0}
    correct = total_eval = 0
    tier_stats = {}
    for p in preds:
        if not p.get('price_at'):
            continue
        ret = (cur - p['price_at']) / p['price_at']
        is_correct = (p['direction'] == 'bullish' and ret > 0) or (p['direction'] == 'bearish' and ret < 0)
        if is_correct:
            correct += 1
        total_eval += 1
        tier = p.get('conviction_tier')
        if tier:
            if tier not in tier_stats:
                tier_stats[tier] = {'correct': 0, 'total': 0}
            tier_stats[tier]['total'] += 1
            if is_correct:
                tier_stats[tier]['correct'] += 1
    for s in tier_stats.values():
        if s['total'] > 0:
            s['accuracy'] = round(s['correct'] / s['total'], 4)
    result = {
        'total': len(preds), 'evaluated': total_eval,
        'correct': correct, 'accuracy': correct / max(1, total_eval),
    }
    if tier_stats:
        result['by_conviction'] = tier_stats
    return result


def check_alerts() -> list:
    """Check all active alerts against current prices. Returns triggered alerts."""
    from cli.db import get_alerts, trigger_alert
    alerts = get_alerts()
    triggered = []
    for a in alerts:
        try:
            p = get_price(a['ticker'])
            price = p['price']
            hit = False
            if a['condition'] == 'above' and price >= a['threshold']:
                hit = True
            elif a['condition'] == 'below' and price <= a['threshold']:
                hit = True
            if hit:
                trigger_alert(a['id'])
                triggered.append({**a, 'current_price': price})
        except Exception:
            continue
    return triggered


def auto_alerts(ticker: str, mode: str = 'normal') -> list:
    """Auto-set alerts at support/resistance levels for a ticker.
    
    mode: 'normal', 'conservative' (tighter stops), 'aggressive' (wider targets)
    """
    from cli.db import add_alert
    analysis = get_analysis(ticker)
    alerts_set = []
    # Mode multipliers: conservative = tighter, aggressive = wider
    stop_mult = {'conservative': 0.5, 'normal': 1.0, 'aggressive': 1.5}.get(mode, 1.0)
    target_mult = {'conservative': 0.7, 'normal': 1.0, 'aggressive': 2.0}.get(mode, 1.0)
    price = analysis['price']
    for h in ('short', 'medium', 'long'):
        info = analysis['horizons'][h]
        stop = info.get('stop')
        target = info.get('target')
        if stop:
            adj_stop = price - (price - stop) * stop_mult
            add_alert(ticker, 'below', round(adj_stop, 2))
            alerts_set.append({'condition': 'below', 'threshold': round(adj_stop, 2), 'label': f'{h} stop-loss'})
        if target:
            adj_target = price + (target - price) * target_mult
            add_alert(ticker, 'above', round(adj_target, 2))
            alerts_set.append({'condition': 'above', 'threshold': round(adj_target, 2), 'label': f'{h} target'})
    return alerts_set



def _bt_walk_forward_signals(X_all, returns):
    """Generate walk-forward backtest signals using rolling train/test windows."""
    import xgboost as xgb_lib
    import lightgbm as lgb_lib
    bt_train_window, bt_test_window = 252, 21
    n_bt = X_all.shape[0]
    y_cls = (returns[:n_bt] > 0).astype(int)
    sig_arr = np.zeros(n_bt, dtype=int)
    conf_vals = np.full(n_bt, 0.5)

    bt_start = bt_train_window
    while bt_start + bt_test_window <= n_bt:
        bt_end = min(bt_start + bt_test_window, n_bt)
        X_tr = X_all[max(0, bt_start - bt_train_window):bt_start]
        y_tr_cls = y_cls[max(0, bt_start - bt_train_window):bt_start]
        X_te = X_all[bt_start:bt_end]
        if len(np.unique(y_tr_cls)) < 2:
            bt_start += bt_test_window
            continue
        n_pos, n_neg = y_tr_cls.sum(), len(y_tr_cls) - y_tr_cls.sum()
        spw = n_neg / max(n_pos, 1)
        xm = xgb_lib.XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.5, reg_alpha=0.1,
                                     reg_lambda=1.0, min_child_weight=5, gamma=0.0,
                                     scale_pos_weight=spw, eval_metric='logloss',
                                     random_state=42, n_jobs=-1)
        lm = lgb_lib.LGBMClassifier(n_estimators=30, max_depth=2, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.5, reg_alpha=0.1,
                                      reg_lambda=1.0, min_child_weight=5,
                                      scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1)
        xm.fit(X_tr, y_tr_cls)
        lm.fit(X_tr, y_tr_cls)
        prob = 0.5 * xm.predict_proba(X_te)[:, 1] + 0.5 * lm.predict_proba(X_te)[:, 1]
        for j in range(len(prob)):
            idx = bt_start + j
            conf_vals[idx] = prob[j]
            if prob[j] > 0.52: sig_arr[idx] = 1
            elif prob[j] < 0.48: sig_arr[idx] = -1
        bt_start += bt_test_window
    return sig_arr, conf_vals


def _bt_compute_metrics(gross_rets, net_rets, active_mask, signals, position_sizes,
                         slippage, commission, spread):
    """Compute backtest performance metrics from trade returns."""
    gross_returns = gross_rets if len(gross_rets) > 0 else np.array([0.0])
    net_returns = net_rets if len(net_rets) > 0 else np.array([0.0])
    total_trades = int(np.sum(active_mask))
    wins = int(np.sum(net_returns > 0))
    win_rate = wins / total_trades if total_trades > 0 else 0
    sharpe = 0.0
    if len(net_returns) > 1 and np.std(net_returns) > 0:
        sharpe = float(np.sqrt(252) * np.mean(net_returns) / np.std(net_returns))
    cum = np.cumprod(1 + net_returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    gross_wins = float(np.sum(gross_returns[gross_returns > 0]))
    gross_losses = float(abs(np.sum(gross_returns[gross_returns < 0])))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0
    avg_holding = 0.0
    if total_trades > 0:
        sig_np = np.array(signals)
        active = (sig_np != 0).astype(int)
        diffs = np.diff(active, prepend=0, append=0)
        starts, ends = np.where(diffs == 1)[0], np.where(diffs == -1)[0]
        avg_holding = float(np.mean(ends - starts)) if len(starts) > 0 and len(ends) > 0 else 1.0
    return {
        'gross_return': round(float(np.sum(gross_returns)) * 100, 4),
        'net_return': round(float(np.sum(net_returns)) * 100, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_dd, 4),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4),
        'total_trades': total_trades,
        'avg_holding_period': avg_holding,
        'avg_position_size': round(float(np.mean(position_sizes)), 4) if len(position_sizes) > 0 else 0.0,
        'slippage': slippage, 'commission': commission, 'spread': spread,
    }


def _bt_horizon_win_rates(ens, X_all, prices, n):
    """Compute per-horizon win rates using the trained ensemble model."""
    win_rate_by_horizon = {}
    for h, lookahead in [('short', 1), ('medium', 5), ('long', 20)]:
        try:
            directions = []
            for i in range(min(len(X_all), 200)):
                d, prob, _ = ens.predict_direction_ensemble(X_all[i], h)
                conf = prob if d == 'bullish' else 1 - prob
                if conf > 0.55:
                    directions.append((i, 1 if d == 'bullish' else -1))
            if not directions:
                win_rate_by_horizon[h] = 0.5
                continue
            indices = [d[0] for d in directions]
            preds_dir = np.array([d[1] for d in directions])
        except (ValueError, TypeError, AttributeError):
            win_rate_by_horizon[h] = 0.5
            continue
        actuals = np.array([
            (prices[min(i + lookahead, n - 1)] - prices[i]) / prices[i] for i in indices
        ])
        correct = np.sum((preds_dir > 0) == (actuals > 0))
        win_rate_by_horizon[h] = round(float(correct / len(actuals)), 4)
    return win_rate_by_horizon

def run_backtest(ticker: str, days: int = 365, detailed: bool = False,
                 slippage: float = 0.0005, commission: float = 0.0, spread: float = 0.0002) -> dict:
    """Run backtest for a ticker with out-of-sample walk-forward predictions."""
    _ensure_models()
    ticker = ticker.upper()
    ens = _ensure_ticker_model(ticker)
    price_df = generate_ohlcv(ticker, "daily", days)
    spot = float(price_df['close'].iloc[-1])
    opts = get_options_chain(ticker, spot)
    fund = get_fundamentals(ticker)
    sent = _get_sentiment_safe(ticker)
    macro = get_macro_data()
    meta = _load_ticker_meta(ticker)
    use_seq = meta.get('include_sequence', False) if meta else False
    spy_df, sector_df = _fetch_spy_sector_data(ticker, days)
    feats = _store.compute_all_features(price_df, opts, fund, sent, macro,
                                         spy_df=spy_df, sector_df=sector_df,
                                         include_sequence=use_seq)

    prices = price_df['close'].values
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, 0)

    # Cost per round trip
    cost_per_trade = slippage * 2 + spread

    # Apply feature selection if model has it
    n = len(feats)
    X_all = feats.iloc[:n - 1].values
    if ens.selected_features is not None:
        valid_idx = [i for i in ens.selected_features if i < X_all.shape[1]]
        if valid_idx:
            X_all = X_all[:, valid_idx]
        expected = len(ens.selected_features)
        if X_all.shape[1] < expected:
            pad_width = expected - X_all.shape[1]
            X_all = np.pad(X_all, ((0, 0), (0, pad_width)), constant_values=0)

    sig_arr, conf_vals = _bt_walk_forward_signals(X_all, returns)
    signals = sig_arr.tolist()

    active_mask = sig_arr != 0
    active_idx = np.where(active_mask)[0]
    position_sizes = np.clip(np.abs(conf_vals[active_idx] - 0.5) * 2, 0.2, 1.0)
    gross_rets = sig_arr[active_mask] * returns[active_idx] * position_sizes
    net_rets = gross_rets - cost_per_trade * position_sizes

    trades = []
    if detailed:
        for j, i in enumerate(active_idx):
            trades.append({
                'day': int(i), 'price': float(prices[i]),
                'signal': 'BUY' if sig_arr[i] > 0 else 'SELL',
                'return_gross': round(float(gross_rets[j]) * 100, 4),
                'return_net': round(float(net_rets[j]) * 100, 4),
                'cost': round(float(cost_per_trade * position_sizes[j]) * 100, 4),
                'position_size': round(float(position_sizes[j]), 2),
                'confidence': round(float(conf_vals[i]), 4),
            })

    result = _bt_compute_metrics(gross_rets, net_rets, active_mask, signals, position_sizes,
                                  slippage, commission, spread)
    result['win_rate_by_horizon'] = _bt_horizon_win_rates(ens, X_all, prices, n)
    if detailed:
        result['trades'] = trades
    return result


def top_movers(tickers: list) -> list:
    """Get top movers (sorted by absolute change %) from a list of tickers."""
    results = []
    for t in tickers:
        try:
            p = get_price(t)
            results.append(p)
        except Exception:
            continue
    return sorted(results, key=lambda x: abs(x['change_pct']), reverse=True)


def _eval_op(actual: float, op: str, threshold: float) -> bool:
    """Evaluate a comparison operator."""
    if op == '<': return actual < threshold
    if op == '>': return actual > threshold
    if op == '<=': return actual <= threshold
    if op == '>=': return actual >= threshold
    if op == '=': return abs(actual - threshold) < 0.01
    return False


def scan_tickers(tickers: list, filters: str) -> list:
    """Scan tickers with flexible filter expressions like 'rsi<30 AND volume>2x'.
    
    Supported filters: rsi, stoch, macd, bb, adx, volume, change, roc5, roc20, roc60,
    cmf, atrp, psar, adxr, force, cci, trix, ultosc, vortex, williams7, dpo, mass, emv,
    cmo, aroon_up, aroon_down, kst, connors_rsi, choppiness, coppock, elder_bull, elder_bear, rvi,
    supertrend, squeeze, hma, stc, kvo
    """
    import re
    parts = [p.strip() for p in re.split(r'\s+AND\s+', filters, flags=re.IGNORECASE)]
    results = []
    for t in tickers:
        try:
            price_df = generate_ohlcv(t.upper(), "daily", 60)
            from backend.features.tier2_technical import Tier2Technical
            feats = Tier2Technical.compute(price_df)
            last = feats.iloc[-1]
            spot = float(price_df['close'].iloc[-1])
            prev = float(price_df['close'].iloc[-2]) if len(price_df) > 1 else spot
            vol = int(price_df['volume'].iloc[-1])
            avg_vol = int(price_df['volume'].tail(20).mean())
            vol_ratio = vol / max(1, avg_vol)
            change_pct = (spot - prev) / prev * 100

            vals = {
                'rsi': float(last.get('rsi_14', 50)),
                'stoch': float(last.get('stoch_k', 50)),
                'macd': float(last.get('macd_hist', 0)),
                'bb': float(last.get('bb_position', 0.5)),
                'adx': float(last.get('adx_14', 0)),
                'volume': vol_ratio,
                'change': change_pct,
                'roc5': float(last.get('roc_5', 0)),
                'roc20': float(last.get('roc_20', 0)),
                'roc60': float(last.get('roc_60', 0)),
                'cmf': float(last.get('cmf', 0)),
                'atrp': float(last.get('atrp', 0)),
                'psar': float(last.get('psar_dist', 0)),
                'adxr': float(last.get('adxr', 0)),
                'force': float(last.get('force_index', 0)),
                'cci': float(last.get('cci', 0)),
                'trix': float(last.get('trix', 0)),
                'ultosc': float(last.get('ultimate_osc', 50)),
                'vortex': float(last.get('vortex', 0)),
                'williams7': float(last.get('williams_r_7', -50)),
                'dpo': float(last.get('dpo', 0)),
                'mass': float(last.get('mass_index', 25)),
                'emv': float(last.get('emv', 0)),
                'cmo': float(last.get('cmo', 0)),
                'aroon_up': float(last.get('aroon_up', 50)),
                'aroon_down': float(last.get('aroon_down', 50)),
                'kst': float(last.get('kst', 0)),
                'connors_rsi': float(last.get('connors_rsi', 50)),
                'choppiness': float(last.get('choppiness', 50)),
                'coppock': float(last.get('coppock_curve', 0)),
                'elder_bull': float(last.get('elder_bull', 0)),
                'elder_bear': float(last.get('elder_bear', 0)),
                'rvi': float(last.get('rvi', 0)),
                'supertrend': float(last.get('supertrend', 0)),
                'squeeze': float(last.get('squeeze_mom', 0)),
                'hma': float(last.get('hma_dist', 0)),
                'stc': float(last.get('stc', 50)),
                'kvo': float(last.get('kvo', 0)),
            }

            match = True
            matched_reasons = []
            for p in parts:
                m = re.match(r'(\w+)\s*([<>]=?|=)\s*(-?[\d.]+)(%|x)?', p)
                if not m:
                    continue
                key, op, val_str, _suffix = m.group(1).lower(), m.group(2), float(m.group(3)), m.group(4)
                actual = vals.get(key)
                if actual is None:
                    match = False
                    break
                if not _eval_op(actual, op, val_str):
                    match = False; break
                matched_reasons.append(f"{key}={actual:.1f}")

            if match and matched_reasons:
                results.append({
                    'ticker': t.upper(), 'price': spot, 'change_pct': change_pct,
                    'rsi': vals['rsi'], 'stoch_k': vals['stoch'], 'vol_ratio': vol_ratio,
                    'reason': ', '.join(matched_reasons),
                })
        except Exception:
            continue
    return results


def portfolio_risk(positions: list) -> dict:
    """Compute portfolio-level risk metrics: VaR, beta, correlation matrix."""
    if not positions:
        return {'var_95': 0, 'beta': 0, 'total_value': 0, 'tickers': []}
    tickers = [p['ticker'] for p in positions]
    weights = []
    returns_list = []
    total_value = 0
    for p in positions:
        try:
            pr = get_price(p['ticker'])
            val = pr['price'] * p['qty']
        except Exception:
            val = p['entry_price'] * p['qty']
        total_value += val
        weights.append(val)

    if total_value == 0:
        return {'var_95': 0, 'beta': 0, 'total_value': 0, 'tickers': tickers}

    weights = np.array(weights) / total_value

    # Fetch returns for each ticker
    for t in tickers:
        try:
            df = generate_ohlcv(t, "daily", 60)
            r = df['close'].pct_change().dropna().values
            returns_list.append(r)
        except Exception:
            returns_list.append(np.zeros(58))

    # Align lengths
    min_len = min(len(r) for r in returns_list)
    returns_arr = np.column_stack([r[-min_len:] for r in returns_list])

    # Portfolio returns
    port_returns = returns_arr @ weights

    # VaR 95%
    var_95 = float(np.percentile(port_returns, 5)) * total_value

    # Beta vs SPY
    try:
        spy_df = generate_ohlcv("SPY", "daily", 60)
        spy_r = spy_df['close'].pct_change().dropna().values[-min_len:]
        cov = np.cov(port_returns, spy_r)
        beta = float(cov[0, 1] / (cov[1, 1] + 1e-8))
    except Exception:
        beta = 1.0

    # Volatility
    vol = float(np.std(port_returns) * np.sqrt(252))

    # Sharpe ratio (assume risk-free rate ~4.5%)
    ann_return = float(np.mean(port_returns) * 252)
    sharpe = (ann_return - 0.045) / (vol + 1e-8)

    return {
        'var_95': round(var_95, 2),
        'beta': round(beta, 2),
        'volatility': round(vol, 4),
        'sharpe': round(sharpe, 2),
        'annualized_return': round(ann_return, 4),
        'total_value': round(total_value, 2),
        'tickers': tickers,
        'weights': {t: round(float(w), 4) for t, w in zip(tickers, weights)},
    }


def portfolio_optimize(positions: list) -> dict:
    """Suggest optimized weights using min-variance and max-Sharpe heuristics."""
    if len(positions) < 2:
        return {'error': 'Need at least 2 positions to optimize'}
    tickers = [p['ticker'] for p in positions]
    returns_list = []
    for t in tickers:
        try:
            df = generate_ohlcv(t, "daily", 120)
            r = df['close'].pct_change().dropna().values
            returns_list.append(r)
        except Exception:
            returns_list.append(np.zeros(118))
    min_len = min(len(r) for r in returns_list)
    returns_arr = np.column_stack([r[-min_len:] for r in returns_list])
    mean_ret = np.mean(returns_arr, axis=0) * 252
    cov = np.cov(returns_arr.T) * 252
    n = len(tickers)

    # Current weights
    total = sum(p['entry_price'] * p['qty'] for p in positions)
    cur_w = np.array([p['entry_price'] * p['qty'] / (total + 1e-8) for p in positions])

    # Min-variance: w = inv(cov) * 1 / (1' * inv(cov) * 1)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-6)
        ones = np.ones(n)
        mv_w = inv_cov @ ones / (ones @ inv_cov @ ones)
        mv_w = np.clip(mv_w, 0, 1)
        mv_w /= mv_w.sum()
    except Exception:
        mv_w = np.ones(n) / n

    # Max-Sharpe heuristic: w proportional to return/vol
    vols = np.sqrt(np.diag(cov)) + 1e-8
    sharpe_ratios = (mean_ret - 0.045) / vols
    sr_pos = np.clip(sharpe_ratios, 0.01, None)
    ms_w = sr_pos / sr_pos.sum()

    # Equal weight
    eq_w = np.ones(n) / n

    def port_stats(w: np.ndarray) -> dict[str, float]:
        ret = float(w @ mean_ret)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - 0.045) / (vol + 1e-8)
        return {'return': round(ret, 4), 'volatility': round(vol, 4), 'sharpe': round(sharpe, 2)}

    return {
        'tickers': tickers,
        'current': {'weights': {t: round(float(w), 4) for t, w in zip(tickers, cur_w)}, **port_stats(cur_w)},
        'min_variance': {'weights': {t: round(float(w), 4) for t, w in zip(tickers, mv_w)}, **port_stats(mv_w)},
        'max_sharpe': {'weights': {t: round(float(w), 4) for t, w in zip(tickers, ms_w)}, **port_stats(ms_w)},
        'equal_weight': {'weights': {t: round(float(w), 4) for t, w in zip(tickers, eq_w)}, **port_stats(eq_w)},
    }


def replay_analysis(ticker: str, date_str: str) -> dict:
    """Replay analysis as of a historical date using per-ticker model with full ML features."""
    from datetime import datetime as _dt
    ticker = ticker.upper()
    target = _dt.strptime(date_str, '%Y-%m-%d')
    days_ago = (_dt.now() - target).days + 365
    price_df = generate_ohlcv(ticker, "daily", days_ago)
    price_df.index = pd.to_datetime(price_df.index) if not hasattr(price_df.index, 'date') else price_df.index
    mask = price_df.index <= pd.Timestamp(target)
    hist = price_df[mask]
    if len(hist) < 30:
        raise ValueError(f"Not enough data before {date_str} for {ticker}")
    spot = float(hist['close'].iloc[-1])
    prev = float(hist['close'].iloc[-2])
    change_pct = (spot - prev) / prev * 100
    vol = int(hist['volume'].iloc[-1])
    avg_vol = int(hist['volume'].tail(20).mean())

    # Use per-ticker model if available, else fall back to global
    meta = _load_ticker_meta(ticker)
    has_ticker_model = meta is not None and _ticker_model_dir(ticker).exists()

    opts = get_options_chain(ticker, spot)
    fund = get_fundamentals(ticker)
    sent = _get_sentiment_safe(ticker)
    macro = get_macro_data()

    if has_ticker_model:
        ens = _ensure_ticker_model(ticker)
        spy_df, sector_df = _fetch_spy_sector_data(ticker, len(hist))
        use_seq = meta.get('include_sequence', False)
        feats = _store.compute_all_features(hist, opts, fund, sent, macro,
                                             spy_df=spy_df, sector_df=sector_df,
                                             include_sequence=use_seq)
        X_row = feats.values[-1]
        feature_names = list(feats.columns)
        if ens.selected_features is not None:
            valid_idx = [i for i in ens.selected_features if i < len(X_row)]
            if valid_idx:
                X_row = X_row[valid_idx]
                feature_names = [feature_names[i] for i in valid_idx]
            expected = len(ens.selected_features)
            if len(X_row) < expected:
                X_row = np.pad(X_row, (0, expected - len(X_row)), constant_values=0)
                feature_names += [f'pad_{i}' for i in range(expected - len(feature_names))]
        hist_std = meta.get('historical_daily_std')
        raw = ens.predict_all_horizons(X_row, feature_names=feature_names, historical_std=hist_std)
        wf_avg = meta.get('walk_forward', {}).get('average', {})
    else:
        _ensure_models()
        feats = _store.compute_all_features(hist, opts, fund, sent, macro)
        raw = _ensemble.predict_all_horizons(feats.values)
        wf_avg = {}

    atr = _gf(feats, 'atr_14') or spot * 0.02

    from backend.models.explain import format_conviction_verdict, format_shap_explanation, best_horizon_recommendation
    horizons = {}
    for h in ('short', 'medium', 'long'):
        p = raw[h]
        inv = calculate_invalidation(spot, p['direction'], h, atr)
        d = {
            'prediction': float(p['prediction']),
            'confidence': float(p['confidence']),
            'direction': p['direction'],
            'stop': inv['stop_loss'],
            'target': inv['take_profit'],
            'conviction_tier': p.get('conviction_tier', 'LOW'),
            'conviction_label': p.get('conviction_label', 'low conviction'),
            'conviction_verdict': format_conviction_verdict(
                p['direction'], p['confidence'], wf_avg.get(h)),
        }
        if p.get('shap'):
            d['shap'] = p['shap']
            d['shap_text'] = format_shap_explanation(p['shap'], p['direction'])
        if p.get('vol_zscore') is not None:
            d['vol_zscore'] = p['vol_zscore']
            d['vol_zscore_desc'] = p.get('vol_zscore_desc', '')
        horizons[h] = d

    # Check actual outcome if we have future data
    future = price_df[price_df.index > pd.Timestamp(target)]
    outcome = {}
    if len(future) > 0:
        for h, days in [('short', 1), ('medium', 5), ('long', 20)]:
            idx = min(days, len(future) - 1)
            future_price = float(future['close'].iloc[idx])
            actual_ret = (future_price - spot) / spot * 100
            predicted_dir = horizons[h]['direction']
            correct = (predicted_dir == 'bullish' and actual_ret > 0) or \
                      (predicted_dir == 'bearish' and actual_ret < 0)
            outcome[h] = {'actual_return': round(actual_ret, 2), 'correct': correct, 'future_price': future_price}

    return {
        'ticker': ticker, 'date': date_str, 'price': spot,
        'change_pct': change_pct, 'volume': vol, 'avg_volume': avg_vol,
        'horizons': horizons, 'outcome': outcome,
        'has_ticker_model': has_ticker_model,
        'best_horizon': best_horizon_recommendation(wf_avg) if wf_avg else None,
    }


def replay_range(ticker: str, start: str, end: str) -> list:
    """Replay analysis across a date range. Returns list of replay results."""
    from datetime import datetime, timedelta
    s = datetime.strptime(start, '%Y-%m-%d')
    e = datetime.strptime(end, '%Y-%m-%d')
    results = []
    current = s
    while current <= e:
        date_str = current.strftime('%Y-%m-%d')
        try:
            r = replay_analysis(ticker, date_str)
            results.append(r)
        except Exception:
            pass  # skip dates with no data (weekends/holidays)
        current += timedelta(days=1)
    return results


def portfolio_rebalance(positions: list, strategy: str = 'max_sharpe') -> list:
    """Compute trades needed to rebalance portfolio to target weights.
    
    Returns list of {ticker, action, shares, value} dicts.
    """
    if len(positions) < 2:
        return []
    opt = portfolio_optimize(positions)
    if 'error' in opt:
        return []
    target_weights = opt.get(strategy, opt.get('max_sharpe', {})).get('weights', {})
    if not target_weights:
        return []
    # Get current values
    total_value = 0
    current = {}
    for p in positions:
        try:
            pr = get_price(p['ticker'])['price']
        except Exception:
            pr = p['entry_price']
        val = pr * p['qty']
        current[p['ticker']] = {'price': pr, 'qty': p['qty'], 'value': val}
        total_value += val

    trades = []
    for t, tw in target_weights.items():
        target_val = tw * total_value
        cur = current.get(t, {'price': 0, 'qty': 0, 'value': 0})
        diff_val = target_val - cur['value']
        if abs(diff_val) < 1:
            continue
        diff_shares = diff_val / cur['price'] if cur['price'] > 0 else 0
        trades.append({
            'ticker': t,
            'action': 'BUY' if diff_val > 0 else 'SELL',
            'shares': round(abs(diff_shares), 2),
            'value': round(abs(diff_val), 2),
            'current_weight': round(cur['value'] / (total_value + 1e-8), 4),
            'target_weight': round(tw, 4),
        })
    return sorted(trades, key=lambda x: x['value'], reverse=True)


def momentum_ranking(tickers: list) -> list:
    """Rank tickers by multi-timeframe momentum score.
    
    Combines ROC(5), ROC(20), ROC(60), Connors RSI, and ADX into a composite score.
    """
    results = []
    for t in tickers:
        try:
            price_df = generate_ohlcv(t.upper(), "daily", 120)
            from backend.features.tier2_technical import Tier2Technical
            feats = Tier2Technical.compute(price_df)
            last = feats.iloc[-1]
            spot = float(price_df['close'].iloc[-1])
            prev = float(price_df['close'].iloc[-2]) if len(price_df) > 1 else spot
            roc5 = float(last.get('roc_5', 0))
            roc20 = float(last.get('roc_20', 0))
            roc60 = float(last.get('roc_60', 0))
            crsi = float(last.get('connors_rsi', 50))
            adx = float(last.get('adx_14', 0))
            chop = float(last.get('choppiness', 50))
            # Composite: weight short-term less, long-term more
            score = roc5 * 0.15 + roc20 * 0.25 + roc60 * 0.3 + (crsi - 50) * 0.15 + (adx - 25) * 0.15
            trending = chop < 50
            results.append({
                'ticker': t.upper(), 'price': spot,
                'change_pct': (spot - prev) / prev * 100,
                'roc_5': roc5, 'roc_20': roc20, 'roc_60': roc60,
                'connors_rsi': crsi, 'adx': adx, 'choppiness': chop,
                'score': round(score, 2), 'trending': trending,
            })
        except Exception:
            continue
    return sorted(results, key=lambda x: x['score'], reverse=True)


def portfolio_tax_report(positions: list, sold_positions: list = None) -> dict:
    """Generate capital gains tax report for portfolio positions.
    
    Returns dict with realized gains (from sold), unrealized gains (from current),
    and summary totals.
    """
    from cli.db import get_db
    realized = []
    unrealized = []
    
    # Unrealized gains from current positions
    for p in positions:
        try:
            pr = get_price(p['ticker'])['price']
        except Exception:
            pr = p.get('entry_price', 0)
        cost_basis = p['entry_price'] * p['qty']
        current_val = pr * p['qty']
        gain = current_val - cost_basis
        pct = (gain / cost_basis * 100) if cost_basis > 0 else 0
        # Determine short-term vs long-term (>1 year)
        from datetime import datetime
        added = p.get('added_at', '')
        if added:
            try:
                days_held = (datetime.now() - datetime.fromisoformat(added)).days
            except Exception:
                days_held = 0
        else:
            days_held = 0
        term = 'long' if days_held > 365 else 'short'
        unrealized.append({
            'ticker': p['ticker'], 'qty': p['qty'],
            'cost_basis': round(cost_basis, 2), 'current_value': round(current_val, 2),
            'gain': round(gain, 2), 'gain_pct': round(pct, 2),
            'term': term, 'days_held': days_held,
        })
    
    # Realized gains from sold positions (from DB)
    try:
        db = get_db()
        rows = db.execute(
            "SELECT ticker, entry_price, sell_price, qty, sold_at, added_at FROM sold_positions ORDER BY sold_at DESC"
        ).fetchall()
        for r in rows:
            cost = r[1] * r[3]
            proceeds = r[2] * r[3]
            gain = proceeds - cost
            try:
                days_held = (datetime.fromisoformat(r[4]) - datetime.fromisoformat(r[5])).days
            except Exception:
                days_held = 0
            realized.append({
                'ticker': r[0], 'qty': r[3],
                'cost_basis': round(cost, 2), 'proceeds': round(proceeds, 2),
                'gain': round(gain, 2), 'term': 'long' if days_held > 365 else 'short',
            })
    except Exception:
        pass  # sold_positions table may not exist yet
    
    total_unrealized = sum(u['gain'] for u in unrealized)
    total_realized = sum(r['gain'] for r in realized)
    short_gains = sum(r['gain'] for r in realized if r['term'] == 'short')
    long_gains = sum(r['gain'] for r in realized if r['term'] == 'long')
    
    return {
        'unrealized': unrealized, 'realized': realized,
        'total_unrealized': round(total_unrealized, 2),
        'total_realized': round(total_realized, 2),
        'short_term_gains': round(short_gains, 2),
        'long_term_gains': round(long_gains, 2),
    }


def portfolio_correlation(positions: list) -> dict:
    """Compute correlation matrix for portfolio positions."""
    if len(positions) < 2:
        return {'tickers': [p['ticker'] for p in positions], 'matrix': [[1.0]], 'pairs': []}
    tickers = [p['ticker'] for p in positions]
    import pandas as pd
    closes = {}
    for t in tickers:
        try:
            df_t = generate_ohlcv(t, "daily", 90)
            closes[t] = df_t['close'].values
        except Exception:
            continue
    if len(closes) < 2:
        return {'tickers': list(closes.keys()), 'matrix': [[1.0]], 'pairs': []}
    min_len = min(len(v) for v in closes.values())
    df = pd.DataFrame({t: v[-min_len:] for t, v in closes.items()}).pct_change().dropna()
    corr = df.corr()
    pairs = []
    done = set()
    for i, t1 in enumerate(corr.columns):
        for j, t2 in enumerate(corr.columns):
            if i < j and (t1, t2) not in done:
                pairs.append({'ticker1': t1, 'ticker2': t2, 'correlation': round(float(corr.iloc[i, j]), 3)})
                done.add((t1, t2))
    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return {'tickers': list(corr.columns), 'matrix': corr.values.round(3).tolist(), 'pairs': pairs}


def smart_alerts(ticker: str) -> list:
    """Auto-generate alerts based on key technical levels (support, resistance, Bollinger, pivots)."""
    from cli.db import add_alert
    analysis = get_analysis(ticker)
    price = analysis['price']
    feats = get_features(ticker)
    alerts_set = []

    # Bollinger Band alerts
    bb_upper = feats.get('bb_upper', 0)
    bb_lower = feats.get('bb_lower', 0)
    if bb_upper > price:
        add_alert(ticker, 'above', round(bb_upper, 2))
        alerts_set.append({'condition': 'above', 'threshold': round(bb_upper, 2), 'label': 'BB upper band'})
    if bb_lower > 0 and bb_lower < price:
        add_alert(ticker, 'below', round(bb_lower, 2))
        alerts_set.append({'condition': 'below', 'threshold': round(bb_lower, 2), 'label': 'BB lower band'})

    # Support/resistance from horizons
    for h in ('short', 'medium', 'long'):
        info = analysis['horizons'][h]
        for key, cond, label in [('support', 'below', 'support'), ('resistance', 'above', 'resistance'),
                                  ('stop', 'below', 'stop-loss'), ('target', 'above', 'target')]:
            val = info.get(key)
            if val and val != price:
                add_alert(ticker, cond, round(val, 2))
                alerts_set.append({'condition': cond, 'threshold': round(val, 2), 'label': f'{h} {label}'})

    return alerts_set


def portfolio_diversification(positions: list) -> dict:
    """Calculate portfolio diversification score based on sector spread and correlation."""
    import yfinance as yf
    if not positions:
        return {'score': 0, 'grade': 'N/A', 'sectors': {}, 'suggestions': ['Add positions first']}
    
    sectors = {}
    for p in positions:
        try:
            info = yf.Ticker(p['ticker']).info
            sec = info.get('sector', 'Unknown')
        except Exception:
            sec = 'Unknown'
        sectors[sec] = sectors.get(sec, 0) + 1
    
    n_sectors = len(sectors)
    n_positions = len(positions)
    # Herfindahl index: lower = more diversified
    weights = [c / n_positions for c in sectors.values()]
    hhi = sum(w ** 2 for w in weights)
    # Score: 100 = perfectly diversified, 0 = single stock
    score = max(0, min(100, int((1 - hhi) * 100 * min(n_sectors / 5, 1))))
    
    if score >= 80: grade = 'A'
    elif score >= 60: grade = 'B'
    elif score >= 40: grade = 'C'
    elif score >= 20: grade = 'D'
    else: grade = 'F'
    
    suggestions = []
    if n_sectors < 3:
        suggestions.append(f'Only {n_sectors} sector(s) — add positions in different sectors')
    max_sector = max(sectors.items(), key=lambda x: x[1])
    if max_sector[1] / n_positions > 0.5:
        suggestions.append(f'{max_sector[0]} is {max_sector[1]}/{n_positions} positions — reduce concentration')
    if n_positions < 5:
        suggestions.append('Consider adding more positions for better diversification')
    
    return {'score': score, 'grade': grade, 'sectors': sectors, 'suggestions': suggestions}


STRESS_SCENARIOS = {
    'market_crash': {'label': 'Market Crash (-20%)', 'shock': -0.20},
    'correction': {'label': 'Correction (-10%)', 'shock': -0.10},
    'rate_hike': {'label': 'Rate Hike (-5%)', 'shock': -0.05},
    'rally': {'label': 'Bull Rally (+15%)', 'shock': 0.15},
    'black_swan': {'label': 'Black Swan (-35%)', 'shock': -0.35},
}


def portfolio_stress_test(positions: list, scenario: str = None) -> list:
    """Simulate portfolio impact under stress scenarios."""
    scenarios = {scenario: STRESS_SCENARIOS[scenario]} if scenario else STRESS_SCENARIOS
    results = []
    for key, s in scenarios.items():
        total_before = 0
        total_after = 0
        details = []
        for p in positions:
            try:
                pr = get_price(p['ticker'])
                cur = pr['price']
            except Exception:
                cur = p['entry_price']
            val = cur * p['qty']
            # Apply beta-adjusted shock (higher beta = more impact)
            try:
                df = generate_ohlcv(p['ticker'], "daily", 60)
                spy_df = generate_ohlcv("SPY", "daily", 60)
                r = df['close'].pct_change().dropna().values
                spy_r = spy_df['close'].pct_change().dropna().values
                min_l = min(len(r), len(spy_r))
                cov = np.cov(r[-min_l:], spy_r[-min_l:])
                beta = cov[0, 1] / (cov[1, 1] + 1e-8)
            except Exception:
                beta = 1.0
            adj_shock = s['shock'] * max(0.5, min(2.0, beta))
            new_val = val * (1 + adj_shock)
            total_before += val
            total_after += new_val
            details.append({'ticker': p['ticker'], 'before': round(val, 2),
                           'after': round(new_val, 2), 'impact': round(new_val - val, 2),
                           'beta': round(float(beta), 2)})
        results.append({
            'scenario': key, 'label': s['label'],
            'total_before': round(total_before, 2), 'total_after': round(total_after, 2),
            'total_impact': round(total_after - total_before, 2),
            'impact_pct': round((total_after - total_before) / (total_before + 1e-8) * 100, 1),
            'details': details,
        })
    return results


def portfolio_performance(positions: list) -> dict:
    """Calculate portfolio performance metrics: total value, daily/weekly/monthly returns."""
    total_value = 0
    total_cost = 0
    holdings = []
    for p in positions:
        try:
            pr = get_price(p['ticker'])
            cur = pr['price']
        except Exception:
            cur = p['entry_price']
        val = cur * p['qty']
        cost = p['entry_price'] * p['qty']
        total_value += val
        total_cost += cost
        # Get historical prices for period returns
        try:
            df = generate_ohlcv(p['ticker'], "daily", 30)
            closes = df['close'].tolist()
            d1 = closes[-2] if len(closes) >= 2 else cur
            w1 = closes[-5] if len(closes) >= 5 else cur
            m1 = closes[0] if len(closes) >= 20 else cur
        except Exception:
            d1 = w1 = m1 = cur
        holdings.append({
            'ticker': p['ticker'], 'qty': p['qty'], 'entry': p['entry_price'],
            'current': cur, 'value': round(val, 2), 'cost': round(cost, 2),
            'total_return_pct': round((cur - p['entry_price']) / p['entry_price'] * 100, 2),
            'daily_return_pct': round((cur - d1) / (d1 + 1e-8) * 100, 2),
            'weekly_return_pct': round((cur - w1) / (w1 + 1e-8) * 100, 2),
            'monthly_return_pct': round((cur - m1) / (m1 + 1e-8) * 100, 2),
        })
    total_return_pct = round((total_value - total_cost) / (total_cost + 1e-8) * 100, 2)
    return {
        'total_value': round(total_value, 2),
        'total_cost': round(total_cost, 2),
        'total_pnl': round(total_value - total_cost, 2),
        'total_return_pct': total_return_pct,
        'holdings': holdings,
    }


def feature_drift_analysis(ticker: str) -> dict:
    """Analyze feature drift for a ticker: current values vs training distribution."""
    ticker = ticker.upper()
    meta = _load_ticker_meta(ticker)
    if not meta:
        return {'error': f'No trained model for {ticker}'}
    train_means = np.array(meta.get('feature_means', []))
    train_stds = np.array(meta.get('feature_stds', []))
    feat_names = meta.get('feature_names', [])
    if len(train_means) == 0 or len(train_stds) == 0:
        return {'error': 'No training statistics saved — retrain model first'}

    # Get current features
    try:
        price_df = generate_ohlcv(ticker, "daily", 120)
        spot = float(price_df['close'].iloc[-1])
        opts = get_options_chain(ticker, spot)
        fund = get_fundamentals(ticker)
        sent = _get_sentiment_safe(ticker)
        macro = get_macro_data()
        spy_df, sector_df = _fetch_spy_sector_data(ticker, 120)
        feats = _store.compute_all_features(price_df, opts, fund, sent, macro,
                                             spy_df=spy_df, sector_df=sector_df)
    except Exception as e:
        return {'error': f'Failed to compute features: {e}'}

    current = feats.iloc[-1].values
    n = min(len(current), len(train_means), len(train_stds))
    safe_stds = np.where(train_stds[:n] > 1e-8, train_stds[:n], 1.0)
    z_scores = (current[:n] - train_means[:n]) / safe_stds

    # All features with z-scores
    all_drift = []
    for i in range(n):
        name = feat_names[i] if i < len(feat_names) else f'feature_{i}'
        all_drift.append({
            'feature': name,
            'z_score': round(float(z_scores[i]), 2),
            'current': round(float(current[i]), 4),
            'train_mean': round(float(train_means[i]), 4),
            'train_std': round(float(train_stds[i]), 4),
        })

    # Sort by absolute z-score
    all_drift.sort(key=lambda x: abs(x['z_score']), reverse=True)

    # Summary stats
    abs_z = np.abs(z_scores)
    n_severe = int(np.sum(abs_z > 3.0))
    n_moderate = int(np.sum((abs_z > 2.0) & (abs_z <= 3.0)))
    n_mild = int(np.sum((abs_z > 1.0) & (abs_z <= 2.0)))
    avg_drift = round(float(np.mean(abs_z)), 2)

    # Drift risk level
    if n_severe > 10 or avg_drift > 2.0:
        risk = 'HIGH'
        risk_desc = 'Significant distribution shift — predictions may be unreliable'
    elif n_severe > 3 or avg_drift > 1.0:
        risk = 'MODERATE'
        risk_desc = 'Some features outside training range — monitor closely'
    else:
        risk = 'LOW'
        risk_desc = 'Features within expected range'

    return {
        'ticker': ticker,
        'total_features': n,
        'severe_drift': n_severe,
        'moderate_drift': n_moderate,
        'mild_drift': n_mild,
        'avg_drift': avg_drift,
        'risk': risk,
        'risk_desc': risk_desc,
        'top_drifted': all_drift[:15],
        'trained_at': meta.get('trained_at'),
    }


def backtest_compare(ticker: str, days: int = 365) -> dict:
    """Run backtest and compare with saved baseline (if any)."""
    ticker = ticker.upper()
    model_dir = _ticker_model_dir(ticker)
    baseline_path = model_dir / 'backtest_baseline.json'

    # Run current backtest
    current = run_backtest(ticker, days=days)

    # Load baseline if exists
    baseline = None
    if baseline_path.exists():
        import json as _json
        try:
            baseline = _json.loads(baseline_path.read_text())
        except Exception:
            pass

    # Save current as new baseline
    import json as _json
    from datetime import datetime
    save_data = {**current, 'saved_at': datetime.now().isoformat()}
    model_dir.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(_json.dumps(save_data))

    if baseline is None:
        return {'ticker': ticker, 'current': current, 'baseline': None, 'comparison': None}

    # Compare metrics
    metrics = ['net_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']
    comparison = {}
    for m in metrics:
        cur_val = current.get(m, 0)
        base_val = baseline.get(m, 0)
        delta = cur_val - base_val
        # For max_drawdown, lower is better
        improved = delta > 0 if m != 'max_drawdown' else delta < 0
        comparison[m] = {
            'current': cur_val,
            'baseline': base_val,
            'delta': round(delta, 4),
            'improved': improved,
        }

    return {
        'ticker': ticker,
        'current': current,
        'baseline': {'saved_at': baseline.get('saved_at'), **{m: baseline.get(m, 0) for m in metrics}},
        'comparison': comparison,
    }


def feature_interactions(ticker: str, top_n: int = 10) -> dict:
    """Analyze top feature interaction pairs using SHAP interaction values."""
    ticker = ticker.upper()
    meta = _load_ticker_meta(ticker)
    if not meta:
        return {'error': f'No trained model for {ticker}'}
    ens = _get_ticker_ensemble(ticker)
    if not ens.loaded:
        return {'error': f'No trained model for {ticker}'}

    feat_names = meta.get('feature_names', [])

    # Get current features
    try:
        price_df = generate_ohlcv(ticker, "daily", 120)
        spot = float(price_df['close'].iloc[-1])
        opts = get_options_chain(ticker, spot)
        fund = get_fundamentals(ticker)
        sent = _get_sentiment_safe(ticker)
        macro = get_macro_data()
        spy_df, sector_df = _fetch_spy_sector_data(ticker, 120)
        feats = _store.compute_all_features(price_df, opts, fund, sent, macro,
                                             spy_df=spy_df, sector_df=sector_df)
    except Exception as e:
        return {'error': f'Failed to compute features: {e}'}

    X = feats.iloc[-min(50, len(feats)):].values
    if ens.selected_features is not None:
        valid_idx = [i for i in ens.selected_features if i < X.shape[1]]
        if valid_idx:
            X = X[:, valid_idx]
            feat_names = [feat_names[i] for i in valid_idx if i < len(feat_names)]

    # Use XGB model for SHAP interaction values
    model = None
    for m in [ens.xgb_short, ens.xgb_medium, ens.xgb_long]:
        if m is not None:
            model = m
            break
    if model is None:
        return {'error': 'No XGBoost model available'}

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        interaction_values = explainer.shap_interaction_values(X[:20])
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1] if len(interaction_values) > 1 else interaction_values[0]
    except Exception:
        # Fallback: use feature importance correlation
        return _feature_importance_correlation(ens, X, feat_names, top_n)

    # Average absolute interaction values across samples
    n_feats = interaction_values.shape[1]
    avg_interactions = np.mean(np.abs(interaction_values), axis=0)

    # Extract top off-diagonal interactions
    pairs = []
    for i in range(min(n_feats, len(feat_names))):
        for j in range(i + 1, min(n_feats, len(feat_names))):
            pairs.append((feat_names[i], feat_names[j], float(avg_interactions[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    from backend.models.explain import _readable_name
    top_pairs = [{
        'feature_1': _readable_name(p[0]),
        'feature_2': _readable_name(p[1]),
        'interaction_strength': round(p[2], 6),
        'raw_names': [p[0], p[1]],
    } for p in pairs[:top_n]]

    return {'ticker': ticker, 'interactions': top_pairs, 'method': 'shap_interaction'}


def _feature_importance_correlation(ens, X, feat_names, top_n):
    """Fallback: find correlated important features."""
    from backend.models.explain import _readable_name
    importances = {}
    for model in [ens.xgb_short, ens.lgbm_short, ens.xgb_medium, ens.lgbm_medium]:
        if model is None:
            continue
        try:
            imp = model.feature_importances_
            for i, v in enumerate(imp):
                if i < len(feat_names):
                    importances[feat_names[i]] = importances.get(feat_names[i], 0) + v
        except Exception:
            continue

    if not importances:
        return {'error': 'No feature importances available', 'interactions': []}

    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
    top_indices = [feat_names.index(name) for name, _ in top_feats if name in feat_names]

    pairs = []
    for i_idx, i in enumerate(top_indices):
        for j_idx in range(i_idx + 1, len(top_indices)):
            j = top_indices[j_idx]
            if i < X.shape[1] and j < X.shape[1]:
                with np.errstate(invalid='ignore'):
                    corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                if not np.isnan(corr):
                    pairs.append((feat_names[i], feat_names[j], abs(float(corr))))
    pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        'interactions': [{
            'feature_1': _readable_name(p[0]),
            'feature_2': _readable_name(p[1]),
            'interaction_strength': round(p[2], 4),
            'raw_names': [p[0], p[1]],
        } for p in pairs[:top_n]],
        'method': 'correlation',
    }


def feature_group_importance(ticker: str) -> dict:
    """Show feature importance aggregated by feature group/tier."""
    ticker = ticker.upper()
    meta = _load_ticker_meta(ticker)
    if not meta:
        return {'error': f'No trained model for {ticker}'}
    ens = _get_ticker_ensemble(ticker)
    if not ens.loaded:
        return {'error': f'No trained model for {ticker}'}

    feat_names = meta.get('feature_names', [])
    if not feat_names:
        return {'error': 'No feature names in model metadata'}

    # Aggregate importances across models
    importances = np.zeros(len(feat_names))
    n_models = 0
    for model in [ens.xgb_short, ens.lgbm_short, ens.xgb_medium, ens.lgbm_medium,
                  ens.xgb_long, ens.lgbm_long]:
        if model is None:
            continue
        try:
            imp = model.feature_importances_
            for i in range(min(len(imp), len(importances))):
                importances[i] += imp[i]
            n_models += 1
        except Exception:
            continue

    if n_models == 0:
        return {'error': 'No model importances available'}
    importances /= n_models

    # Group by feature category
    groups = {}
    for i, name in enumerate(feat_names):
        group = _store.feature_group(name)
        if group not in groups:
            groups[group] = {'total_importance': 0.0, 'count': 0, 'top_features': []}
        groups[group]['total_importance'] += importances[i]
        groups[group]['count'] += 1
        groups[group]['top_features'].append((name, float(importances[i])))

    # Sort top features within each group
    for g in groups.values():
        g['top_features'] = sorted(g['top_features'], key=lambda x: x[1], reverse=True)[:5]
        g['avg_importance'] = g['total_importance'] / max(g['count'], 1)

    # Sort groups by total importance
    sorted_groups = sorted(groups.items(), key=lambda x: x[1]['total_importance'], reverse=True)

    total_imp = sum(g['total_importance'] for _, g in sorted_groups) or 1.0
    result_groups = []
    for name, g in sorted_groups:
        from backend.models.explain import _readable_name
        result_groups.append({
            'group': name,
            'total_importance': round(g['total_importance'], 6),
            'pct': round(g['total_importance'] / total_imp * 100, 1),
            'count': g['count'],
            'top_features': [{'name': _readable_name(f), 'importance': round(v, 6)}
                             for f, v in g['top_features']],
        })

    return {'ticker': ticker, 'groups': result_groups, 'total_features': len(feat_names)}


def compare_models(ticker1: str, ticker2: str) -> dict:
    """Compare two ticker models side by side."""
    t1, t2 = ticker1.upper(), ticker2.upper()
    m1 = _load_ticker_meta(t1)
    m2 = _load_ticker_meta(t2)
    if not m1:
        return {'error': f'No trained model for {t1}'}
    if not m2:
        return {'error': f'No trained model for {t2}'}

    def _extract(meta, ticker):
        wf = meta.get('walk_forward', {}).get('average', {})
        health = meta.get('model_health', {})
        return {
            'ticker': ticker,
            'health_grade': health.get('grade', '?'),
            'health_score': health.get('score', 0),
            'wf_short': wf.get('short'),
            'wf_medium': wf.get('medium'),
            'wf_long': wf.get('long'),
            'features': meta.get('selected_feature_count') or meta.get('feature_count'),
            'samples': meta.get('samples'),
            'calibrated': meta.get('has_calibrators', False),
            'trained_at': meta.get('trained_at'),
            'ensemble_weights': meta.get('ensemble_weights', {}),
        }

    return {'model1': _extract(m1, t1), 'model2': _extract(m2, t2)}


def retrain_recommendations() -> list:
    """Get list of tickers that need retraining, sorted by urgency."""
    statuses = model_status()
    recs = []
    for s in statuses:
        reasons = []
        urgency = 0
        age = s.get('age_days')
        if age is not None and age > 14:
            reasons.append(f"model is {age} days old")
            urgency += min(age // 7, 5)
        grade = s.get('health_grade', '?')
        if grade in ('D', 'F'):
            reasons.append(f"health grade {grade}")
            urgency += 3
        elif grade == 'C':
            reasons.append(f"health grade {grade}")
            urgency += 1
        if not s.get('calibrated'):
            reasons.append("not calibrated")
            urgency += 1
        wf_vals = [s.get(f'wf_{h}') for h in ('short', 'medium', 'long') if s.get(f'wf_{h}') is not None]
        if wf_vals and max(wf_vals) < 0.55:
            reasons.append(f"low accuracy ({max(wf_vals)*100:.0f}%)")
            urgency += 2
        if reasons:
            recs.append({
                'ticker': s['ticker'],
                'urgency': urgency,
                'reasons': reasons,
                'health_grade': grade,
                'age_days': age,
            })
    recs.sort(key=lambda x: x['urgency'], reverse=True)
    return recs
