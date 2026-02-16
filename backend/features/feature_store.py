"""Feature Store - Combines all features including lagged, sector, and price-derived."""
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from .tier1_price_volume import Tier1PriceVolume
from .tier2_technical import Tier2Technical
from .tier3_options import Tier3Options
from .tier4_fundamentals import Tier4Fundamentals
from .tier5_sentiment import Tier5Sentiment
from .tier6_institutional import Tier6Institutional
from .tier7_macro import Tier7Macro
from .tier8_microstructure import Tier8Microstructure
from .lagged import (compute_lagged_features, compute_price_derived_features,
                     compute_sequence_features,
                     lagged_feature_names, PRICE_DERIVED_NAMES)
from .sector import compute_sector_features, SECTOR_FEATURE_NAMES


class FeatureStore:
    """Centralized feature computation and caching."""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def compute_all_features(
        self,
        price_df: pd.DataFrame,
        options_data: Union[pd.DataFrame, List[Dict], None] = None,
        fundamentals: Union[pd.DataFrame, List[Dict], None] = None,
        news: List[Dict] = None,
        macro_data: Dict = None,
        institutional_data: Dict = None,
        spy_df: pd.DataFrame = None,
        sector_df: pd.DataFrame = None,
        include_sequence: bool = False,
    ) -> pd.DataFrame:
        """Compute all features including lagged and sector-relative."""

        # Tier 1: Price & Volume
        tier1 = Tier1PriceVolume.compute(price_df)
        # Tier 2: Technical
        tier2 = Tier2Technical.compute(price_df)
        # Tier 8: Microstructure
        tier8 = Tier8Microstructure.compute(price_df)

        # Combine time-series features
        features = pd.concat([tier1, tier2, tier8], axis=1)

        # Price-derived sequence features
        price_derived = compute_price_derived_features(price_df)
        features = pd.concat([features, price_derived], axis=1)

        # Sector-relative features
        sector_feats = compute_sector_features(price_df, spy_df, sector_df)
        features = pd.concat([features, sector_feats], axis=1)

        # Lagged features (must come after base features are computed)
        lagged = compute_lagged_features(features)
        features = pd.concat([features, lagged], axis=1)

        # Sequence features (flattened 20-day window of top features)
        if include_sequence:
            seq = compute_sequence_features(features)
            features = pd.concat([features, seq], axis=1)

        # Add point-in-time features
        spot_price = price_df['close'].iloc[-1] if len(price_df) > 0 else 100
        n_rows = len(features)

        options_list = []
        if options_data is not None:
            if isinstance(options_data, pd.DataFrame) and not options_data.empty:
                options_list = options_data.to_dict('records')
            elif isinstance(options_data, list):
                options_list = options_data

        pit_features = {}

        # Tier 3: Options
        tier3 = Tier3Options.compute(options_list, spot_price)
        pit_features.update(tier3)

        fund_list = []
        if fundamentals is not None:
            if isinstance(fundamentals, dict):
                if 'quarterly' in fundamentals and hasattr(fundamentals['quarterly'], 'to_dict'):
                    fund_list = fundamentals['quarterly'].to_dict('records')
                elif 'latest_quarter' in fundamentals:
                    fund_list = [fundamentals['latest_quarter']]
            elif isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
                fund_list = fundamentals.to_dict('records')
            elif isinstance(fundamentals, list):
                fund_list = fundamentals

        # Tier 4: Fundamentals
        tier4 = Tier4Fundamentals.compute(fund_list, spot_price)
        pit_features.update(tier4)

        # Tier 5: Sentiment
        news_list, filings_list, calls_list = [], [], []
        if news:
            if isinstance(news, dict):
                if 'news' in news and hasattr(news['news'], 'to_dict'):
                    news_list = news['news'].to_dict('records')
                if 'sec_filings' in news and hasattr(news['sec_filings'], 'to_dict'):
                    filings_list = news['sec_filings'].to_dict('records')
                if 'earnings_calls' in news and hasattr(news['earnings_calls'], 'to_dict'):
                    calls_list = news['earnings_calls'].to_dict('records')
            elif isinstance(news, list):
                news_list = news
        tier5 = Tier5Sentiment.compute(news_list, filings_list, calls_list)
        pit_features.update(tier5)

        # Tier 6: Institutional
        tier6 = Tier6Institutional.compute(institutional_data)
        pit_features.update(tier6)

        # Tier 7: Macro
        tier7 = Tier7Macro.compute(macro_data or {})
        pit_features.update(tier7)

        # Tier 9: Regime
        from .regime import compute_regime_features
        regime = compute_regime_features(price_df, macro_data)
        regime_numeric = {k: v for k, v in regime.items() if isinstance(v, (int, float))}
        pit_features.update(regime_numeric)

        # Add all point-in-time features at once
        pit_df = pd.DataFrame({k: [v] * n_rows for k, v in pit_features.items()}, index=features.index)
        features = pd.concat([features, pit_df], axis=1)

        return features.fillna(0).replace([np.inf, -np.inf], 0)

    def get_feature_vector(self, features: pd.DataFrame) -> np.ndarray:
        """Get latest feature vector for prediction."""
        return features.iloc[-1].values

    @staticmethod
    def feature_count() -> Dict[str, int]:
        """Return feature count by tier."""
        names = FeatureStore.all_feature_names()
        return {
            'tier1_price_volume': 23,
            'tier2_technical': 85,
            'tier3_options': 25,
            'tier4_fundamentals': 20,
            'tier5_sentiment': 25,
            'tier6_institutional': 5,
            'tier7_macro': 15,
            'tier8_microstructure': 6,
            'tier9_regime': 15,
            'lagged': sum(1 for n in names if '_lag' in n or '_roc5' in n or '_mean5' in n or '_std5' in n),
            'price_derived': len(PRICE_DERIVED_NAMES),
            'sector_relative': len(SECTOR_FEATURE_NAMES),
            'total': len(names),
        }

    @staticmethod
    def all_feature_names() -> List[str]:
        """Return all feature names."""
        regime_names = [
            'regime_bull', 'regime_bear', 'regime_sideways',
            'vol_regime_low', 'vol_regime_normal', 'vol_regime_high',
            'trend_no_trend', 'trend_trending', 'trend_strong_trend', 'adx_value',
            'mom_strong_up', 'mom_up', 'mom_down', 'mom_strong_down', 'momentum_roc20',
        ]
        base_ts = (
            Tier1PriceVolume.feature_names() +
            Tier2Technical.feature_names() +
            Tier8Microstructure.feature_names()
        )
        base_names = (
            base_ts +
            PRICE_DERIVED_NAMES +
            SECTOR_FEATURE_NAMES +
            lagged_feature_names(base_ts) +
            Tier3Options.feature_names() +
            Tier4Fundamentals.feature_names() +
            Tier5Sentiment.feature_names() +
            Tier6Institutional.feature_names() +
            Tier7Macro.feature_names() +
            regime_names
        )
        return base_names


    @staticmethod
    def feature_group(name: str) -> str:
        """Classify a feature name into its group/tier."""
        if '_lag' in name or '_roc5' in name or '_mean5' in name or '_std5' in name:
            return 'lagged'
        if name.startswith(('return_vs_', 'spy_', 'sector_', 'correlation_with_')):
            return 'sector_relative'
        if name in PRICE_DERIVED_NAMES:
            return 'price_derived'
        if name in SECTOR_FEATURE_NAMES:
            return 'sector_relative'
        t1 = set(Tier1PriceVolume.feature_names())
        if name in t1:
            return 'price_volume'
        t2 = set(Tier2Technical.feature_names())
        if name in t2:
            return 'technical'
        t3 = set(Tier3Options.feature_names())
        if name in t3:
            return 'options'
        t4 = set(Tier4Fundamentals.feature_names())
        if name in t4:
            return 'fundamentals'
        t5 = set(Tier5Sentiment.feature_names())
        if name in t5:
            return 'sentiment'
        t6 = set(Tier6Institutional.feature_names())
        if name in t6:
            return 'institutional'
        t7 = set(Tier7Macro.feature_names())
        if name in t7:
            return 'macro'
        t8 = set(Tier8Microstructure.feature_names())
        if name in t8:
            return 'microstructure'
        if name.startswith(('regime_', 'vol_regime_', 'trend_', 'mom_', 'momentum_roc20', 'adx_value')):
            return 'regime'
        if '_t' in name and name.split('_t')[-1].isdigit():
            return 'sequence'
        return 'other'
