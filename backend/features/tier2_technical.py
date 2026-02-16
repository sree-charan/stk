"""Tier 2: Technical Indicators (80 features)."""
import numpy as np
import pandas as pd

class Tier2Technical:
    """Generate 80 technical indicator features."""
    
    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        # Moving averages (8)
        for p in [5, 10, 20, 50]:
            f[f'sma_{p}'] = close.rolling(p).mean()
            f[f'ema_{p}'] = close.ewm(span=p).mean()
        
        # Momentum indicators (8)
        f['rsi_14'] = Tier2Technical._rsi(close, 14)
        f['rsi_7'] = Tier2Technical._rsi(close, 7)
        f['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        f['macd_signal'] = f['macd'].ewm(span=9).mean()
        f['macd_hist'] = f['macd'] - f['macd_signal']
        f['roc_5'] = close.pct_change(5) * 100
        f['roc_10'] = close.pct_change(10) * 100
        f['roc_20'] = close.pct_change(20) * 100
        f['roc_60'] = close.pct_change(60) * 100
        f['momentum_10'] = close - close.shift(10)
        f['williams_r'] = (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min() + 1e-8) * -100
        
        # Volatility indicators (6)
        f['atr_14'] = Tier2Technical._atr(high, low, close, 14)
        f['bb_upper'] = close.rolling(20).mean() + 2 * close.rolling(20).std()
        f['bb_lower'] = close.rolling(20).mean() - 2 * close.rolling(20).std()
        f['bb_width'] = (f['bb_upper'] - f['bb_lower']) / close.rolling(20).mean()
        f['bb_position'] = (close - f['bb_lower']) / (f['bb_upper'] - f['bb_lower'] + 1e-8)
        f['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Trend strength (4)
        f['adx_14'] = Tier2Technical._adx(high, low, close, 14)
        f['trend_strength'] = abs(close - close.rolling(20).mean()) / close.rolling(20).std()
        f['price_channel_pos'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-8)
        f['linear_reg_slope'] = close.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        
        # Volume-based (4)
        f['obv'] = (np.sign(close.diff()) * volume).cumsum()
        f['obv_sma'] = f['obv'].rolling(20).mean()
        f['mfi_14'] = Tier2Technical._mfi(high, low, close, volume, 14)
        f['vpt'] = (close.pct_change() * volume).cumsum()
        f['obv_slope'] = f['obv'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True)
        
        # Stochastic oscillator (2)
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        f['stoch_k'] = 100 * (close - low14) / (high14 - low14 + 1e-8)
        f['stoch_d'] = f['stoch_k'].rolling(3).mean()
        
        # Fibonacci retracement levels (4) - distance from key fib levels
        swing_high = high.rolling(50).max()
        swing_low = low.rolling(50).min()
        fib_range = swing_high - swing_low + 1e-8
        f['fib_236'] = (close - (swing_high - 0.236 * fib_range)) / fib_range
        f['fib_382'] = (close - (swing_high - 0.382 * fib_range)) / fib_range
        f['fib_500'] = (close - (swing_high - 0.500 * fib_range)) / fib_range
        f['fib_618'] = (close - (swing_high - 0.618 * fib_range)) / fib_range
        
        # Ichimoku cloud (5)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        f['ichimoku_tenkan'] = (close - tenkan) / (close + 1e-8)
        f['ichimoku_kijun'] = (close - kijun) / (close + 1e-8)
        f['ichimoku_tk_cross'] = (tenkan - kijun) / (close + 1e-8)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        f['ichimoku_cloud_width'] = (senkou_a - senkou_b) / (close + 1e-8)
        f['ichimoku_cloud_pos'] = (close - senkou_a) / (close + 1e-8)
        
        # Pivot points (3)
        prev_h, prev_l, prev_c = high.shift(1), low.shift(1), close.shift(1)
        pivot = (prev_h + prev_l + prev_c) / 3
        f['pivot_distance'] = (close - pivot) / (close + 1e-8)
        f['pivot_r1_dist'] = (close - (2 * pivot - prev_l)) / (close + 1e-8)
        f['pivot_s1_dist'] = (close - (2 * pivot - prev_h)) / (close + 1e-8)
        
        # Keltner channels (3)
        kelt_mid = close.ewm(span=20).mean()
        kelt_atr = Tier2Technical._atr(high, low, close, 10)
        f['keltner_upper'] = (kelt_mid + 2 * kelt_atr - close) / (close + 1e-8)
        f['keltner_lower'] = (close - (kelt_mid - 2 * kelt_atr)) / (close + 1e-8)
        f['keltner_position'] = (close - (kelt_mid - 2 * kelt_atr)) / (4 * kelt_atr + 1e-8)
        
        # Donchian channels (3)
        don_high = high.rolling(20).max()
        don_low = low.rolling(20).min()
        don_width = don_high - don_low
        f['donchian_position'] = (close - don_low) / (don_width + 1e-8)
        f['donchian_width'] = don_width / (close + 1e-8)
        f['donchian_breakout'] = (close - don_high.shift(1)) / (close + 1e-8)
        
        # ATRP - ATR as percentage of close (1)
        f['atrp'] = Tier2Technical._atr(high, low, close, 14) / (close + 1e-8) * 100
        
        # Chaikin Money Flow (1)
        mfv = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
        f['cmf'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
        # Parabolic SAR (1) - simplified: distance from SAR to close as ratio
        f['psar_dist'] = Tier2Technical._psar_distance(high, low, close)
        
        # ADXR - Average Directional Movement Rating (1)
        adx = f['adx_14']
        f['adxr'] = (adx + adx.shift(14)) / 2
        
        # Accumulation/Distribution Line (1) - normalized slope
        ad = (((close - low) - (high - close)) / (high - low + 1e-8) * volume).cumsum()
        f['adl_slope'] = ad.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True) / (volume.rolling(10).mean() + 1e-8)
        
        # Force Index (1) - EMA of price change * volume
        fi = close.diff() * volume
        f['force_index'] = fi.ewm(span=13).mean() / (volume.rolling(20).mean() * close.rolling(20).mean() + 1e-8)
        
        # CCI - Commodity Channel Index (1)
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        f['cci'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-8)
        
        # Trix (1) - triple-smoothed EMA rate of change
        ema1 = close.ewm(span=15).mean()
        ema2 = ema1.ewm(span=15).mean()
        ema3 = ema2.ewm(span=15).mean()
        f['trix'] = ema3.pct_change() * 10000
        
        # Ultimate Oscillator (1)
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-8)
        avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-8)
        avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-8)
        f['ultimate_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        
        # Vortex Indicator (1) - VI+ minus VI- as trend signal
        vm_plus = abs(high - low.shift(1))
        vm_minus = abs(low - high.shift(1))
        tr_vi = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        vi_plus = vm_plus.rolling(14).sum() / (tr_vi.rolling(14).sum() + 1e-8)
        vi_minus = vm_minus.rolling(14).sum() / (tr_vi.rolling(14).sum() + 1e-8)
        f['vortex'] = vi_plus - vi_minus
        
        # Williams %R 7-period (1)
        f['williams_r_7'] = (high.rolling(7).max() - close) / (high.rolling(7).max() - low.rolling(7).min() + 1e-8) * -100
        
        # Detrended Price Oscillator (1) - close minus SMA shifted back
        dpo_period = 20
        f['dpo'] = close - close.rolling(dpo_period).mean().shift(dpo_period // 2 + 1)
        
        # Mass Index (1) - volatility expansion signal
        ema_hl = (high - low).ewm(span=9).mean()
        ema_ema_hl = ema_hl.ewm(span=9).mean()
        mass_ratio = ema_hl / (ema_ema_hl + 1e-8)
        f['mass_index'] = mass_ratio.rolling(25).sum()
        
        # Ease of Movement (1) - price movement relative to volume
        dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 1e6) / (high - low + 1e-8)
        f['emv'] = (dm / (box_ratio + 1e-8)).rolling(14).mean()
        
        # Chande Momentum Oscillator (1) - momentum bounded -100 to 100
        cmo_period = 14
        cmo_diff = close.diff()
        cmo_up = cmo_diff.where(cmo_diff > 0, 0).rolling(cmo_period).sum()
        cmo_down = (-cmo_diff.where(cmo_diff < 0, 0)).rolling(cmo_period).sum()
        f['cmo'] = 100 * (cmo_up - cmo_down) / (cmo_up + cmo_down + 1e-8)
        
        # Aroon Indicator (2) - trend direction based on time since high/low
        aroon_period = 25
        f['aroon_up'] = high.rolling(aroon_period + 1).apply(lambda x: x.argmax() / aroon_period * 100, raw=True)
        f['aroon_down'] = low.rolling(aroon_period + 1).apply(lambda x: x.argmin() / aroon_period * 100, raw=True)
        
        # Know Sure Thing (1) - weighted sum of smoothed ROC at 4 timeframes
        roc10 = close.pct_change(10) * 100
        roc15 = close.pct_change(15) * 100
        roc20 = close.pct_change(20) * 100
        roc30 = close.pct_change(30) * 100
        f['kst'] = (roc10.rolling(10).mean() + 2 * roc15.rolling(10).mean() +
                    3 * roc20.rolling(10).mean() + 4 * roc30.rolling(15).mean())
        
        # Connors RSI (1) - composite: RSI(3) + streak RSI(2) + percentile rank(100)
        rsi3 = Tier2Technical._rsi(close, 3)
        streak = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        streak_cum = streak.groupby((streak != streak.shift()).cumsum()).cumsum()
        streak_rsi = Tier2Technical._rsi(streak_cum, 2)
        pct_rank = close.pct_change().rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 100 else 50, raw=False)
        f['connors_rsi'] = (rsi3 + streak_rsi + pct_rank) / 3
        
        # Choppiness Index (1) - range-bound (high ~60-100) vs trending (low ~0-40)
        atr1 = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr_sum = atr1.rolling(14).sum()
        hh = high.rolling(14).max()
        ll = low.rolling(14).min()
        f['choppiness'] = 100 * np.log10(atr_sum / (hh - ll + 1e-8)) / np.log10(14)
        
        # ATR Bands (2) - upper/lower bands based on ATR
        atr14 = Tier2Technical._atr(high, low, close, 14)
        ema20 = close.ewm(span=20).mean()
        f['atr_band_upper'] = (ema20 + 2 * atr14 - close) / (close + 1e-8)
        f['atr_band_lower'] = (close - (ema20 - 2 * atr14)) / (close + 1e-8)
        
        # Coppock Curve (1) - long-term momentum buy signal (WMA of sum of ROC(14) + ROC(11))
        roc14_cc = close.pct_change(14) * 100
        roc11_cc = close.pct_change(11) * 100
        cc_sum = roc14_cc + roc11_cc
        weights = np.arange(1, 11)
        f['coppock_curve'] = cc_sum.rolling(10).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == 10 else 0, raw=True)
        
        # Elder Ray Index (2) - Bull Power and Bear Power relative to EMA(13)
        ema13 = close.ewm(span=13).mean()
        f['elder_bull'] = (high - ema13) / (close + 1e-8)
        f['elder_bear'] = (low - ema13) / (close + 1e-8)
        
        # Relative Vigor Index (1) - momentum based on close-open vs high-low
        co = close - df['open']
        hl = high - low + 1e-8
        f['rvi'] = co.rolling(10).mean() / hl.rolling(10).mean()
        
        # Supertrend (1) - ATR-based trend direction: positive=bullish, negative=bearish
        st_atr = Tier2Technical._atr(high, low, close, 10)
        st_mid = (high + low) / 2
        st_upper = st_mid + 3 * st_atr
        st_lower = st_mid - 3 * st_atr
        supertrend = pd.Series(0.0, index=close.index)  # noqa: F841
        direction = pd.Series(1, index=close.index)  # 1=up, -1=down
        for i in range(1, len(close)):
            if close.iloc[i] > st_upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < st_lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
        st_val = direction.astype(float) * (close - st_mid) / (close + 1e-8)
        f['supertrend'] = st_val
        
        # Squeeze Momentum (1) - BB inside Keltner = squeeze; value = momentum direction
        bb_up = close.rolling(20).mean() + 2 * close.rolling(20).std()
        bb_lo = close.rolling(20).mean() - 2 * close.rolling(20).std()
        kc_mid = close.ewm(span=20).mean()
        kc_atr = Tier2Technical._atr(high, low, close, 10)
        kc_up = kc_mid + 1.5 * kc_atr
        kc_lo = kc_mid - 1.5 * kc_atr
        squeeze_on = (bb_lo > kc_lo) & (bb_up < kc_up)
        # Momentum = linear regression of (close - midline) over 20 bars
        midline = (high.rolling(20).max() + low.rolling(20).min()) / 2
        delta = close - (midline + close.rolling(20).mean()) / 2
        mom = delta.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True)
        # Encode: positive mom + squeeze = building pressure; sign indicates direction
        f['squeeze_mom'] = mom * (1 + squeeze_on.astype(float))
        
        # Hull Moving Average (1) - smoother trend: distance from HMA as ratio
        hma_period = 20
        half_wma = close.rolling(hma_period // 2).mean()
        full_wma = close.rolling(hma_period).mean()
        hull_raw = (2 * half_wma - full_wma)
        hma = hull_raw.rolling(int(np.sqrt(hma_period))).mean()
        f['hma_dist'] = (close - hma) / (close + 1e-8)
        
        # Schaff Trend Cycle (1) - combines MACD + stochastic for fast trend detection
        stc_macd = close.ewm(span=23).mean() - close.ewm(span=50).mean()
        stc_lo = stc_macd.rolling(10).min()
        stc_hi = stc_macd.rolling(10).max()
        stc_stoch1 = (stc_macd - stc_lo) / (stc_hi - stc_lo + 1e-8) * 100
        stc_pf = stc_stoch1.ewm(span=3).mean()
        pf_lo = stc_pf.rolling(10).min()
        pf_hi = stc_pf.rolling(10).max()
        stc_stoch2 = (stc_pf - pf_lo) / (pf_hi - pf_lo + 1e-8) * 100
        f['stc'] = stc_stoch2.ewm(span=3).mean()
        
        # Klinger Volume Oscillator (1) - volume-based trend confirmation
        hlc = high + low + close
        dm_kvo = hlc - hlc.shift(1)
        trend = pd.Series(np.where(dm_kvo > 0, 1, -1), index=close.index, dtype=float)
        sv = trend * volume
        kvo_fast = sv.ewm(span=34).mean()
        kvo_slow = sv.ewm(span=55).mean()
        kvo = kvo_fast - kvo_slow
        f['kvo'] = kvo / (volume.rolling(20).mean() + 1e-8)
        
        return f.fillna(0)
    
    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-8))
    
    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0).clip(lower=0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0).clip(lower=0)
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-8)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-8)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        return dx.rolling(period).mean()
    
    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        typical = (high + low + close) / 3
        mf = typical * volume
        pos_mf = mf.where(typical > typical.shift(), 0).rolling(period).sum()
        neg_mf = mf.where(typical < typical.shift(), 0).rolling(period).sum()
        return 100 - 100 / (1 + pos_mf / (neg_mf + 1e-8))
    
    @staticmethod
    def _psar_distance(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Simplified Parabolic SAR: returns normalized distance from SAR to close."""
        n = len(close)
        sar = np.zeros(n)
        af, af_step, af_max = 0.02, 0.02, 0.2
        bull = True
        sar[0] = float(low.iloc[0])
        ep = float(high.iloc[0])
        for i in range(1, n):
            h, l, _c = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if bull:
                if l < sar[i]:
                    bull = False
                    sar[i] = ep
                    ep = l
                    af = af_step
                else:
                    if h > ep:
                        ep = h
                        af = min(af + af_step, af_max)
            else:
                if h > sar[i]:
                    bull = True
                    sar[i] = ep
                    ep = h
                    af = af_step
                else:
                    if l < ep:
                        ep = l
                        af = min(af + af_step, af_max)
        sar_s = pd.Series(sar, index=close.index)
        return (close - sar_s) / (close + 1e-8)
    
    @staticmethod
    def feature_names() -> list:
        return [
            'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20', 'sma_50', 'ema_50',
            'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist', 'roc_5', 'roc_10', 'roc_20', 'roc_60', 'momentum_10', 'williams_r',
            'atr_14', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'volatility_20',
            'adx_14', 'trend_strength', 'price_channel_pos', 'linear_reg_slope',
            'obv', 'obv_sma', 'mfi_14', 'vpt', 'obv_slope',
            'stoch_k', 'stoch_d',
            'fib_236', 'fib_382', 'fib_500', 'fib_618',
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_tk_cross', 'ichimoku_cloud_width', 'ichimoku_cloud_pos',
            'pivot_distance', 'pivot_r1_dist', 'pivot_s1_dist',
            'keltner_upper', 'keltner_lower', 'keltner_position',
            'donchian_position', 'donchian_width', 'donchian_breakout',
            'atrp', 'cmf',
            'psar_dist', 'adxr', 'adl_slope', 'force_index',
            'cci', 'trix', 'ultimate_osc', 'vortex',
            'williams_r_7', 'dpo', 'mass_index', 'emv',
            'cmo', 'aroon_up', 'aroon_down', 'kst',
            'connors_rsi', 'choppiness', 'atr_band_upper', 'atr_band_lower',
            'coppock_curve', 'elder_bull', 'elder_bear', 'rvi',
            'supertrend', 'squeeze_mom', 'hma_dist',
            'stc', 'kvo',
        ]
