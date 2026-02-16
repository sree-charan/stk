#!/usr/bin/env python3
"""Stock Chat CLI - stk command."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box
import time

console = Console()


def _verdict(direction: str, confidence: float) -> str:
    c = int(confidence * 100) if confidence <= 1 else int(confidence)
    if direction == 'bullish':
        return f"🟢 BUY ({c}% confidence)"
    elif direction == 'bearish':
        return f"🔴 SELL ({c}% confidence)"
    return f"🟡 HOLD ({c}% confidence)"


def _fmt_vol(v: int) -> str:
    if v >= 1e9: return f"{v/1e9:.1f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return str(v)


def _sparkline(values: list, width: int = 30) -> str:
    """Generate ASCII sparkline from a list of numeric values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    # Resample to width if needed
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in values)


@click.group()
@click.version_option("1.2.3", prog_name="stk")
@click.option('--verbose', '-v', 'global_verbose', is_flag=True, help='Show debug/warning logs')
@click.pass_context
def cli(ctx, global_verbose) -> None:
    """Stock Chat - ML-powered stock analysis CLI."""
    import logging
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = global_verbose
    level = logging.DEBUG if global_verbose else logging.ERROR
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    # Suppress noisy third-party warnings
    for name in ('urllib3', 'yfinance', 'newsapi', 'requests', 'lightgbm', 'optuna'):
        logging.getLogger(name).setLevel(level)


@cli.command()
@click.argument('ticker')
@click.option('--short', 'horizon', flag_value='short', help='Short-term (1 hour)')
@click.option('--medium', 'horizon', flag_value='medium', help='Medium-term (5 days)')
@click.option('--long', 'horizon', flag_value='long', help='Long-term (60 days)')
@click.option('--verbose', '-v', is_flag=True, help='Show all signals and raw features')
@click.option('--signals', is_flag=True, help='Show only signals (all of them)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--mock', is_flag=True, help='Use mock data (offline mode)')
def analyze(ticker, horizon, verbose, signals, as_json, mock) -> None:
    """Full analysis for a stock ticker."""
    from cli.engine import get_analysis, get_features
    from cli.errors import InvalidTickerError, NetworkError, RateLimitError
    _orig = {}
    if mock:
        import backend.data.real_providers as rp
        from backend.data import mock_generators as mg
        _mock_map = {'generate_ohlcv': mg.get_ohlcv, 'get_options_chain': mg.get_options_chain,
                     'get_fundamentals': mg.get_fundamentals, 'get_sentiment': mg.get_sentiment, 'get_macro_data': mg.get_macro_data}
        _orig = {k: getattr(rp, k) for k in _mock_map}
        for k, v in _mock_map.items():
            setattr(rp, k, v)
    try:
        with console.status(f"[bold green]Analyzing {ticker.upper()}{'  (mock)' if mock else ''}..."):
            try:
                a = get_analysis(ticker)
                raw_feats = get_features(ticker) if verbose else None
            except InvalidTickerError:
                console.print(f"[red]Error: '{ticker.upper()}' is not a valid ticker symbol.[/red]")
                return
            except RateLimitError as e:
                console.print(f"[red]Error: {e}[/red]")
                return
            except NetworkError as e:
                console.print(f"[red]Error: {e}[/red]")
                return
            except Exception as e:
                console.print(f"[red]Error analyzing {ticker}: {e}[/red]")
                return

        if as_json:
            import json
            click.echo(json.dumps(a, indent=2, default=str))
            return

        # --signals mode: only show all signals, no price/horizon info
        if signals:
            console.print(f"[green]Bullish Signals ({len(a.get('all_bullish', a['bullish']))}):[/green]")
            for s in a.get('all_bullish', a['bullish']):
                console.print(f"  🟢 {s}")
            console.print(f"\n[red]Bearish Signals ({len(a.get('all_bearish', a['bearish']))}):[/red]")
            for s in a.get('all_bearish', a['bearish']):
                console.print(f"  🔴 {s}")
            return

        vol_ratio = a['volume'] / max(1, a['avg_volume'])
        header = f"Price: ${a['price']:.2f} ({a['change_pct']:+.1f}%)  Volume: {_fmt_vol(a['volume'])} ({vol_ratio:.1f}x avg)"
        if a.get('regime_display'):
            header += f"\n{a['regime_display']}"
        if a.get('predictability'):
            level, icon, desc = a['predictability']
            header += f"\nPredictability: {icon} {level} — {desc}"
        if a.get('best_horizon'):
            bh = a['best_horizon']
            header += f"\nBest horizon: {bh['label']} ({bh['accuracy']*100:.0f}% WF accuracy) — {bh['reason']}"
        console.print(Panel(header, title=f"{a['ticker']} — {a['name']}", border_style="cyan"))

        horizons_to_show = [horizon] if horizon else ['short', 'medium', 'long']
        labels = {'short': 'SHORT-TERM (1 hour)', 'medium': 'MEDIUM-TERM (5 days)', 'long': 'LONG-TERM (60 days)'}

        for h in horizons_to_show:
            d = a['horizons'][h]
            pred_suffix = ''
            if d.get('predictability_icon'):
                pred_suffix = f"  {d['predictability_icon']} {d['predictability']} predictability"
            console.print(f"\n[bold]{labels[h]}{pred_suffix}[/bold]")
            # Show conviction verdict instead of raw confidence
            if d.get('conviction_verdict'):
                cal_tag = ' [calibrated]' if d.get('calibrated') else ''
                regime_tag = ' [vol-adjusted]' if d.get('regime_adjusted') else ''
                console.print(f"  Verdict: {d['conviction_verdict']}{cal_tag}{regime_tag}")
            else:
                console.print(f"  Verdict: {_verdict(d['direction'], d['confidence'])}")
            # Show volatility z-score if available
            if d.get('vol_zscore') is not None:
                z = d['vol_zscore']
                desc = d.get('vol_zscore_desc', '')
                interval_text = f"  90% range: {d['conf_interval_text']}" if d.get('conf_interval_text') else ''
                console.print(f"  Expected: {d['prediction']*100:+.2f}% ({z:+.1f}σ — {desc})")
                if interval_text:
                    unc = f"  ({d['uncertainty']})" if d.get('uncertainty') else ''
                    console.print(f"  {interval_text}{unc}")
            # Show signals relevant to this horizon
            sigs = a['bullish'][:2] if d['direction'] == 'bullish' else a['bearish'][:2]
            if sigs:
                console.print(f"  Signals: {', '.join(sigs)}")
            console.print(f"  Entry: ${d.get('entry_lo', 0):.2f}-${d.get('entry_hi', 0):.2f}")
            console.print(f"  Stop: ${d['stop']:.2f}")
            console.print(f"  Target: ${d['target']:.2f}")
            if h in ('medium', 'long'):
                console.print(f"  Support: ${d.get('support', 0):.2f}")
                console.print(f"  Resistance: ${d.get('resistance', 0):.2f}")
            # SHAP explanation (always show, not just verbose)
            if d.get('shap_text'):
                console.print(f"\n  [dim]{d['shap_text']}[/dim]")
            # Multi-timeframe voting
            if d.get('mtf_verdict'):
                console.print(f"  Timeframe vote: {d['mtf_verdict']}")
            # Signal quality
            sq = d.get('signal_quality')
            if sq:
                from backend.models.explain import format_signal_quality
                console.print(f"  {format_signal_quality(sq)}")

        # Show signals: verbose = all, default = top 5
        if verbose:
            bull = a.get('all_bullish', a['bullish'])
            bear = a.get('all_bearish', a['bearish'])
        else:
            bull = a['bullish']
            bear = a['bearish']
        console.print(f"\n[green]Top Bullish:[/green] {', '.join(bull)}")
        console.print(f"[red]Top Bearish:[/red] {', '.join(bear)}")

        if a.get('fetched_at'):
            age_str = ''
            if a.get('model_age_days') is not None:
                age = a['model_age_days']
                if age > 30:
                    age_str = f"  ⚠️ Model is {age}d old — consider retraining"
                elif age > 7:
                    age_str = f"  Model trained {age}d ago"
            # Model health
            health = a.get('model_health')
            health_str = ''
            if health:
                icons = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
                health_str = f"  {icons.get(health['grade'], '⚪')} Health: {health['grade']} ({health['score']:.0f}/100)"
            trend_str = ''
            if a.get('health_trend'):
                trend_str = f"  {a['health_trend']}"
            console.print(f"\n[dim]Data as of {a['fetched_at']}{age_str}{health_str}{trend_str}[/dim]")

        # Feature drift warnings
        drift = a.get('drift_warnings', [])
        if drift:
            console.print(f"\n[yellow]⚠ Feature drift detected ({len(drift)} features outside training range):[/yellow]")
            for d in drift[:3]:
                console.print(f"  {d['feature']}: {d['z_score']:+.1f}σ from training mean")

        # Health degradation alert
        if a.get('health_degradation'):
            console.print(f"\n[yellow]{a['health_degradation']}[/yellow]")

        # Past prediction accuracy
        pred_acc = a.get('prediction_accuracy')
        if pred_acc:
            parts = []
            for h in ('short', 'medium', 'long'):
                if h in pred_acc:
                    pa = pred_acc[h]
                    parts.append(f"{h}: {pa['accuracy']*100:.0f}% ({pa['correct']}/{pa['total']})")
            if parts:
                console.print(f"\n[dim]Past prediction accuracy: {', '.join(parts)}[/dim]")

        if verbose and raw_feats is not None:
            # Calibration quality (Brier scores)
            brier = a.get('brier_scores')
            if brier:
                console.print("\n[dim]Calibration quality (Brier score, lower=better):[/dim]")
                for h in ('short', 'medium', 'long'):
                    if h in brier:
                        raw_b = brier[h].get('raw', 0)
                        cal_b = brier[h].get('calibrated')
                        if cal_b is not None:
                            console.print(f"  [dim]{h.capitalize()}: {raw_b:.4f} → {cal_b:.4f}[/dim]")
                        else:
                            console.print(f"  [dim]{h.capitalize()}: {raw_b:.4f}[/dim]")

            # Feature importance table
            top_feats = a.get('top_features', [])
            if top_feats:
                console.print()
                fi = Table(title="Top Feature Importances", box=box.ROUNDED)
                fi.add_column("Feature", style="dim")
                fi.add_column("Importance", justify="right")
                fi.add_column("Value", justify="right")
                for f in top_feats:
                    val_str = f"{f['value']:.4f}" if f['value'] is not None else "N/A"
                    fi.add_row(f['name'], f"{f['importance']*100:.1f}%", val_str)
                console.print(fi)

            # Calibration curve summary (from OOS data)
            cal_curves = a.get('calibration_curves') or {}
            if cal_curves:
                for ch, curve in cal_curves.items():
                    console.print(f"\n[dim]Calibration curve — {ch} horizon (predicted vs actual):[/dim]")
                    for bucket in curve:
                        bar_len = int(bucket['actual'] * 20)
                        bar = '█' * bar_len + '░' * (20 - bar_len)
                        console.print(f"  [dim]{bucket['predicted']:.0%} → {bucket['actual']:.0%} ({bucket['count']}n) {bar}[/dim]")

            console.print()
            ft = Table(title="Feature Values", box=box.ROUNDED)
            ft.add_column("Feature", style="dim")
            ft.add_column("Value", justify="right")
            for name, val in sorted(raw_feats.items()):
                ft.add_row(name, f"{val:.4f}" if isinstance(val, float) else str(val))
            console.print(ft)
    finally:
        if _orig:
            import backend.data.real_providers as rp
            for k, v in _orig.items():
                setattr(rp, k, v)


@cli.command()
@click.argument('ticker', required=False)
@click.option('--all', 'retrain_all_flag', is_flag=True, help='Retrain all saved ticker models')
@click.option('--tune', is_flag=True, help='Enable Optuna hyperparameter tuning (slower but better)')
def retrain(ticker, retrain_all_flag, tune) -> None:
    """Retrain a ticker's ML model, or all saved models with --all."""
    from cli.engine import retrain_ticker, retrain_all as _retrain_all
    if retrain_all_flag:
        with console.status("[bold green]Retraining all models..."):
            results = _retrain_all()
        if not results:
            console.print("[yellow]No saved models found to retrain.[/yellow]")
            return
        for t, r in results.items():
            if 'error' in r:
                console.print(f"[red]{t}: {r['error']}[/red]")
            else:
                console.print(f"[green]{t}: {r['short']*100:.1f}% short, {r['medium']*100:.1f}% medium, {r['long']*100:.1f}% long[/green]")
        return
    if not ticker:
        console.print("[red]Provide a TICKER or use --all[/red]")
        return
    label = f"Retraining {ticker.upper()}{'  (with tuning)' if tune else ''}..."
    console.print(f"[bold green]{label}[/bold green]")
    try:
        acc = retrain_ticker(ticker, tune=tune)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return
    # Model quality badge based on average accuracy
    avg_acc = (acc['short'] + acc['medium'] + acc['long']) / 3
    if avg_acc > 0.60:
        badge = "🟢 Good"
    elif avg_acc > 0.55:
        badge = "🟡 Fair"
    else:
        badge = "🔴 Weak"
    console.print(f"[green]Retrain complete: {acc['short']*100:.1f}% short, {acc['medium']*100:.1f}% medium, {acc['long']*100:.1f}% long  ({badge})[/green]")


@cli.command()
def models() -> None:
    """Show health and accuracy summary for all trained models."""
    from cli.engine import model_status
    statuses = model_status()
    if not statuses:
        console.print("[yellow]No trained models found.[/yellow]")
        return
    t = Table(title="Trained Models", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Health", justify="center")
    t.add_column("WF Short", justify="right")
    t.add_column("WF Medium", justify="right")
    t.add_column("WF Long", justify="right")
    t.add_column("Features", justify="right")
    t.add_column("Cal", justify="center")
    t.add_column("Best", justify="center")
    t.add_column("Age", justify="right")
    icons = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
    for s in statuses:
        icon = icons.get(s['health_grade'], '⚪')
        health = f"{icon} {s['health_grade']} ({s['health_score']:.0f})"
        wf_s = f"{s['wf_short']*100:.1f}%" if s['wf_short'] else '-'
        wf_m = f"{s['wf_medium']*100:.1f}%" if s['wf_medium'] else '-'
        wf_l = f"{s['wf_long']*100:.1f}%" if s['wf_long'] else '-'
        total = s.get('total_features')
        sel = s.get('features')
        feats = f"{sel}/{total}" if sel and total else (str(sel or total or '-'))
        cal = '✓' if s.get('calibrated') else '-'
        if s.get('avg_brier') is not None:
            cal = f"✓ {s['avg_brier']:.3f}" if s.get('calibrated') else f"{s['avg_brier']:.3f}"
        age = f"{s['age_days']}d" if s['age_days'] is not None else '-'
        best = s.get('best_horizon', '-') or '-'
        t.add_row(s['ticker'], health, wf_s, wf_m, wf_l, feats, cal, best, age)
    console.print(t)


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def explain(ticker, as_json) -> None:
    """Show detailed model breakdown for a ticker.

    Example: stk explain TSLA
    """
    from cli.engine import get_model_explanation
    info = get_model_explanation(ticker)
    if 'error' in info:
        console.print(f"[red]{info['error']}[/red]")
        return
    if as_json:
        import json
        console.print(json.dumps(info, indent=2, default=str))
        return
    console.print(f"\n[bold]{info['ticker']}[/bold] Model Explanation")
    console.print(f"  Trained: {info.get('trained_at', 'unknown')}")
    console.print(f"  Samples: {info.get('samples', '?')}  Features: {info.get('selected_feature_count', '?')}/{info.get('feature_count', '?')}")
    # Health
    h = info.get('health', {})
    icons = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
    icon = icons.get(h.get('grade', ''), '⚪')
    console.print(f"  Health: {icon} {h.get('grade', '?')} ({h.get('score', 0):.0f}/100)")
    # WF accuracy
    wf = info.get('wf_accuracy', {})
    wf_std = info.get('wf_std', {})
    wf_ranges = info.get('wf_ranges', {})
    wf_trend = info.get('wf_trend', {})
    if wf:
        console.print("\n  Walk-Forward Accuracy:")
        for hz in ('short', 'medium', 'long'):
            if hz in wf:
                lo, hi = wf_ranges.get(hz, (0, 0))
                t = wf_trend.get(hz, '')
                t_icon = '📈' if t == 'improving' else ('📉' if t == 'degrading' else '➡️')
                console.print(f"    {hz.capitalize()}: {wf[hz]*100:.1f}% ± {wf_std.get(hz, 0)*100:.1f}% "
                              f"(range: {lo*100:.0f}-{hi*100:.0f}%) {t_icon}")
    # Brier scores
    brier = info.get('brier_scores', {})
    if brier:
        console.print("\n  Calibration (Brier scores, lower=better):")
        for hz in ('short', 'medium', 'long'):
            b = brier.get(hz, {})
            raw = b.get('raw')
            cal = b.get('calibrated')
            if raw is not None:
                cal_str = f" → {cal:.4f} calibrated" if cal is not None else ""
                console.print(f"    {hz.capitalize()}: {raw:.4f}{cal_str}")
    # Ensemble
    ew = info.get('ensemble_weights', {})
    if ew:
        xw = ew.get('xgb', 0)
        lw = ew.get('lgbm', 0)
        xbar = '█' * int(xw * 20) + '░' * (20 - int(xw * 20))
        lbar = '█' * int(lw * 20) + '░' * (20 - int(lw * 20))
        console.print("\n  Ensemble weights:")
        console.print(f"    XGB:  {xw:.2f} {xbar}")
        console.print(f"    LGBM: {lw:.2f} {lbar}")
    ed = info.get('ensemble_diversity', {})
    if ed:
        for hz, d in ed.items():
            console.print(f"  Diversity ({hz}): {d.get('diversity', 0)*100:.1f}% ({d.get('description', '')})")
    # Top features
    tf = info.get('top_features', [])
    if tf:
        console.print(f"\n  Top features: {', '.join(tf[:10])}")
    # Health trend
    ht = info.get('health_trend', [])
    if ht:
        grades = ' → '.join(f"{e['grade']}({e['score']:.0f})" for e in ht)
        console.print(f"\n  Health trend: {grades}")
    # Feature changelog
    cl = info.get('feature_changelog', [])
    if cl:
        console.print("\n  Feature changes:")
        for entry in cl:
            added = entry.get('added', [])
            removed = entry.get('removed', [])
            if added or removed:
                console.print(f"    {entry.get('timestamp', '?')[:10]}: +{len(added)} -{len(removed)}")
    # Feature stability
    fs_top = info.get('feature_stability_top', [])
    if fs_top:
        stable_n = info.get('stable_feature_count', 0)
        console.print(f"\n  Feature stability: {stable_n} features stable across ≥50% of WF windows")
        from backend.models.explain import _readable_name
        for f, v in fs_top[:5]:
            console.print(f"    {_readable_name(f)}: {v:.0%}")
    # Conformal interval summary
    conf = info.get('conformal_summary', {})
    if conf:
        console.print("\n  Prediction Intervals (90% conformal):")
        for h in ('short', 'medium', 'long'):
            c = conf.get(h)
            if c:
                w = c['interval_width_90']
                label = '🎯 tight' if w < 5 else ('📊 moderate' if w < 15 else '🌫️ wide')
                console.print(f"    {h.capitalize()}: ±{w/2:.1f}% ({label}, n={c['n_residuals']})")
    console.print()


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def calibration(ticker, as_json) -> None:
    """Show probability calibration diagnostics for a ticker's model.

    Example: stk calibration TSLA
    """
    from cli.engine import get_model_explanation
    info = get_model_explanation(ticker)
    if 'error' in info:
        console.print(f"[red]{info['error']}[/red]")
        return
    if as_json:
        import json
        cal_data = {
            'ticker': info['ticker'],
            'brier_scores': info.get('brier_scores', {}),
            'calibrated': info.get('calibrated', False),
        }
        # Load calibration curves from meta
        from cli.engine import _load_ticker_meta
        meta = _load_ticker_meta(ticker)
        if meta:
            cal_data['calibration_curves'] = meta.get('calibration_curves', {})
        console.print(json.dumps(cal_data, indent=2, default=str))
        return

    console.print(f"\n[bold]{ticker.upper()}[/bold] Calibration Diagnostics")
    cal_status = "✓ Calibrated" if info.get('calibrated') else "✗ Uncalibrated"
    console.print(f"  Status: {cal_status}")

    brier = info.get('brier_scores', {})
    if brier:
        console.print("\n  Brier Scores (lower = better, 0.25 = random):")
        for h in ('short', 'medium', 'long'):
            b = brier.get(h, {})
            raw = b.get('raw')
            cal = b.get('calibrated')
            if raw is not None:
                quality = '🟢' if raw < 0.22 else ('🟡' if raw < 0.25 else '🔴')
                cal_str = f" → {cal:.4f} calibrated" if cal is not None else ""
                console.print(f"    {h.capitalize()}: {quality} {raw:.4f}{cal_str}")
    else:
        console.print("\n  [yellow]No Brier scores available. Retrain to generate.[/yellow]")

    # Show calibration curves
    from cli.engine import _load_ticker_meta
    meta = _load_ticker_meta(ticker)
    curves = meta.get('calibration_curves', {}) if meta else {}
    if curves:
        console.print("\n  Calibration Curves (predicted vs actual):")
        for h in ('short', 'medium', 'long'):
            bins = curves.get(h, [])
            if bins:
                console.print(f"    {h.capitalize()}:")
                for b in bins:
                    pred = b['predicted']
                    act = b['actual']
                    n = b['count']
                    gap = abs(pred - act)
                    quality = '✓' if gap < 0.05 else ('~' if gap < 0.10 else '✗')
                    console.print(f"      {quality} Predicted: {pred:.2f}  Actual: {act:.2f}  (n={n}, gap={gap:.2f})")
        # ASCII chart
        from backend.models.explain import format_ascii_calibration_curve
        for h in ('short', 'medium', 'long'):
            bins = curves.get(h, [])
            if bins:
                console.print(f"\n  {h.capitalize()} calibration chart:")
                console.print(format_ascii_calibration_curve(bins))
    console.print()


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def price(ticker, as_json) -> None:
    """Current price and change for a ticker."""
    from cli.engine import get_price
    with console.status(f"Fetching {ticker.upper()}..."):
        try:
            p = get_price(ticker)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if as_json:
        import json
        console.print(json.dumps(p, indent=2, default=str))
        return
    color = "green" if p['change'] >= 0 else "red"
    console.print(f"[bold]{p['ticker']}[/bold]  ${p['price']:.2f}  [{color}]{p['change']:+.2f} ({p['change_pct']:+.1f}%)[/{color}]")
    console.print(f"  Range: ${p['low']:.2f} - ${p['high']:.2f}  Volume: {_fmt_vol(p['volume'])}")


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def news(ticker, as_json) -> None:
    """Recent news with sentiment for a ticker."""
    from cli.engine import get_news
    with console.status(f"Fetching news for {ticker.upper()}..."):
        try:
            articles = get_news(ticker)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if as_json:
        import json
        console.print(json.dumps(articles, indent=2, default=str))
        return
    if not articles:
        console.print(f"No recent news for {ticker.upper()}")
        return
    t = Table(title=f"{ticker.upper()} News", box=box.ROUNDED)
    t.add_column("Sentiment", width=10)
    t.add_column("Headline", ratio=3)
    t.add_column("Source", width=15)
    for a in articles:
        s = a.get('sentiment', 0)
        color = "green" if s > 0.1 else ("red" if s < -0.1 else "yellow")
        t.add_row(f"[{color}]{s:+.2f}[/{color}]", str(a.get('headline', ''))[:80], str(a.get('source', '')))
    console.print(t)


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def earnings(ticker, as_json) -> None:
    """Earnings info for a ticker."""
    from cli.engine import get_earnings
    with console.status(f"Fetching earnings for {ticker.upper()}..."):
        try:
            e = get_earnings(ticker)
        except Exception as ex:
            console.print(f"[red]Error: {ex}[/red]")
            return
    if as_json:
        import json
        console.print(json.dumps(e, indent=2, default=str))
        return
    val = e.get('valuation', {})
    if val:
        console.print(Panel(
            f"P/E: {val.get('pe_ratio', 'N/A')}  Forward P/E: {val.get('forward_pe', 'N/A')}  "
            f"P/S: {val.get('ps_ratio', 'N/A')}  P/B: {val.get('pb_ratio', 'N/A')}",
            title=f"{ticker.upper()} Valuation", border_style="cyan"))
    qs = e.get('quarters', [])
    if qs:
        t = Table(title="Recent Quarters", box=box.ROUNDED)
        t.add_column("Quarter")
        t.add_column("EPS", justify="right")
        t.add_column("Revenue", justify="right")
        t.add_column("Margin", justify="right")
        for q in qs:
            rev = q.get('revenue', 0)
            rev_str = f"${rev/1e9:.2f}B" if rev > 1e9 else f"${rev/1e6:.0f}M" if rev > 1e6 else str(rev)
            t.add_row(str(q.get('quarter', '')), f"${q.get('eps', 0):.2f}", rev_str,
                       f"{q.get('net_margin', 0)*100:.1f}%")
        console.print(t)


@cli.command()
@click.argument('tickers', nargs=-1, required=True)
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--chart', is_flag=True, help='Show ASCII sparkline price charts')
@click.option('--correlation', is_flag=True, help='Show correlation matrix between tickers')
def compare(tickers, as_json, chart, correlation) -> None:
    """Compare analysis for multiple tickers side by side.

    Example: stk compare TSLA AAPL GOOGL
    Example: stk compare TSLA AAPL --chart
    Example: stk compare TSLA AAPL --correlation
    """
    if len(tickers) < 2:
        console.print("[red]Error: Provide at least 2 tickers to compare.[/red]")
        return
    from cli.engine import get_analysis, get_price_history, correlate_tickers
    results = {}
    histories = {}
    with console.status(f"[bold green]Comparing {', '.join(t.upper() for t in tickers)}..."):
        for t in tickers:
            try:
                results[t.upper()] = get_analysis(t)
                if chart:
                    histories[t.upper()] = get_price_history(t, 30)
            except Exception as e:
                console.print(f"[red]Error fetching {t.upper()}: {e}[/red]")

    if not results:
        return

    if as_json:
        import json; click.echo(json.dumps(results, default=str)); return

    t = Table(title="Comparison", box=box.ROUNDED)
    t.add_column("", style="bold")
    for tk in results:
        t.add_column(tk, justify="right")

    t.add_row("Price", *[f"${r['price']:.2f}" for r in results.values()])
    t.add_row("Change", *[
        f"[{'green' if r['change_pct']>=0 else 'red'}]{r['change_pct']:+.1f}%[/]"
        for r in results.values()])
    t.add_row("Volume", *[_fmt_vol(r['volume']) for r in results.values()])
    t.add_row("Sector", *[r.get('sector', 'N/A') for r in results.values()])
    t.add_row("P/E", *[f"{r['pe_ratio']:.1f}" if r.get('pe_ratio') else 'N/A' for r in results.values()])
    t.add_row("Mkt Cap", *[_fmt_vol(r['market_cap']) if r.get('market_cap') else 'N/A' for r in results.values()])

    for h, label in [('short', 'Short'), ('medium', 'Medium'), ('long', 'Long')]:
        t.add_row(f"{label} Verdict", *[
            _verdict(r['horizons'][h]['direction'], r['horizons'][h]['confidence'])
            for r in results.values()])

    t.add_row("Top Bullish", *[r['bullish'][0] if r['bullish'] else '-' for r in results.values()])
    t.add_row("Top Bearish", *[r['bearish'][0] if r['bearish'] else '-' for r in results.values()])
    console.print(t)

    if chart and histories:
        console.print()
        console.print(Panel("[bold]30-Day Price Sparklines[/bold]"))
        for tk, prices in histories.items():
            sparkline = _sparkline(prices)
            pct = (prices[-1] - prices[0]) / prices[0] * 100 if prices else 0
            color = "green" if pct >= 0 else "red"
            console.print(f"  [bold]{tk:>5}[/bold]  {sparkline}  [{color}]{pct:+.1f}%[/]")

    if correlation and len(results) >= 2:
        console.print()
        tks = list(results.keys())
        ct = Table(title="Correlation Matrix (180d)", box=box.ROUNDED)
        ct.add_column("", style="bold")
        for tk in tks:
            ct.add_column(tk, justify="right")
        for i, t1 in enumerate(tks):
            row = []
            for j, t2 in enumerate(tks):
                if i == j:
                    row.append("[bold]1.00[/bold]")
                else:
                    try:
                        c = correlate_tickers(t1, t2, 180)
                        v = c['correlation']
                        color = "green" if abs(v) < 0.5 else "yellow" if abs(v) < 0.8 else "red"
                        row.append(f"[{color}]{v:.2f}[/]")
                    except Exception:
                        row.append("N/A")
            ct.add_row(t1, *row)
        console.print(ct)


# Portfolio commands
@cli.command()
@click.argument('ticker')
@click.option('--entry', required=True, type=float, help='Entry price')
@click.option('--qty', required=True, type=float, help='Number of shares')
def hold(ticker, entry, qty) -> None:
    """Add a position to your portfolio."""
    if entry <= 0:
        console.print("[red]Error: Entry price must be positive.[/red]")
        return
    if qty <= 0:
        console.print("[red]Error: Quantity must be positive.[/red]")
        return
    from cli.db import add_position
    add_position(ticker, entry, qty)
    console.print(f"[green]Added {qty:.0f} shares of {ticker.upper()} @ ${entry:.2f}[/green]")


@cli.command()
@click.argument('ticker')
def sell(ticker) -> None:
    """Remove a position from your portfolio."""
    from cli.db import get_positions, remove_position
    pos = [p for p in get_positions() if p['ticker'] == ticker.upper()]
    if not pos:
        console.print(f"[yellow]{ticker.upper()} is not in your positions.[/yellow]")
        return
    remove_position(ticker)
    console.print(f"[yellow]Removed {ticker.upper()} from positions[/yellow]")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--sort', type=click.Choice(['ticker', 'pnl', 'value', 'change'], case_sensitive=False),
              default=None, help='Sort by field (ticker, pnl, value, change)')
@click.option('--group-by', 'group_by', type=click.Choice(['sector'], case_sensitive=False),
              default=None, help='Group positions by sector')
def positions(as_json, sort, group_by) -> None:
    """Show all positions with live P&L.

    Examples:
        stk positions               # default order
        stk positions --sort pnl    # sort by P&L (highest first)
        stk positions --group-by sector
    """
    from cli.db import get_positions
    from cli.engine import get_price
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    rows = []
    total_pnl = 0
    for p in pos:
        try:
            pr = get_price(p['ticker'])
            cur = pr['price']
        except Exception:
            cur = p['entry_price']
        pnl = (cur - p['entry_price']) * p['qty']
        pnl_pct = (cur - p['entry_price']) / p['entry_price'] * 100
        total_pnl += pnl
        rows.append({**p, 'current': cur, 'pnl': pnl, 'pnl_pct': pnl_pct})
        # Save snapshot for cumulative tracking
        try:
            from cli.db import save_snapshot
            save_snapshot(p['ticker'], cur, pnl_pct)
        except Exception:
            pass
    if sort:
        sort_keys = {'ticker': ('ticker', False), 'pnl': ('pnl', True),
                     'value': (lambda r: r['current'] * r['qty'], True), 'change': ('pnl_pct', True)}
        sk = sort_keys[sort.lower()]
        if callable(sk[0]):
            rows.sort(key=sk[0], reverse=sk[1])
        else:
            rows.sort(key=lambda r: r.get(sk[0]) or 0, reverse=sk[1])
    if as_json:
        import json; click.echo(json.dumps({'positions': rows, 'total_pnl': total_pnl})); return
    t = Table(title="Portfolio Positions", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Qty", justify="right")
    t.add_column("Entry", justify="right")
    t.add_column("Current", justify="right")
    t.add_column("P&L", justify="right")
    t.add_column("P&L %", justify="right")
    t.add_column("Trend", justify="left")
    for r in rows:
        color = "green" if r['pnl'] >= 0 else "red"
        # Get recent price history for sparkline
        try:
            from cli.engine import get_price_history
            hist = get_price_history(r['ticker'], 14)
            spark = _sparkline(hist, 10)
        except Exception:
            spark = ""
        t.add_row(r['ticker'], f"{r['qty']:.0f}", f"${r['entry_price']:.2f}",
                   f"${r['current']:.2f}", f"[{color}]${r['pnl']:+,.2f}[/{color}]",
                   f"[{color}]{r['pnl_pct']:+.1f}%[/{color}]", spark)
    console.print(t)
    color = "green" if total_pnl >= 0 else "red"
    console.print(f"\nTotal P&L: [{color}]${total_pnl:+,.2f}[/{color}]")

    if group_by == 'sector':
        from cli.engine import SECTOR_TICKERS
        # Build reverse map: ticker -> sector
        ticker_sector = {}
        for sector, tks in SECTOR_TICKERS.items():
            for tk in tks:
                ticker_sector[tk] = sector
        groups = {}
        for r in rows:
            sec = ticker_sector.get(r['ticker'], 'Other')
            groups.setdefault(sec, []).append(r)
        console.print()
        st = Table(title="By Sector", box=box.SIMPLE)
        st.add_column("Sector", style="bold")
        st.add_column("Positions", justify="right")
        st.add_column("P&L", justify="right")
        for sec in sorted(groups):
            sec_pnl = sum(r['pnl'] for r in groups[sec])
            c = "green" if sec_pnl >= 0 else "red"
            st.add_row(sec, str(len(groups[sec])), f"[{c}]${sec_pnl:+,.2f}[/{c}]")
        console.print(st)


@cli.command('position-history')
@click.argument('ticker')
@click.option('--limit', default=30, type=int, help='Number of snapshots')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def position_history(ticker, limit, as_json) -> None:
    """Show cumulative return history for a position.

    Example: stk position-history TSLA --limit 10
    """
    from cli.db import get_snapshots
    snaps = get_snapshots(ticker, limit)
    if not snaps:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print(f"[dim]No history for {ticker.upper()}. Run [bold]stk positions[/bold] to record snapshots.[/dim]")
        return
    if as_json:
        import json; click.echo(json.dumps(snaps, default=str)); return
    t = Table(title=f"{ticker.upper()} P&L History", box=box.SIMPLE)
    t.add_column("Date", style="dim")
    t.add_column("Price", justify="right")
    t.add_column("P&L %", justify="right")
    for s in reversed(snaps):
        c = "green" if s['pnl_pct'] >= 0 else "red"
        date_str = s['snapshot_at'][:10] if s.get('snapshot_at') else 'N/A'
        t.add_row(date_str, f"${s['price']:.2f}", f"[{c}]{s['pnl_pct']:+.1f}%[/{c}]")
    console.print(t)


# Watchlist commands
@cli.command()
@click.argument('ticker')
def watch(ticker) -> None:
    """Add a ticker to your watchlist."""
    from cli.db import add_watch
    add_watch(ticker)
    console.print(f"[green]Added {ticker.upper()} to watchlist[/green]")


@cli.command()
@click.argument('ticker')
def unwatch(ticker) -> None:
    """Remove a ticker from your watchlist."""
    from cli.db import get_watchlist, remove_watch
    wl = [w for w in get_watchlist() if w['ticker'] == ticker.upper()]
    if not wl:
        console.print(f"[yellow]{ticker.upper()} is not in your watchlist.[/yellow]")
        return
    remove_watch(ticker)
    console.print(f"[yellow]Removed {ticker.upper()} from watchlist[/yellow]")


@cli.command('watchlist-export')
@click.option('--output', '-o', default=None, help='Output file path (default: stdout)')
def watchlist_export(output) -> None:
    """Export watchlist to JSON.

    Examples:
        stk watchlist-export                    # print to stdout
        stk watchlist-export -o watchlist.json  # save to file
    """
    from cli.db import get_watchlist
    import json
    wl = get_watchlist()
    tickers = [w['ticker'] for w in wl]
    if output:
        Path(output).write_text(json.dumps(tickers, indent=2))
        console.print(f"[green]Exported {len(tickers)} ticker(s) to {output}[/green]")
    else:
        click.echo(json.dumps(tickers, indent=2))


@cli.command('watchlist-import')
@click.argument('file_path', type=click.Path(exists=True))
def watchlist_import(file_path) -> None:
    """Import watchlist from a JSON file.

    Example: stk watchlist-import watchlist.json
    """
    from cli.db import add_watch
    import json
    tickers = json.loads(Path(file_path).read_text())
    count = 0
    for t in tickers:
        if isinstance(t, str) and t.strip():
            add_watch(t.strip())
            count += 1
    console.print(f"[green]Imported {count} ticker(s) to watchlist.[/green]")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--signals', '-s', is_flag=True, help='Show quick buy/sell/hold signals')
@click.option('--sort', type=click.Choice(['ticker', 'price', 'change'], case_sensitive=False),
              default=None, help='Sort by field (ticker, price, change)')
def watchlist(as_json, signals, sort) -> None:
    """Show watchlist with current signals.

    Examples:
        stk watchlist               # basic price view
        stk watchlist --signals     # include buy/sell/hold verdicts
        stk watchlist --sort change # sort by % change
    """
    from cli.db import get_watchlist
    from cli.engine import get_price
    wl = get_watchlist()
    if not wl:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print("Watchlist empty. Use [bold]stk watch TICKER[/bold] to add.")
        return
    rows = []
    for w in wl:
        try:
            p = get_price(w['ticker'])
            row = {'ticker': w['ticker'], 'price': p['price'], 'change_pct': p['change_pct']}
        except Exception:
            row = {'ticker': w['ticker'], 'price': None, 'change_pct': None}
        if signals and row['price'] is not None:
            try:
                from cli.engine import get_analysis
                a = get_analysis(w['ticker'])
                h = a.get('horizons', {})
                short = h.get('short', {})
                row['signal'] = short.get('direction', 'neutral')
                row['confidence'] = short.get('confidence', 0)
            except Exception:
                row['signal'] = 'unknown'
                row['confidence'] = 0
        rows.append(row)
    if sort:
        sort_keys = {'ticker': ('ticker', False), 'price': ('price', True), 'change': ('change_pct', True)}
        key, rev = sort_keys[sort.lower()]
        rows.sort(key=lambda r: r.get(key) or 0, reverse=rev)
    if as_json:
        import json; click.echo(json.dumps(rows)); return
    title = "Watchlist (with signals)" if signals else "Watchlist"
    t = Table(title=title, box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    if signals:
        t.add_column("Signal", justify="center")
    for r in rows:
        if r['price'] is not None:
            color = "green" if r['change_pct'] >= 0 else "red"
            row_vals = [r['ticker'], f"${r['price']:.2f}",
                        f"[{color}]{r['change_pct']:+.1f}%[/{color}]"]
            if signals:
                row_vals.append(_verdict(r.get('signal', 'neutral'), r.get('confidence', 0)))
        else:
            row_vals = [r['ticker'], "N/A", "N/A"]
            if signals:
                row_vals.append("[dim]N/A[/dim]")
        t.add_row(*row_vals)
    console.print(t)


@cli.command()
@click.argument('message')
def chat(message) -> None:
    """Natural language stock query."""
    from cli.engine import chat_query
    with console.status("[bold green]Thinking..."):
        try:
            resp = chat_query(message)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    console.print(Panel(resp, title="Stock Assistant", border_style="green"))


@cli.command()
def dashboard() -> None:
    """Live dashboard with positions, watchlist, and alerts."""
    from cli.db import get_positions, get_watchlist
    from cli.engine import get_price, check_alerts

    def _build() -> None:
        layout = Layout()
        # Positions
        pos = get_positions()
        pos_t = Table(title="Positions", box=box.SIMPLE)
        pos_t.add_column("Ticker")
        pos_t.add_column("P&L", justify="right")
        for p in pos:
            try:
                pr = get_price(p['ticker'])
                pnl = (pr['price'] - p['entry_price']) * p['qty']
                c = "green" if pnl >= 0 else "red"
                pos_t.add_row(p['ticker'], f"[{c}]${pnl:+,.2f}[/{c}]")
            except Exception:
                pos_t.add_row(p['ticker'], "N/A")

        # Watchlist
        wl = get_watchlist()
        wl_t = Table(title="Watchlist", box=box.SIMPLE)
        wl_t.add_column("Ticker")
        wl_t.add_column("Price", justify="right")
        wl_t.add_column("Chg", justify="right")
        for w in wl:
            try:
                pr = get_price(w['ticker'])
                c = "green" if pr['change'] >= 0 else "red"
                wl_t.add_row(w['ticker'], f"${pr['price']:.2f}", f"[{c}]{pr['change_pct']:+.1f}%[/{c}]")
            except Exception:
                wl_t.add_row(w['ticker'], "N/A", "N/A")

        # Check alerts
        try:
            triggered = check_alerts()
        except Exception:
            triggered = []
        alert_text = ""
        if triggered:
            alert_text = "\n".join(
                f"🔔 {a['ticker']} {a['condition']} ${a['threshold']:.2f} (now ${a['current_price']:.2f})"
                for a in triggered)

        left = Panel(pos_t, border_style="cyan")
        right = Panel(wl_t, border_style="green")
        layout.split_column(
            Layout(name="main", ratio=4),
            Layout(name="alerts", ratio=1, visible=bool(alert_text)),
        )
        layout["main"].split_row(Layout(left, name="left"), Layout(right, name="right"))
        if alert_text:
            layout["alerts"].update(Panel(alert_text, title="⚠ Alerts", border_style="red"))
        else:
            layout["alerts"].update(Panel("[dim]No alerts triggered[/dim]", title="Alerts", border_style="dim"))
        return layout

    console.print("[bold]Live Dashboard[/bold] (Ctrl+C to exit)\n")
    try:
        with Live(_build(), console=console, refresh_per_second=0.1) as live:
            while True:
                time.sleep(30)
                live.update(_build())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed.[/yellow]")


# Config commands
@cli.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value) -> None:
    """Set a config value (e.g., fred-key, news-key)."""
    from cli.config import set_key
    set_key(key, value)
    console.print(f"[green]Set {key}[/green]")


@config.command('get')
@click.argument('key')
def config_get(key) -> None:
    """Get a config value."""
    from cli.config import get as cfg_get
    v = cfg_get(key)
    if v is None:
        console.print(f"[yellow]{key} not set[/yellow]")
    else:
        console.print(f"{key} = {v}")


@config.command('list')
def config_list() -> None:
    """Show all config values."""
    _show_config()


@config.command('show')
def config_show() -> None:
    """Show all config values (alias for 'list')."""
    _show_config()


@config.command('reset')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def config_reset(yes) -> None:
    """Reset all config values to defaults.

    Example: stk config reset --yes
    """
    if not yes:
        if not click.confirm("Reset all config values?"):
            return
    from cli.config import reset
    reset()
    console.print("[green]Config reset to defaults.[/green]")


def _show_config() -> None:
    from cli.config import _load
    cfg = _load()
    if not cfg:
        console.print("[yellow]No config values set.[/yellow]")
        return
    for k, v in sorted(cfg.items()):
        display = v if 'key' not in k.lower() else v[:4] + '...' + v[-4:] if len(str(v)) > 8 else '****'
        console.print(f"  {k} = {display}")


@cli.command('cache-clean')
@click.option('--max-age', default=24, type=int, help='Max age in hours (default: 24)')
def cache_clean(max_age) -> None:
    """Remove old cache entries."""
    from backend.utils.retry import cleanup_cache
    removed = cleanup_cache(max_age_hours=max_age)
    console.print(f"[green]Removed {removed} stale cache entries (>{max_age}h old)[/green]")


@cli.command()
@click.argument('ticker')
@click.option('--format', 'fmt', type=click.Choice(['json', 'csv', 'html']), default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: stdout)')
def export(ticker, fmt, output) -> None:
    """Export analysis data to JSON, CSV, or HTML.

    Example: stk export TSLA --format html -o tsla.html
    """
    from cli.engine import get_analysis, get_features
    import json
    import csv
    import io
    with console.status(f"[bold green]Fetching {ticker.upper()}..."):
        try:
            a = get_analysis(ticker)
            feats = get_features(ticker)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if fmt == 'json':
        data = {**a, 'features': feats}
        text = json.dumps(data, indent=2, default=str)
    elif fmt == 'csv':
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['field', 'value'])
        w.writerow(['ticker', a['ticker']])
        w.writerow(['price', a['price']])
        w.writerow(['change_pct', a['change_pct']])
        for h, d in a['horizons'].items():
            w.writerow([f'{h}_direction', d['direction']])
            w.writerow([f'{h}_confidence', d['confidence']])
        for k, v in feats.items():
            w.writerow([k, v])
        text = buf.getvalue()
    else:  # html
        tk = a['ticker']
        chg = a['change_pct']
        chg_color = '#22c55e' if chg >= 0 else '#ef4444'
        rows = ''
        for h in ('short', 'medium', 'long'):
            d = a['horizons'][h]
            label = {'short': 'Short-Term', 'medium': 'Medium-Term', 'long': 'Long-Term'}[h]
            color = '#22c55e' if d['direction'] == 'bullish' else ('#ef4444' if d['direction'] == 'bearish' else '#eab308')
            conf = int(d['confidence'] * 100) if d['confidence'] <= 1 else int(d['confidence'])
            rows += f'<tr><td>{label}</td><td style="color:{color};font-weight:bold">{d["direction"].upper()}</td><td>{conf}%</td><td>${d.get("stop","N/A")}</td><td>${d.get("target","N/A")}</td></tr>\n'
        bull = ''.join(f'<li style="color:#22c55e">{s}</li>' for s in a.get('bullish', [])[:5])
        bear = ''.join(f'<li style="color:#ef4444">{s}</li>' for s in a.get('bearish', [])[:5])
        feat_rows = ''.join(f'<tr><td>{k}</td><td>{v:.4f}</td></tr>' for k, v in list(feats.items())[:30])
        text = f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>{tk} Analysis</title>
<style>body{{font-family:system-ui;max-width:800px;margin:2em auto;background:#0f172a;color:#e2e8f0}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}th,td{{padding:8px 12px;border:1px solid #334155;text-align:left}}
th{{background:#1e293b}}h1,h2{{color:#38bdf8}}.header{{background:#1e293b;padding:1em;border-radius:8px;margin-bottom:1em}}</style></head>
<body><div class="header"><h1>{tk} — {a.get("name",tk)}</h1>
<p>Price: <b>${a["price"]:.2f}</b> <span style="color:{chg_color}">({chg:+.1f}%)</span></p></div>
<h2>Predictions</h2><table><tr><th>Horizon</th><th>Direction</th><th>Confidence</th><th>Stop</th><th>Target</th></tr>{rows}</table>
<h2>Signals</h2><div style="display:flex;gap:2em"><div><h3>Bullish</h3><ul>{bull}</ul></div><div><h3>Bearish</h3><ul>{bear}</ul></div></div>
<h2>Key Features</h2><table><tr><th>Feature</th><th>Value</th></tr>{feat_rows}</table>
<p style="color:#64748b;font-size:0.8em">Generated by stk CLI</p></body></html>'''

    if output:
        Path(output).write_text(text)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(text)


@cli.command()
@click.argument('ticker')
@click.option('--days', default=365, type=int, help='Days of history (default: 365)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--detailed', is_flag=True, help='Show per-trade breakdown')
def backtest(ticker, days, as_json, detailed) -> None:
    """Run backtest for a ticker.

    Example: stk backtest TSLA --days 180
    """
    from cli.engine import run_backtest
    with console.status(f"[bold green]Backtesting {ticker.upper()} ({days} days)..."):
        try:
            result = run_backtest(ticker, days, detailed=detailed)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if as_json:
        import json; click.echo(json.dumps(result, default=str)); return

    console.print(Panel(
        f"Sharpe: {result['sharpe_ratio']:.2f}  "
        f"Max DD: {result['max_drawdown']:.1%}  "
        f"Win Rate: {result['win_rate']:.0%}  "
        f"Profit Factor: {result['profit_factor']:.2f}",
        title=f"{ticker.upper()} Backtest ({days}d)", border_style="cyan"))
    console.print(f"  Trades: {result['total_trades']}  "
                  f"Gross Return: {result['gross_return']:.2f}%  "
                  f"Net Return: {result['net_return']:.2f}%  "
                  f"Avg Holding: {result['avg_holding_period']:.1f}d")
    if result.get('avg_position_size'):
        console.print(f"  Avg Position Size: {result['avg_position_size']:.0%} (confidence-weighted)")
    if 'win_rate_by_horizon' in result:
        wrh = result['win_rate_by_horizon']
        console.print(f"  Win Rate by Horizon: Short {wrh.get('short', 0)*100:.0f}% | "
                      f"Medium {wrh.get('medium', 0)*100:.0f}% | Long {wrh.get('long', 0)*100:.0f}%")
    console.print(f"  Costs: Slippage {result.get('slippage', 0)*100:.2f}% | "
                  f"Spread {result.get('spread', 0)*100:.2f}% | "
                  f"Commission ${result.get('commission', 0):.2f}")

    if detailed and 'trades' in result:
        t_table = Table(title="Per-Trade Breakdown", box=box.ROUNDED)
        t_table.add_column("Day", justify="right")
        t_table.add_column("Price", justify="right")
        t_table.add_column("Signal")
        t_table.add_column("Conf", justify="right")
        t_table.add_column("Size", justify="right")
        t_table.add_column("Gross %", justify="right")
        t_table.add_column("Net %", justify="right")
        for tr in result['trades'][:50]:  # limit display
            color = "green" if tr['return_net'] > 0 else "red"
            t_table.add_row(
                str(tr['day']), f"${tr['price']:.2f}", tr['signal'],
                f"{tr.get('confidence', 0)*100:.0f}%",
                f"{tr.get('position_size', 1.0):.0%}",
                f"[{color}]{tr['return_gross']:+.2f}%[/{color}]",
                f"[{color}]{tr['return_net']:+.2f}%[/{color}]")
        console.print(t_table)
        if len(result['trades']) > 50:
            console.print(f"[dim]... and {len(result['trades']) - 50} more trades[/dim]")


DEFAULT_SCREEN_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'JPM']


@cli.command()
@click.argument('ticker')
@click.option('--date', 'date_str', required=True, help='Historical date (YYYY-MM-DD)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def replay(ticker, date_str, as_json) -> None:
    """Replay analysis as of a historical date.

    Shows what the model would have predicted and whether it was correct.

    Example: stk replay TSLA --date 2025-06-15
    """
    from cli.engine import replay_analysis
    with console.status(f"[bold green]Replaying {ticker.upper()} @ {date_str}..."):
        try:
            r = replay_analysis(ticker, date_str)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if as_json:
        import json; click.echo(json.dumps(r, default=str)); return

    model_tag = " [dim](per-ticker model)[/dim]" if r.get('has_ticker_model') else " [dim](global model)[/dim]"
    console.print(Panel(
        f"Price: ${r['price']:.2f} ({r['change_pct']:+.1f}%)  Volume: {_fmt_vol(r['volume'])}{model_tag}",
        title=f"{r['ticker']} — {r['date']}", border_style="cyan"))

    labels = {'short': 'SHORT-TERM', 'medium': 'MEDIUM-TERM', 'long': 'LONG-TERM'}
    for h in ('short', 'medium', 'long'):
        d = r['horizons'][h]
        console.print(f"\n[bold]{labels[h]}[/bold]")
        verdict = d.get('conviction_verdict') or _verdict(d['direction'], d['confidence'])
        console.print(f"  Predicted: {verdict}")
        if d.get('vol_zscore') is not None:
            console.print(f"  Expected: {d['prediction']*100:+.2f}% ({d['vol_zscore']:.1f}σ — {d.get('vol_zscore_desc', '')})")
        if d.get('shap_text'):
            for line in d['shap_text'].split('\n'):
                console.print(f"  {line}")
        out = r['outcome'].get(h)
        if out:
            o_color = "green" if out['correct'] else "red"
            console.print(f"  Actual: [{o_color}]{out['actual_return']:+.1f}% → ${out['future_price']:.2f}[/{o_color}]  "
                          f"[{'green' if out['correct'] else 'red'}]{'✓ Correct' if out['correct'] else '✗ Wrong'}[/]")
        else:
            console.print("  Actual: [dim]No future data available[/dim]")


@cli.command('replay-range')
@click.argument('ticker')
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def replay_range_cmd(ticker, start, end, as_json) -> None:
    """Replay analysis across a date range.

    Shows prediction accuracy over time.

    Example: stk replay-range TSLA --start 2025-01-01 --end 2025-06-01
    """
    from cli.engine import replay_range
    with console.status(f"[bold green]Replaying {ticker.upper()} from {start} to {end}..."):
        try:
            results = replay_range(ticker, start, end)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if not results:
        console.print("[yellow]No data found for the given range.[/yellow]")
        return

    if as_json:
        import json; click.echo(json.dumps(results, default=str)); return

    t = Table(title=f"{ticker.upper()} Replay: {start} → {end}", box=box.ROUNDED)
    t.add_column("Date", style="bold")
    t.add_column("Price", justify="right")
    t.add_column("Short", justify="center")
    t.add_column("Medium", justify="center")
    t.add_column("Long", justify="center")

    correct_counts = {'short': 0, 'medium': 0, 'long': 0}
    total_counts = {'short': 0, 'medium': 0, 'long': 0}

    for r in results:
        row = [r['date'], f"${r['price']:.2f}"]
        for h in ('short', 'medium', 'long'):
            d = r['horizons'][h]
            out = r['outcome'].get(h)
            tier = d.get('conviction_tier', '')
            tier_icon = {'HIGH': '🟢', 'MODERATE': '🟡', 'LOW': '⚪'}.get(tier, '')
            verdict = tier_icon or ("🟢" if d['direction'] == 'bullish' else "🔴" if d['direction'] == 'bearish' else "🟡")
            if out:
                total_counts[h] += 1
                if out['correct']:
                    correct_counts[h] += 1
                    verdict += " ✓"
                else:
                    verdict += " ✗"
            row.append(verdict)
        t.add_row(*row)

    console.print(t)
    # Summary
    for h in ('short', 'medium', 'long'):
        total = total_counts[h]
        if total > 0:
            acc = correct_counts[h] / total * 100
            console.print(f"  {h.capitalize()} accuracy: {correct_counts[h]}/{total} ({acc:.0f}%)")


@cli.command()
@click.option('--criteria', type=click.Choice(['oversold', 'overbought', 'volume']), default='oversold',
              help='Screening criteria (default: oversold)')
@click.option('--tickers', '-t', default=None, help='Comma-separated tickers (default: top 10)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def screen(criteria, tickers, as_json) -> None:
    """Screen stocks for trading signals.

    Examples:
        stk screen                          # oversold in top 10
        stk screen --criteria volume        # volume spikes
        stk screen -t TSLA,AAPL,NVDA       # custom list
    """
    from cli.engine import screen_tickers
    ticker_list = tickers.split(',') if tickers else DEFAULT_SCREEN_TICKERS
    with console.status(f"[bold green]Screening {len(ticker_list)} tickers ({criteria})..."):
        try:
            results = screen_tickers(ticker_list, criteria)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if as_json:
        import json; click.echo(json.dumps(results)); return

    if not results:
        console.print(f"[yellow]No tickers match '{criteria}' criteria.[/yellow]")
        return

    t = Table(title=f"Screen: {criteria.upper()}", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    t.add_column("RSI", justify="right")
    t.add_column("Stoch %K", justify="right")
    t.add_column("Vol Ratio", justify="right")
    t.add_column("Reason")
    for r in results:
        color = "green" if r['change_pct'] >= 0 else "red"
        t.add_row(
            r['ticker'], f"${r['price']:.2f}",
            f"[{color}]{r['change_pct']:+.1f}%[/{color}]",
            f"{r['rsi']:.1f}", f"{r['stoch_k']:.1f}",
            f"{r['vol_ratio']:.1f}x", r['reason'])
    console.print(t)


SCAN_PRESETS = {
    'oversold': 'rsi<30 AND stoch<20',
    'overbought': 'rsi>70 AND stoch>80',
    'breakout': 'volume>2 AND adx>25',
    'momentum': 'rsi>50 AND macd>0 AND adx>20',
    'value': 'rsi<40 AND change>-10',
}


@cli.command()
@click.argument('filters', required=False)
@click.option('--tickers', '-t', default=None, help='Comma-separated tickers (default: top 10)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--preset', '-p', default=None,
              help='Use a preset filter (built-in: oversold, overbought, breakout, momentum, value; or custom)')
@click.option('--watch', '-w', default=0, type=int, help='Re-run every N seconds (0=off)')
def scan(filters, tickers, as_json, preset, watch) -> None:
    """Scan tickers with flexible filter expressions.

    Filters use AND to combine: 'rsi<30 AND volume>2'
    Operators: <, >, <=, >=, =
    Negative values supported: 'change>-5', 'williams7<-80'

    Supported: rsi, stoch, macd, bb, adx, volume (ratio), change (%),
    roc5, roc20, roc60, cmf, atrp, psar, adxr, force, cci, trix,
    ultosc, vortex, williams7, dpo, mass, emv

    Presets: oversold, overbought, breakout, momentum, value

    Examples:
        stk scan 'rsi<30 AND volume>2'
        stk scan --preset oversold
        stk scan -p breakout -t TSLA,AAPL,NVDA
        stk scan -p momentum --watch 60
    """
    if preset and not filters:
        if preset.lower() in SCAN_PRESETS:
            filters = SCAN_PRESETS[preset.lower()]
        else:
            from cli.db import get_custom_preset
            cp = get_custom_preset(preset)
            if cp:
                filters = cp['filters']
            else:
                console.print(f"[red]Error: Unknown preset '{preset}'. Use stk scan-presets to list.[/red]")
                return
    elif not filters and not preset:
        console.print("[red]Error: Provide a filter expression or use --preset[/red]")
        return
    from cli.engine import scan_tickers
    ticker_list = tickers.split(',') if tickers else DEFAULT_SCREEN_TICKERS
    with console.status(f"[bold green]Scanning {len(ticker_list)} tickers..."):
        try:
            results = scan_tickers(ticker_list, filters)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if as_json:
        import json; click.echo(json.dumps(results)); return

    if not results:
        console.print(f"[yellow]No tickers match filter: {filters}[/yellow]")
        return

    t = Table(title=f"Scan: {filters}", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    t.add_column("RSI", justify="right")
    t.add_column("Stoch %K", justify="right")
    t.add_column("Vol Ratio", justify="right")
    t.add_column("Match")
    for r in results:
        color = "green" if r['change_pct'] >= 0 else "red"
        t.add_row(
            r['ticker'], f"${r['price']:.2f}",
            f"[{color}]{r['change_pct']:+.1f}%[/{color}]",
            f"{r['rsi']:.1f}", f"{r['stoch_k']:.1f}",
            f"{r['vol_ratio']:.1f}x", r['reason'])
    console.print(t)

    if watch > 0:
        try:
            while True:
                console.print(f"\n[dim]Refreshing in {watch}s... (Ctrl+C to stop)[/dim]")
                time.sleep(watch)
                console.clear()
                with console.status(f"[bold green]Scanning {len(ticker_list)} tickers..."):
                    results = scan_tickers(ticker_list, filters)
                if not results:
                    console.print(f"[yellow]No tickers match filter: {filters}[/yellow]")
                    continue
                t2 = Table(title=f"Scan: {filters}", box=box.ROUNDED)
                t2.add_column("Ticker", style="bold")
                t2.add_column("Price", justify="right")
                t2.add_column("Change", justify="right")
                t2.add_column("RSI", justify="right")
                t2.add_column("Stoch %K", justify="right")
                t2.add_column("Vol Ratio", justify="right")
                t2.add_column("Match")
                for r in results:
                    color = "green" if r['change_pct'] >= 0 else "red"
                    t2.add_row(
                        r['ticker'], f"${r['price']:.2f}",
                        f"[{color}]{r['change_pct']:+.1f}%[/{color}]",
                        f"{r['rsi']:.1f}", f"{r['stoch_k']:.1f}",
                        f"{r['vol_ratio']:.1f}x", r['reason'])
                console.print(t2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch stopped.[/yellow]")


@cli.command()
@click.argument('ticker')
@click.option('--limit', '-n', default=20, type=int, help='Number of records (default: 20)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def history(ticker, limit, as_json) -> None:
    """Show prediction history with accuracy stats.

    Example: stk history TSLA
    """
    from cli.engine import get_prediction_history, get_price, prediction_accuracy
    preds = get_prediction_history(ticker)
    if not preds:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print(f"[yellow]No prediction history for {ticker.upper()}. Run 'stk analyze {ticker.upper()}' first.[/yellow]")
        return

    if as_json:
        import json
        acc = prediction_accuracy(ticker)
        click.echo(json.dumps({'predictions': preds[:limit], 'accuracy': acc}, default=str))
        return

    # Accuracy summary
    acc = prediction_accuracy(ticker)
    if acc.get('evaluated', 0) > 0:
        a_color = "green" if acc['accuracy'] >= 0.5 else "red"
        summary = (f"Predictions: {acc['total']}  Evaluated: {acc['evaluated']}  "
                   f"Correct: {acc['correct']}  Accuracy: [{a_color}]{acc['accuracy']:.0%}[/{a_color}]")
        by_conv = acc.get('by_conviction', {})
        if by_conv:
            parts = []
            for tier in ('HIGH', 'MODERATE', 'LOW'):
                s = by_conv.get(tier)
                if s and s['total'] > 0:
                    icon = {'HIGH': '🟢', 'MODERATE': '🟡', 'LOW': '⚪'}.get(tier, '')
                    parts.append(f"{icon}{tier}: {s['accuracy']:.0%} ({s['total']})")
            if parts:
                summary += f"\n  By conviction: {' | '.join(parts)}"
        console.print(Panel(summary, title=f"{ticker.upper()} Prediction Accuracy", border_style="cyan"))

    try:
        cur = get_price(ticker)['price']
    except Exception:
        cur = None

    t = Table(title=f"{ticker.upper()} Prediction History", box=box.ROUNDED)
    t.add_column("Date", style="dim")
    t.add_column("Horizon")
    t.add_column("Direction")
    t.add_column("Conviction")
    t.add_column("Confidence", justify="right")
    t.add_column("Price At", justify="right")
    t.add_column("Outcome", justify="right")

    for p in preds[:limit]:
        direction = p['direction']
        d_color = "green" if direction == 'bullish' else ("red" if direction == 'bearish' else "yellow")
        outcome = ""
        if cur and p['price_at']:
            actual_return = (cur - p['price_at']) / p['price_at'] * 100
            correct = (direction == 'bullish' and actual_return > 0) or \
                      (direction == 'bearish' and actual_return < 0)
            o_color = "green" if correct else "red"
            outcome = f"[{o_color}]{actual_return:+.1f}%[/{o_color}]"
        tier = p.get('conviction_tier', '')
        tier_icon = {'HIGH': '🟢', 'MODERATE': '🟡', 'LOW': '⚪'}.get(tier, '')
        tier_str = f"{tier_icon} {tier}" if tier else "-"
        t.add_row(
            p['created_at'][:16], p['horizon'],
            f"[{d_color}]{direction}[/{d_color}]",
            tier_str,
            f"{p['confidence']:.0%}" if p['confidence'] <= 1 else f"{p['confidence']:.0f}%",
            f"${p['price_at']:.2f}", outcome)
    console.print(t)


@cli.command()
@click.argument('sector_name')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def sector(sector_name, as_json) -> None:
    """Sector-level screening and analysis.

    Sectors: Technology, Healthcare, Finance, Consumer, Energy

    Example: stk sector Technology
    """
    from cli.engine import sector_analysis, SECTOR_TICKERS
    matched = None
    for s in SECTOR_TICKERS:
        if s.lower() == sector_name.lower():
            matched = s
            break
    if not matched:
        console.print(f"[red]Error: Unknown sector '{sector_name}'. Available: {', '.join(SECTOR_TICKERS.keys())}[/red]")
        return
    with console.status(f"[bold green]Analyzing {matched} sector..."):
        try:
            results = sector_analysis(matched)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if as_json:
        import json; click.echo(json.dumps(results)); return
    if not results:
        console.print(f"[yellow]No data for {matched} sector.[/yellow]")
        return
    t = Table(title=f"{matched} Sector", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    t.add_column("RSI", justify="right")
    t.add_column("MACD", justify="right")
    t.add_column("Vol Ratio", justify="right")
    for r in results:
        color = "green" if r['change_pct'] >= 0 else "red"
        macd_c = "green" if r['macd_hist'] > 0 else "red"
        t.add_row(
            r['ticker'], f"${r['price']:.2f}",
            f"[{color}]{r['change_pct']:+.1f}%[/{color}]",
            f"{r['rsi']:.1f}",
            f"[{macd_c}]{r['macd_hist']:+.2f}[/{macd_c}]",
            f"{r['vol_ratio']:.1f}x")
    console.print(t)
    avg_chg = sum(r['change_pct'] for r in results) / len(results)
    bullish_count = sum(1 for r in results if r['change_pct'] > 0)
    console.print(f"\n  Avg Change: {avg_chg:+.1f}%  Bullish: {bullish_count}/{len(results)}")


# --- Alerts ---

@cli.group()
def alerts() -> None:
    """Manage price alerts."""
    pass


@alerts.command('add')
@click.argument('ticker')
@click.option('--above', type=float, help='Alert when price goes above threshold')
@click.option('--below', type=float, help='Alert when price goes below threshold')
def alerts_add(ticker, above, below) -> None:
    """Add a price alert.

    Examples:
        stk alerts add TSLA --above 300
        stk alerts add AAPL --below 170
    """
    from cli.db import add_alert
    if not above and not below:
        console.print("[red]Error: Specify --above or --below threshold.[/red]")
        return
    if above is not None:
        if above <= 0:
            console.print("[red]Error: Threshold must be positive.[/red]")
            return
        add_alert(ticker, 'above', above)
        console.print(f"[green]Alert set: {ticker.upper()} above ${above:.2f}[/green]")
    if below is not None:
        if below <= 0:
            console.print("[red]Error: Threshold must be positive.[/red]")
            return
        add_alert(ticker, 'below', below)
        console.print(f"[green]Alert set: {ticker.upper()} below ${below:.2f}[/green]")


@alerts.command('list')
@click.argument('ticker', required=False)
@click.option('--all', 'show_all', is_flag=True, help='Include paused alerts')
def alerts_list(ticker, show_all) -> None:
    """Show active alerts.

    Examples:
        stk alerts list
        stk alerts list TSLA
        stk alerts list --all
    """
    from cli.db import get_alerts, get_all_active_alerts
    active = get_all_active_alerts(ticker) if show_all else get_alerts(ticker)
    if not active:
        console.print("[yellow]No active alerts.[/yellow]")
        return
    t = Table(title="Active Alerts", box=box.ROUNDED)
    t.add_column("ID", style="dim")
    t.add_column("Ticker", style="bold")
    t.add_column("Condition")
    t.add_column("Threshold", justify="right")
    t.add_column("Status")
    t.add_column("Created", style="dim")
    for a in active:
        paused = a.get('paused', 0)
        status = "[yellow]⏸ paused[/yellow]" if paused else "[green]active[/green]"
        t.add_row(str(a['id']), a['ticker'], a['condition'], f"${a['threshold']:.2f}", status, a['created_at'][:16])
    console.print(t)


@alerts.command('check')
def alerts_check() -> None:
    """Check alerts against current prices.

    Example: stk alerts check
    """
    from cli.engine import check_alerts
    with console.status("[bold green]Checking alerts..."):
        triggered = check_alerts()
    if not triggered:
        console.print("[green]No alerts triggered.[/green]")
        return
    for a in triggered:
        console.print(f"[bold red]🔔 {a['ticker']} is {a['condition']} ${a['threshold']:.2f} "
                       f"(current: ${a['current_price']:.2f})[/bold red]")


@alerts.command('remove')
@click.argument('alert_id', type=int)
def alerts_remove(alert_id) -> None:
    """Remove an alert by ID.

    Example: stk alerts remove 1
    """
    from cli.db import remove_alert
    remove_alert(alert_id)
    console.print(f"[yellow]Removed alert #{alert_id}[/yellow]")


@alerts.command('clear')
@click.argument('ticker', required=False)
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def alerts_clear(ticker, yes) -> None:
    """Remove all active alerts (optionally for a specific ticker).

    Examples:
        stk alerts clear            # clear all alerts
        stk alerts clear TSLA       # clear only TSLA alerts
        stk alerts clear -y         # skip confirmation
    """
    from cli.db import clear_alerts
    label = f"all {ticker.upper()} alerts" if ticker else "all alerts"
    if not yes:
        if not click.confirm(f"Remove {label}?"):
            console.print("[dim]Cancelled.[/dim]")
            return
    removed = clear_alerts(ticker)
    console.print(f"[yellow]Cleared {removed} alert(s).[/yellow]")


@alerts.command('history')
@click.argument('ticker', required=False)
@click.option('--limit', default=20, type=int, help='Max alerts to show')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def alerts_history(ticker, limit, as_json) -> None:
    """Show previously triggered alerts.

    Examples:
        stk alerts history              # all triggered alerts
        stk alerts history TSLA         # only TSLA triggered alerts
        stk alerts history --limit 5    # last 5 triggered
    """
    from cli.db import get_triggered_alerts
    triggered = get_triggered_alerts(ticker, limit)
    if as_json:
        import json; click.echo(json.dumps(triggered)); return
    if not triggered:
        console.print("[dim]No triggered alerts found.[/dim]")
        return
    t = Table(title="Triggered Alerts", box=box.ROUNDED)
    t.add_column("ID", justify="right")
    t.add_column("Ticker", style="bold")
    t.add_column("Condition")
    t.add_column("Threshold", justify="right")
    t.add_column("Triggered At")
    for a in triggered:
        t.add_row(str(a['id']), a['ticker'], a['condition'],
                  f"${a['threshold']:.2f}", a.get('created_at', 'N/A'))
    console.print(t)


@alerts.command('export')
@click.option('--output', '-o', default=None, help='Output file path (default: stdout)')
def alerts_export(output) -> None:
    """Export active alerts to JSON for backup/sharing.

    Examples:
        stk alerts export                    # print to stdout
        stk alerts export -o alerts.json     # save to file
    """
    from cli.db import get_alerts
    import json
    active = get_alerts()
    data = [{'ticker': a['ticker'], 'condition': a['condition'], 'threshold': a['threshold']} for a in active]
    if output:
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Exported {len(data)} alert(s) to {output}[/green]")
    else:
        click.echo(json.dumps(data, indent=2))


@alerts.command('import')
@click.argument('file_path', type=click.Path(exists=True))
def alerts_import(file_path) -> None:
    """Import alerts from a JSON file.

    Example: stk alerts import alerts.json
    """
    from cli.db import add_alert
    import json
    data = json.loads(Path(file_path).read_text())
    count = 0
    for a in data:
        if all(k in a for k in ('ticker', 'condition', 'threshold')):
            add_alert(a['ticker'], a['condition'], a['threshold'])
            count += 1
    console.print(f"[green]Imported {count} alert(s).[/green]")


@alerts.command('auto')
@click.argument('ticker')
@click.option('--conservative', is_flag=True, help='Tighter thresholds (0.5x ATR)')
@click.option('--aggressive', is_flag=True, help='Wider thresholds (2x ATR)')
def alerts_auto(ticker, conservative, aggressive) -> None:
    """Auto-set alerts at support/resistance levels.

    Analyzes the ticker and sets alerts at stop-loss and target levels
    for all timeframes.

    Examples:
      stk alerts auto TSLA
      stk alerts auto TSLA --conservative
      stk alerts auto TSLA --aggressive
    """
    from cli.engine import auto_alerts
    mode = 'conservative' if conservative else ('aggressive' if aggressive else 'normal')
    with console.status(f"[bold green]Analyzing {ticker.upper()} for auto-alerts ({mode})..."):
        result = auto_alerts(ticker, mode=mode)
    if not result:
        console.print("[yellow]No alerts could be set.[/yellow]")
        return
    t = Table(title=f"Auto-Alerts Set for {ticker.upper()} ({mode})", box=box.ROUNDED)
    t.add_column("Label")
    t.add_column("Condition")
    t.add_column("Threshold", justify="right")
    for a in result:
        t.add_row(a['label'], a['condition'], f"${a['threshold']:.2f}")
    console.print(t)


@alerts.command('smart')
@click.argument('ticker')
def alerts_smart(ticker) -> None:
    """Auto-generate alerts at key technical levels (BB bands, support/resistance, targets).

    Examples:
      stk alerts smart TSLA
    """
    from cli.engine import smart_alerts
    with console.status(f"[bold green]Generating smart alerts for {ticker.upper()}..."):
        result = smart_alerts(ticker)
    if not result:
        console.print("[yellow]No alerts could be generated.[/yellow]")
        return
    t = Table(title=f"Smart Alerts for {ticker.upper()}", box=box.ROUNDED)
    t.add_column("Label")
    t.add_column("Condition")
    t.add_column("Threshold", justify="right")
    for a in result:
        t.add_row(a['label'], a['condition'], f"${a['threshold']:.2f}")
    console.print(t)


@alerts.command('schedule')
def alerts_schedule() -> None:
    """Show how to set up recurring alert checks.

    Example: stk alerts schedule
    """
    console.print(Panel(
        "[bold]Recurring Alert Checks[/bold]\n\n"
        "Use cron or a task scheduler to run alert checks automatically:\n\n"
        "[cyan]# Check every 5 minutes during market hours (9:30-16:00 ET, Mon-Fri)[/cyan]\n"
        "*/5 9-15 * * 1-5 stk alerts check\n\n"
        "[cyan]# Check every 15 minutes[/cyan]\n"
        "*/15 9-15 * * 1-5 stk alerts check\n\n"
        "[cyan]# Add to crontab:[/cyan]\n"
        "crontab -e\n\n"
        "[cyan]# Or use watch for a quick session:[/cyan]\n"
        "watch -n 300 stk alerts check",
        title="Alert Scheduling", border_style="blue"))


@alerts.command('stats')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def alerts_stats(as_json) -> None:
    """Show alert statistics summary.

    Example: stk alerts stats
    """
    from cli.db import get_alerts, get_triggered_alerts
    active = get_alerts()
    triggered = get_triggered_alerts(limit=1000)
    tickers = set(a['ticker'] for a in active)
    result = {
        'active': len(active), 'triggered': len(triggered),
        'tickers_monitored': len(tickers),
        'by_condition': {},
    }
    for a in active:
        cond = a['condition']
        result['by_condition'][cond] = result['by_condition'].get(cond, 0) + 1
    if as_json:
        import json; click.echo(json.dumps(result)); return
    console.print(Panel(
        f"Active alerts: [bold]{len(active)}[/bold]\n"
        f"Triggered:     [bold]{len(triggered)}[/bold]\n"
        f"Tickers:       [bold]{len(tickers)}[/bold]",
        title="Alert Statistics", border_style="yellow"))
    if result['by_condition']:
        t = Table(box=box.SIMPLE)
        t.add_column("Condition")
        t.add_column("Count", justify="right")
        for cond, cnt in sorted(result['by_condition'].items()):
            t.add_row(cond, str(cnt))
        console.print(t)


@alerts.command('pause')
@click.option('--id', 'alert_id', type=int, help='Pause a specific alert by ID')
@click.option('--ticker', '-t', help='Pause all alerts for a ticker')
def alerts_pause(alert_id, ticker) -> None:
    """Temporarily pause alerts (skip during checks).

    Examples:
        stk alerts pause --id 3
        stk alerts pause --ticker TSLA
    """
    from cli.db import pause_alert, pause_alerts_by_ticker
    if alert_id:
        if pause_alert(alert_id):
            console.print(f"[yellow]Paused alert #{alert_id}[/yellow]")
        else:
            console.print(f"[red]Alert #{alert_id} not found or already triggered.[/red]")
    elif ticker:
        n = pause_alerts_by_ticker(ticker)
        console.print(f"[yellow]Paused {n} alert(s) for {ticker.upper()}[/yellow]")
    else:
        console.print("[red]Provide --id or --ticker[/red]")


@alerts.command('resume')
@click.option('--id', 'alert_id', type=int, help='Resume a specific alert by ID')
@click.option('--ticker', '-t', help='Resume all alerts for a ticker')
def alerts_resume(alert_id, ticker) -> None:
    """Resume paused alerts.

    Examples:
        stk alerts resume --id 3
        stk alerts resume --ticker TSLA
    """
    from cli.db import resume_alert, resume_alerts_by_ticker
    if alert_id:
        if resume_alert(alert_id):
            console.print(f"[green]Resumed alert #{alert_id}[/green]")
        else:
            console.print(f"[red]Alert #{alert_id} not found or already triggered.[/red]")
    elif ticker:
        n = resume_alerts_by_ticker(ticker)
        console.print(f"[green]Resumed {n} alert(s) for {ticker.upper()}[/green]")
    else:
        console.print("[red]Provide --id or --ticker[/red]")


@cli.command('what-if')
@click.argument('ticker')
@click.option('--qty', type=float, required=True, help='Hypothetical shares')
@click.option('--remove', is_flag=True, help='Simulate removing instead of adding')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def what_if(ticker, qty, remove, as_json) -> None:
    """Simulate adding or removing a position.

    Examples:
        stk what-if NVDA --qty 5
        stk what-if TSLA --qty 10 --remove
        stk what-if AAPL --qty 20 --json
    """
    from cli.db import get_positions
    from cli.engine import get_price
    ticker = ticker.upper()
    try:
        pr = get_price(ticker)
        cur = pr['price']
    except Exception as e:
        console.print(f"[red]Error fetching {ticker}: {e}[/red]")
        return
    hyp_value = cur * qty
    pos = get_positions()
    total_current = 0
    for p in pos:
        try:
            total_current += get_price(p['ticker'])['price'] * p['qty']
        except Exception:
            total_current += p['entry_price'] * p['qty']
    if remove:
        new_total = max(0, total_current - hyp_value)
        action = "Remove"
    else:
        new_total = total_current + hyp_value
        action = "Add"
    weight = hyp_value / new_total * 100 if new_total > 0 else 100
    result = {
        'ticker': ticker, 'price': cur, 'qty': qty, 'action': action.lower(),
        'position_value': round(hyp_value, 2),
        'current_portfolio': round(total_current, 2),
        'new_portfolio': round(new_total, 2),
        'weight_pct': round(weight, 2),
    }
    if as_json:
        import json; click.echo(json.dumps(result)); return
    sign = "-" if remove else "+"
    console.print(Panel(
        f"[bold]{ticker}[/bold] @ ${cur:,.2f} × {qty:.0f} shares = ${hyp_value:,.2f}\n\n"
        f"Current portfolio: ${total_current:,.2f}\n"
        f"After {action.lower()}ing:   ${new_total:,.2f} ({sign}${hyp_value:,.2f})\n"
        f"{'Freed' if remove else 'New'} weight:    {weight:.1f}%",
        title=f"What-If: {action} {ticker}", border_style="cyan"))


# --- Sensitivity ---

@cli.command()
@click.argument('ticker')
@click.option('--top', 'top_n', default=5, type=int, help='Number of features to test (default: 5)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def sensitivity(ticker, top_n, as_json) -> None:
    """Show how predictions change when key features shift ±1 std.

    Example: stk sensitivity TSLA
    """
    from cli.engine import feature_sensitivity
    with console.status(f"[bold green]Computing feature sensitivity for {ticker.upper()}..."):
        try:
            result = feature_sensitivity(ticker, top_n=top_n)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return
    if as_json:
        import json
        console.print(json.dumps(result, indent=2, default=str))
        return
    base = result['base_probability']
    console.print(f"\n[bold]{result['ticker']}[/bold] Feature Sensitivity (base prob: {base:.1%})")
    console.print("  Shows how short-term UP probability changes when each feature moves ±1 std\n")
    for s in result['sensitivities']:
        delta_up = s['prob_plus_1std'] - base
        delta_dn = s['prob_minus_1std'] - base
        bar_up = '▲' if delta_up > 0 else '▼'
        bar_dn = '▲' if delta_dn > 0 else '▼'
        console.print(f"  {s['feature']:<30s} (current: {s['current_value']:.2f})")
        console.print(f"    +1σ → {s['prob_plus_1std']:.1%} ({bar_up}{abs(delta_up)*100:.1f}pp)  "
                       f"-1σ → {s['prob_minus_1std']:.1%} ({bar_dn}{abs(delta_dn)*100:.1f}pp)  "
                       f"sensitivity: {s['sensitivity']*100:.1f}pp")
    console.print()


# --- Correlate ---

@cli.command()
@click.argument('ticker1')
@click.argument('ticker2')
@click.option('--days', default=180, type=int, help='Days of history (default: 180)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def correlate(ticker1, ticker2, days, as_json) -> None:
    """Compute correlation between two tickers.

    Example: stk correlate TSLA AAPL --days 90
    """
    from cli.engine import correlate_tickers
    with console.status(f"[bold green]Computing correlation {ticker1.upper()} vs {ticker2.upper()}..."):
        try:
            r = correlate_tickers(ticker1, ticker2, days)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if as_json:
        import json; click.echo(json.dumps(r)); return

    corr = r['correlation']
    color = "green" if abs(corr) > 0.7 else ("yellow" if abs(corr) > 0.3 else "red")
    strength = "Strong" if abs(corr) > 0.7 else ("Moderate" if abs(corr) > 0.3 else "Weak")
    direction = "positive" if corr > 0 else "negative"

    console.print(Panel(
        f"Correlation: [{color}]{corr:+.4f}[/{color}] ({strength} {direction})\n"
        f"Rolling 30d: min {r['rolling_min']:+.4f}  max {r['rolling_max']:+.4f}  current {r['rolling_current']:+.4f}",
        title=f"{r['ticker1']} vs {r['ticker2']} ({days}d)", border_style="cyan"))


@cli.command('top-movers')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def top_movers_cmd(as_json) -> None:
    """Show biggest movers from your watchlist.

    Example: stk top-movers
    """
    from cli.db import get_watchlist
    from cli.engine import top_movers
    wl = get_watchlist()
    if not wl:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print("Watchlist empty. Use [bold]stk watch TICKER[/bold] to add.")
        return
    tickers = [w['ticker'] for w in wl]
    with console.status(f"[bold green]Fetching {len(tickers)} tickers..."):
        movers = top_movers(tickers)
    if as_json:
        import json; click.echo(json.dumps(movers, default=str)); return
    if not movers:
        console.print("[yellow]No data available.[/yellow]")
        return
    t = Table(title="Top Movers (Watchlist)", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Price", justify="right")
    t.add_column("Change", justify="right")
    t.add_column("Volume", justify="right")
    for m in movers:
        color = "green" if m['change_pct'] >= 0 else "red"
        t.add_row(m['ticker'], f"${m['price']:.2f}",
                   f"[{color}]{m['change_pct']:+.1f}%[/{color}]",
                   _fmt_vol(m['volume']))
    console.print(t)


@cli.command('portfolio-risk')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_risk_cmd(as_json) -> None:
    """Portfolio-level risk analysis (VaR, beta, volatility, Sharpe).

    Example: stk portfolio-risk
    """
    from cli.db import get_positions
    from cli.engine import portfolio_risk
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps({})); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    with console.status("[bold green]Computing portfolio risk..."):
        try:
            r = portfolio_risk(pos)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if as_json:
        import json; click.echo(json.dumps(r)); return
    sharpe_color = "green" if r.get('sharpe', 0) > 1 else ("yellow" if r.get('sharpe', 0) > 0 else "red")
    console.print(Panel(
        f"Total Value: ${r['total_value']:,.2f}\n"
        f"VaR (95%, 1-day): [{'red' if r['var_95'] < 0 else 'green'}]${r['var_95']:+,.2f}[/]\n"
        f"Beta vs SPY: {r['beta']:.2f}\n"
        f"Annualized Vol: {r['volatility']:.1%}\n"
        f"Sharpe Ratio: [{sharpe_color}]{r.get('sharpe', 0):.2f}[/{sharpe_color}]",
        title="Portfolio Risk", border_style="cyan"))
    if r.get('weights'):
        t = Table(title="Position Weights", box=box.SIMPLE)
        t.add_column("Ticker", style="bold")
        t.add_column("Weight", justify="right")
        for tk, w in r['weights'].items():
            t.add_row(tk, f"{w:.1%}")
        console.print(t)


@cli.command('portfolio-correlation')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_correlation_cmd(as_json) -> None:
    """Show correlation matrix of portfolio positions.

    Computes 90-day return correlations between all held positions.
    High correlation means positions move together (less diversification).

    Examples:
      stk portfolio-correlation
      stk portfolio-correlation --json
    """
    from cli.db import get_positions
    from cli.engine import portfolio_correlation
    pos = get_positions()
    if len(pos) < 2:
        console.print("[yellow]Need at least 2 positions for correlation analysis.[/yellow]")
        return
    with console.status("[bold green]Computing correlations..."):
        result = portfolio_correlation(pos)
    if as_json:
        console.print_json(data=result)
        return
    tickers = result['tickers']
    matrix = result['matrix']
    t = Table(title="Position Correlation Matrix (90-day)", box=box.ROUNDED)
    t.add_column("")
    for tk in tickers:
        t.add_column(tk, justify="right")
    for i, tk in enumerate(tickers):
        row = [tk]
        for j in range(len(tickers)):
            v = matrix[i][j]
            color = "green" if abs(v) < 0.3 else ("yellow" if abs(v) < 0.7 else "red")
            row.append(f"[{color}]{v:+.3f}[/{color}]")
        t.add_row(*row)
    console.print(t)
    if result['pairs']:
        console.print("\n[bold]Highest correlations:[/bold]")
        for p in result['pairs'][:5]:
            v = p['correlation']
            color = "green" if abs(v) < 0.3 else ("yellow" if abs(v) < 0.7 else "red")
            console.print(f"  {p['ticker1']}/{p['ticker2']}: [{color}]{v:+.3f}[/{color}]")


@cli.command('portfolio-diversification')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_diversification_cmd(as_json) -> None:
    """Show portfolio diversification score and sector breakdown.

    Analyzes sector concentration and provides a diversification grade (A-F).

    Examples:
      stk portfolio-diversification
      stk portfolio-diversification --json
    """
    from cli.db import get_positions
    from cli.engine import portfolio_diversification
    pos = get_positions()
    if not pos:
        console.print("[yellow]No positions. Use 'stk hold' to add.[/yellow]")
        return
    with console.status("[bold green]Analyzing diversification..."):
        result = portfolio_diversification(pos)
    if as_json:
        console.print_json(data=result)
        return
    color = "green" if result['score'] >= 60 else ("yellow" if result['score'] >= 30 else "red")
    console.print(Panel(
        f"[bold {color}]Score: {result['score']}/100  Grade: {result['grade']}[/bold {color}]",
        title="Portfolio Diversification", border_style=color
    ))
    if result['sectors']:
        t = Table(title="Sector Breakdown", box=box.ROUNDED)
        t.add_column("Sector")
        t.add_column("Positions", justify="right")
        t.add_column("Weight", justify="right")
        total = sum(result['sectors'].values())
        for sec, cnt in sorted(result['sectors'].items(), key=lambda x: -x[1]):
            t.add_row(sec, str(cnt), f"{cnt/total*100:.0f}%")
        console.print(t)
    if result['suggestions']:
        console.print("\n[bold]Suggestions:[/bold]")
        for s in result['suggestions']:
            console.print(f"  • {s}")


@cli.command('portfolio-optimize')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_optimize_cmd(as_json) -> None:
    """Suggest optimized portfolio weights (min-variance, max-Sharpe, equal).

    Example: stk portfolio-optimize
    """
    from cli.db import get_positions
    from cli.engine import portfolio_optimize
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps({})); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    if len(pos) < 2:
        console.print("[yellow]Need at least 2 positions to optimize.[/yellow]")
        return
    with console.status("[bold green]Optimizing portfolio..."):
        try:
            r = portfolio_optimize(pos)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if as_json:
        import json; click.echo(json.dumps(r)); return

    for strategy, label in [('current', 'Current'), ('min_variance', 'Min Variance'), ('max_sharpe', 'Max Sharpe'), ('equal_weight', 'Equal Weight')]:
        data = r[strategy]
        sc = "green" if data['sharpe'] > 1 else ("yellow" if data['sharpe'] > 0 else "red")
        t = Table(title=f"{label} Portfolio", box=box.SIMPLE)
        t.add_column("Ticker", style="bold")
        t.add_column("Weight", justify="right")
        for tk, w in data['weights'].items():
            t.add_row(tk, f"{w:.1%}")
        console.print(t)
        console.print(f"  Return: {data['return']:.1%}  Vol: {data['volatility']:.1%}  Sharpe: [{sc}]{data['sharpe']:.2f}[/{sc}]\n")


@cli.command('portfolio-rebalance')
@click.option('--strategy', type=click.Choice(['max_sharpe', 'min_variance', 'equal_weight']),
              default='max_sharpe', help='Target strategy (default: max_sharpe)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_rebalance_cmd(strategy, as_json) -> None:
    """Show trades needed to rebalance portfolio to target weights.

    Example: stk portfolio-rebalance --strategy min_variance
    """
    from cli.db import get_positions
    from cli.engine import portfolio_rebalance
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    if len(pos) < 2:
        console.print("[yellow]Need at least 2 positions to rebalance.[/yellow]")
        return
    with console.status("[bold green]Computing rebalance trades..."):
        try:
            trades = portfolio_rebalance(pos, strategy)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return
    if as_json:
        import json; click.echo(json.dumps(trades)); return
    if not trades:
        console.print("[green]Portfolio is already balanced.[/green]")
        return
    t = Table(title=f"Rebalance Trades ({strategy.replace('_', ' ').title()})", box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Action", justify="center")
    t.add_column("Shares", justify="right")
    t.add_column("Value", justify="right")
    t.add_column("Current %", justify="right")
    t.add_column("Target %", justify="right")
    for tr in trades:
        color = "green" if tr['action'] == 'BUY' else "red"
        t.add_row(tr['ticker'], f"[{color}]{tr['action']}[/{color}]",
                   f"{tr['shares']:.1f}", f"${tr['value']:,.2f}",
                   f"{tr['current_weight']:.1%}", f"{tr['target_weight']:.1%}")
    console.print(t)


@cli.command('portfolio-summary')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_summary_cmd(as_json) -> None:
    """Combined portfolio overview: risk, diversification, and correlation.

    Example: stk portfolio-summary
    """
    from cli.db import get_positions
    from cli.engine import portfolio_risk, portfolio_diversification, portfolio_correlation
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps({})); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    with console.status("[bold green]Computing portfolio summary..."):
        risk = portfolio_risk(pos)
        div = portfolio_diversification(pos)
        corr = portfolio_correlation(pos) if len(pos) >= 2 else None

    if as_json:
        import json
        click.echo(json.dumps({'risk': risk, 'diversification': div, 'correlation': corr}, default=str))
        return

    # Risk panel
    beta_c = "green" if 0.8 <= risk['beta'] <= 1.2 else "yellow"
    sharpe_c = "green" if risk.get('sharpe', 0) > 1 else "yellow" if risk.get('sharpe', 0) > 0 else "red"
    risk_lines = (
        f"Total Value: [bold]${risk['total_value']:,.2f}[/bold]\n"
        f"VaR (95%):   [red]${risk['var_95']:,.2f}[/red]\n"
        f"Beta:        [{beta_c}]{risk['beta']:.2f}[/{beta_c}]\n"
        f"Volatility:  {risk.get('volatility', 0):.1%}\n"
        f"Sharpe:      [{sharpe_c}]{risk.get('sharpe', 0):.2f}[/{sharpe_c}]"
    )
    console.print(Panel(risk_lines, title="📊 Risk", border_style="cyan"))

    # Diversification panel
    grade_c = {"A": "green", "B": "green", "C": "yellow", "D": "red", "F": "red"}.get(div['grade'], "white")
    div_lines = f"Score: [{grade_c}]{div['score']}/100 ({div['grade']})[/{grade_c}]\n"
    div_lines += "Sectors: " + ", ".join(f"{s} ({c})" for s, c in div['sectors'].items())
    if div['suggestions']:
        div_lines += "\n" + "\n".join(f"  💡 {s}" for s in div['suggestions'])
    console.print(Panel(div_lines, title="🎯 Diversification", border_style="green"))

    # Correlation panel
    if corr and corr.get('pairs'):
        corr_lines = []
        for p in corr['pairs'][:5]:
            c = "red" if abs(p['correlation']) > 0.7 else "yellow" if abs(p['correlation']) > 0.4 else "green"
            corr_lines.append(f"{p['ticker1']}/{p['ticker2']}: [{c}]{p['correlation']:+.3f}[/{c}]")
        console.print(Panel("\n".join(corr_lines), title="🔗 Top Correlations", border_style="magenta"))


@cli.command('portfolio-stress')
@click.option('--scenario', '-s', type=click.Choice(['market_crash', 'correction', 'rate_hike', 'rally', 'black_swan'],
              case_sensitive=False), default=None, help='Specific scenario to test')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_stress_cmd(scenario, as_json) -> None:
    """Stress test portfolio under various market scenarios.

    Scenarios: market_crash (-20%), correction (-10%), rate_hike (-5%),
    rally (+15%), black_swan (-35%). Uses beta-adjusted shocks.

    Examples:
        stk portfolio-stress
        stk portfolio-stress --scenario market_crash
    """
    from cli.db import get_positions
    from cli.engine import portfolio_stress_test
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps([])); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    with console.status("[bold green]Running stress tests..."):
        results = portfolio_stress_test(pos, scenario)
    if as_json:
        import json; click.echo(json.dumps(results)); return
    for r in results:
        color = "green" if r['total_impact'] >= 0 else "red"
        t = Table(title=f"📉 {r['label']}", box=box.ROUNDED)
        t.add_column("Ticker", style="bold")
        t.add_column("Before", justify="right")
        t.add_column("After", justify="right")
        t.add_column("Impact", justify="right")
        t.add_column("Beta", justify="right")
        for d in r['details']:
            dc = "green" if d['impact'] >= 0 else "red"
            t.add_row(d['ticker'], f"${d['before']:,.2f}", f"${d['after']:,.2f}",
                      f"[{dc}]${d['impact']:+,.2f}[/{dc}]", f"{d['beta']:.2f}")
        console.print(t)
        console.print(f"  Total: [{color}]${r['total_impact']:+,.2f} ({r['impact_pct']:+.1f}%)[/{color}]\n")


@cli.command('portfolio-performance')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def portfolio_performance_cmd(as_json) -> None:
    """Show portfolio performance with daily/weekly/monthly returns.

    Example: stk portfolio-performance
    """
    from cli.db import get_positions
    pos = get_positions()
    if not pos:
        if as_json:
            import json; click.echo(json.dumps({})); return
        console.print("No positions. Use [bold]stk hold TICKER --entry PRICE --qty SHARES[/bold] to add.")
        return
    from cli.engine import portfolio_performance
    with console.status("[bold green]Calculating performance..."):
        perf = portfolio_performance(pos)
    if as_json:
        import json; click.echo(json.dumps(perf)); return
    color = "green" if perf['total_pnl'] >= 0 else "red"
    console.print(Panel(
        f"Value: ${perf['total_value']:,.2f}  Cost: ${perf['total_cost']:,.2f}  "
        f"P&L: [{color}]${perf['total_pnl']:+,.2f} ({perf['total_return_pct']:+.1f}%)[/{color}]",
        title="Portfolio Performance"))
    t = Table(box=box.ROUNDED)
    t.add_column("Ticker", style="bold")
    t.add_column("Value", justify="right")
    t.add_column("Total", justify="right")
    t.add_column("Daily", justify="right")
    t.add_column("Weekly", justify="right")
    t.add_column("Monthly", justify="right")
    for h in perf['holdings']:
        def _c(v) -> str: return "green" if v >= 0 else "red"
        t.add_row(h['ticker'], f"${h['value']:,.2f}",
                  f"[{_c(h['total_return_pct'])}]{h['total_return_pct']:+.1f}%[/{_c(h['total_return_pct'])}]",
                  f"[{_c(h['daily_return_pct'])}]{h['daily_return_pct']:+.1f}%[/{_c(h['daily_return_pct'])}]",
                  f"[{_c(h['weekly_return_pct'])}]{h['weekly_return_pct']:+.1f}%[/{_c(h['weekly_return_pct'])}]",
                  f"[{_c(h['monthly_return_pct'])}]{h['monthly_return_pct']:+.1f}%[/{_c(h['monthly_return_pct'])}]")
    console.print(t)


@cli.command('portfolio-history')
@click.option('--limit', '-n', default=30, type=int, help='Number of snapshots to show (default: 30)')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--save', 'do_save', is_flag=True, help='Save a snapshot of current portfolio state')
def portfolio_history(limit, as_json, do_save) -> None:
    """Show portfolio value over time from saved snapshots.

    Use --save to record the current portfolio state. Run periodically
    (e.g., via cron) to build a history.

    Examples:
        stk portfolio-history --save       # save current snapshot
        stk portfolio-history              # show history
        stk portfolio-history --limit 60   # show more history
    """
    from cli.db import get_positions, get_portfolio_snapshots, save_portfolio_snapshot
    if do_save:
        pos = get_positions()
        if not pos:
            console.print("[yellow]No positions to snapshot.[/yellow]")
            return
        from cli.engine import get_price
        total_value = total_cost = 0
        for p in pos:
            try:
                cur = get_price(p['ticker'])['price']
            except Exception:
                cur = p['entry_price']
            total_value += cur * p['qty']
            total_cost += p['entry_price'] * p['qty']
        pnl = total_value - total_cost
        pnl_pct = (pnl / total_cost * 100) if total_cost else 0
        save_portfolio_snapshot(total_value, total_cost, pnl, pnl_pct, len(pos))
        console.print(f"[green]Snapshot saved: ${total_value:,.2f} ({pnl_pct:+.1f}%)[/green]")
        return

    snaps = get_portfolio_snapshots(limit=limit)
    if not snaps:
        console.print("[yellow]No snapshots yet. Use --save to record one.[/yellow]")
        return
    snaps = list(reversed(snaps))  # oldest first for display
    if as_json:
        import json; click.echo(json.dumps(snaps)); return
    t = Table(title="Portfolio History", box=box.ROUNDED)
    t.add_column("Date", style="dim")
    t.add_column("Value", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("P&L", justify="right")
    t.add_column("Return", justify="right")
    t.add_column("Positions", justify="right")
    for s in snaps:
        c = "green" if s['pnl'] >= 0 else "red"
        date_str = s['snapshot_at'][:10] if s.get('snapshot_at') else '?'
        t.add_row(date_str, f"${s['total_value']:,.2f}", f"${s['total_cost']:,.2f}",
                  f"[{c}]${s['pnl']:+,.2f}[/{c}]",
                  f"[{c}]{s['pnl_pct']:+.1f}%[/{c}]",
                  str(s['num_positions']))
    console.print(t)
    # Sparkline of portfolio value
    values = [s['total_value'] for s in snaps]
    if len(values) > 1:
        console.print(f"\n  Value trend: {_sparkline(values)}")


@cli.command('positions-export')
@click.option('--format', '-f', 'fmt', type=click.Choice(['csv', 'json'], case_sensitive=False),
              default='json', help='Export format (default: json)')
@click.option('--output', '-o', default=None, help='Output file path')
def positions_export(fmt, output) -> None:
    """Export positions to CSV or JSON.

    Examples:
        stk positions-export                    # JSON to stdout
        stk positions-export -f csv -o port.csv # CSV to file
    """
    from cli.db import get_positions
    from cli.engine import get_price
    pos = get_positions()
    if not pos:
        console.print("[yellow]No positions to export.[/yellow]")
        return
    rows = []
    for p in pos:
        try:
            cur = get_price(p['ticker'])['price']
        except Exception:
            cur = p['entry_price']
        pnl = (cur - p['entry_price']) * p['qty']
        rows.append({**p, 'current_price': cur, 'pnl': round(pnl, 2),
                     'pnl_pct': round((cur - p['entry_price']) / p['entry_price'] * 100, 2)})
    if fmt == 'json':
        import json
        data = json.dumps(rows, indent=2)
    else:
        import csv
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
        data = buf.getvalue()
    if output:
        Path(output).write_text(data)
        console.print(f"[green]Exported {len(rows)} positions to {output}[/green]")
    else:
        click.echo(data)


@cli.command('positions-import')
@click.argument('file_path', type=click.Path(exists=True))
def positions_import(file_path) -> None:
    """Import positions from a JSON file.

    The file should contain a list of objects with ticker, entry_price, qty fields.

    Example: stk positions-import portfolio.json
    """
    import json as _json
    from cli.db import add_position
    try:
        data = _json.loads(Path(file_path).read_text())
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        return
    if not isinstance(data, list):
        console.print("[red]Error: File must contain a JSON array of positions.[/red]")
        return
    count = 0
    for item in data:
        if isinstance(item, dict) and 'ticker' in item and 'entry_price' in item and 'qty' in item:
            add_position(item['ticker'], float(item['entry_price']), float(item['qty']))
            count += 1
    console.print(f"[green]Imported {count} positions from {file_path}[/green]")


@cli.command('scan-presets')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def scan_presets_cmd(as_json) -> None:
    """List all scan presets (built-in and custom).

    Example: stk scan-presets
    """
    from cli.db import get_custom_presets
    custom = get_custom_presets()
    all_presets = {**SCAN_PRESETS}
    for cp in custom:
        all_presets[cp['name']] = cp['filters']
    if as_json:
        import json; click.echo(json.dumps(all_presets)); return
    t = Table(title="Scan Presets", box=box.ROUNDED)
    t.add_column("Name", style="bold")
    t.add_column("Filters")
    t.add_column("Type")
    for name, filt in sorted(all_presets.items()):
        ptype = "custom" if name in [c['name'] for c in custom] else "built-in"
        t.add_row(name, filt, ptype)
    console.print(t)


@cli.command('scan-save')
@click.argument('name')
@click.argument('filters')
def scan_save(name, filters) -> None:
    """Save a custom scan preset.

    Examples:
        stk scan-save my_dip 'rsi<25 AND change<-3'
        stk scan --preset my_dip
    """
    from cli.db import save_custom_preset
    save_custom_preset(name, filters)
    console.print(f"[green]Saved preset '{name}': {filters}[/green]")


@cli.command('scan-delete')
@click.argument('name')
def scan_delete(name) -> None:
    """Delete a custom scan preset.

    Example: stk scan-delete my_dip
    """
    from cli.db import delete_custom_preset
    if delete_custom_preset(name):
        console.print(f"[green]Deleted preset '{name}'[/green]")
    else:
        console.print(f"[yellow]Preset '{name}' not found.[/yellow]")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def summary(as_json) -> None:
    """Combined portfolio + watchlist overview.

    Example: stk summary
    """
    from cli.db import get_positions, get_watchlist
    from cli.engine import get_price

    # Positions summary
    pos = get_positions()
    total_value = total_pnl = 0
    pos_rows = []
    for p in pos:
        try:
            pr = get_price(p['ticker'])
            cur = pr['price']
        except Exception:
            cur = p['entry_price']
        val = cur * p['qty']
        pnl = (cur - p['entry_price']) * p['qty']
        total_value += val
        total_pnl += pnl
        pos_rows.append((p['ticker'], cur, pnl, (cur - p['entry_price']) / p['entry_price'] * 100))

    if as_json:
        import json
        wl = get_watchlist()
        wl_data = []
        for w in wl:
            try:
                pr = get_price(w['ticker'])
                wl_data.append({'ticker': w['ticker'], 'price': pr['price'], 'change_pct': pr['change_pct']})
            except Exception:
                wl_data.append({'ticker': w['ticker'], 'price': None, 'change_pct': None})
        click.echo(json.dumps({
            'positions': [{'ticker': t, 'price': c, 'pnl': pnl, 'pnl_pct': pp} for t, c, pnl, pp in pos_rows],
            'total_value': total_value, 'total_pnl': total_pnl, 'watchlist': wl_data,
        }, default=str))
        return

    if pos_rows:
        pnl_color = "green" if total_pnl >= 0 else "red"
        console.print(Panel(
            f"Positions: {len(pos_rows)}  Value: ${total_value:,.2f}  "
            f"P&L: [{pnl_color}]${total_pnl:+,.2f}[/{pnl_color}]",
            title="Portfolio", border_style="cyan"))
        t = Table(box=box.SIMPLE)
        t.add_column("Ticker", style="bold")
        t.add_column("Price", justify="right")
        t.add_column("P&L", justify="right")
        for tk, cur, pnl, pnl_pct in pos_rows:
            c = "green" if pnl >= 0 else "red"
            t.add_row(tk, f"${cur:.2f}", f"[{c}]{pnl_pct:+.1f}%[/{c}]")
        console.print(t)
    else:
        console.print("[dim]No positions.[/dim]")

    # Watchlist summary
    wl = get_watchlist()
    if wl:
        wt = Table(title="Watchlist", box=box.SIMPLE)
        wt.add_column("Ticker", style="bold")
        wt.add_column("Price", justify="right")
        wt.add_column("Change", justify="right")
        for w in wl:
            try:
                pr = get_price(w['ticker'])
                c = "green" if pr['change'] >= 0 else "red"
                wt.add_row(w['ticker'], f"${pr['price']:.2f}", f"[{c}]{pr['change_pct']:+.1f}%[/{c}]")
            except Exception:
                wt.add_row(w['ticker'], "N/A", "N/A")
        console.print(wt)
    else:
        console.print("[dim]No watchlist items.[/dim]")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def heatmap(as_json) -> None:
    """Sector/ticker performance heatmap.

    Shows a color-coded grid of sector performance with key metrics.

    Example: stk heatmap
    """
    from cli.engine import SECTOR_TICKERS, get_price
    data = {}
    with console.status("[bold green]Fetching sector data..."):
        for sector, tickers in SECTOR_TICKERS.items():
            rows = []
            for t in tickers:
                try:
                    p = get_price(t)
                    rows.append({'ticker': t, 'price': p['price'], 'change_pct': p['change_pct']})
                except Exception:
                    continue
            if rows:
                avg_chg = sum(r['change_pct'] for r in rows) / len(rows)
                data[sector] = {'tickers': rows, 'avg_change': avg_chg}

    if as_json:
        import json
        click.echo(json.dumps(data, default=str))
        return

    for sector, info in data.items():
        avg = info['avg_change']
        sc = "green" if avg >= 0 else "red"
        t = Table(title=f"{sector} [{sc}]{avg:+.2f}%[/{sc}]", box=box.SIMPLE, show_header=True)
        t.add_column("Ticker", style="bold", width=6)
        t.add_column("Price", justify="right", width=10)
        t.add_column("Change", justify="right", width=8)
        t.add_column("Bar", width=12)
        for r in sorted(info['tickers'], key=lambda x: x['change_pct'], reverse=True):
            c = "green" if r['change_pct'] >= 0 else "red"
            bar_len = min(10, max(1, int(abs(r['change_pct']) * 2)))
            bar_char = "█" * bar_len
            t.add_row(r['ticker'], f"${r['price']:.2f}", f"[{c}]{r['change_pct']:+.2f}%[/{c}]", f"[{c}]{bar_char}[/{c}]")
        console.print(t)
        console.print()


def main() -> None:
    cli()


@cli.command()
@click.argument('shell', type=click.Choice(['bash', 'zsh', 'fish']), required=False)
@click.option('--install', is_flag=True, help='Install completion to shell config file')
def completion(shell, install) -> None:
    """Generate shell completion scripts.

    Examples:
        stk completion bash          # print bash completion script
        stk completion zsh           # print zsh completion script
        stk completion fish          # print fish completion script
        stk completion bash --install  # install to ~/.bashrc
        eval "$(stk completion bash)"  # activate in current shell
    """
    import os
    if not shell:
        # Auto-detect shell
        sh = os.environ.get('SHELL', '')
        if 'zsh' in sh:
            shell = 'zsh'
        elif 'fish' in sh:
            shell = 'fish'
        else:
            shell = 'bash'
        console.print(f"[dim]Detected shell: {shell}[/dim]")

    env_var = '_STK_COMPLETE'
    scripts = {
        'bash': (f'{env_var}=bash_source stk', '~/.bashrc'),
        'zsh': (f'{env_var}=zsh_source stk', '~/.zshrc'),
        'fish': (f'{env_var}=fish_source stk', '~/.config/fish/completions/stk.fish'),
    }
    source_cmd, rc_file = scripts[shell]

    if install:
        from pathlib import Path
        rc_path = Path(rc_file).expanduser()
        marker = '# stk shell completion'
        if shell == 'fish':
            rc_path.parent.mkdir(parents=True, exist_ok=True)
            line = f'{source_cmd} > {rc_file}'
            console.print(f"[yellow]Run:[/yellow] {line}")
        else:
            line = f'eval "$({source_cmd})"'
            if rc_path.exists() and marker in rc_path.read_text():
                console.print(f"[green]Completion already installed in {rc_file}[/green]")
                return
            with open(rc_path, 'a') as f:
                f.write(f'\n{marker}\n{line}\n')
            console.print(f"[green]Installed completion to {rc_file}[/green]")
            console.print(f"[dim]Restart your shell or run: source {rc_file}[/dim]")
    else:
        # Print the eval command for manual use
        source_cmd_str, _ = scripts[shell]
        console.print('[bold]Activate completion:[/bold]')
        console.print(f'  eval "$({source_cmd_str})"')
        console.print('\n[bold]Or install permanently:[/bold]')
        console.print(f'  stk completion {shell} --install')


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--fix', is_flag=True, help='Auto-fix common issues (create dirs, init db)')
def doctor(as_json, fix) -> None:
    """Check system health: API connectivity, database, models.

    Examples:
        stk doctor
        stk doctor --fix
    """
    checks = []
    fixed = []
    # Check/fix data directory
    from pathlib import Path
    stk_dir = Path.home() / '.stk'
    if not stk_dir.exists():
        if fix:
            stk_dir.mkdir(parents=True, exist_ok=True)
            fixed.append('Created ~/.stk directory')
        checks.append({'name': 'Data Dir', 'status': 'ok' if fix else 'fail', 'detail': '~/.stk ' + ('created' if fix else 'missing')})
    else:
        checks.append({'name': 'Data Dir', 'status': 'ok', 'detail': '~/.stk exists'})
    # Check database
    try:
        from cli.db import get_positions
        get_positions()
        checks.append({'name': 'Database', 'status': 'ok', 'detail': 'SQLite accessible'})
    except Exception as e:
        if fix:
            try:
                stk_dir.mkdir(parents=True, exist_ok=True)
                get_positions()
                checks.append({'name': 'Database', 'status': 'ok', 'detail': 'Initialized'})
                fixed.append('Initialized database')
            except Exception as e2:
                checks.append({'name': 'Database', 'status': 'fail', 'detail': str(e2)})
        else:
            checks.append({'name': 'Database', 'status': 'fail', 'detail': str(e)})
    # Check config
    try:
        from cli.config import _load
        cfg = _load()
        checks.append({'name': 'Config', 'status': 'ok', 'detail': f'{len(cfg)} keys'})
    except Exception as e:
        checks.append({'name': 'Config', 'status': 'fail', 'detail': str(e)})
    # Check yfinance
    try:
        from cli.engine import get_price
        get_price('AAPL')
        checks.append({'name': 'Yahoo Finance', 'status': 'ok', 'detail': 'Price fetch working'})
    except Exception as e:
        checks.append({'name': 'Yahoo Finance', 'status': 'fail', 'detail': str(e)})
    # Check models
    try:
        from cli.engine import _ensure_models
        _ensure_models()
        checks.append({'name': 'ML Models', 'status': 'ok', 'detail': 'Models loaded'})
    except Exception as e:
        checks.append({'name': 'ML Models', 'status': 'fail', 'detail': str(e)})
    if as_json:
        import json; click.echo(json.dumps(checks)); return
    t = Table(title="System Health Check", box=box.ROUNDED)
    t.add_column("Component")
    t.add_column("Status")
    t.add_column("Detail")
    for c in checks:
        icon = "✅" if c['status'] == 'ok' else "❌"
        t.add_row(c['name'], icon, c['detail'])
    console.print(t)
    if fixed:
        console.print(f"\n[green]Fixed {len(fixed)} issue(s):[/green]")
        for f in fixed:
            console.print(f"  ✅ {f}")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def momentum(as_json) -> None:
    """Rank watchlist tickers by multi-timeframe momentum score.

    Combines ROC(5/20/60), Connors RSI, ADX, and Choppiness Index
    into a composite momentum score. Higher = stronger momentum.

    Example: stk momentum
    """
    from cli.db import get_watchlist
    from cli.engine import momentum_ranking, SECTOR_TICKERS
    wl = get_watchlist()
    if wl:
        tickers = [w['ticker'] for w in wl]
    else:
        tickers = []
        for v in SECTOR_TICKERS.values():
            tickers.extend(v)
    with console.status("[bold green]Computing momentum scores..."):
        results = momentum_ranking(tickers)

    if as_json:
        import json
        click.echo(json.dumps(results, default=str))
        return

    if not results:
        console.print("[yellow]No momentum data available[/yellow]")
        return

    t = Table(title="Momentum Ranking", box=box.ROUNDED, show_header=True)
    t.add_column("#", style="dim", width=3)
    t.add_column("Ticker", style="bold", width=6)
    t.add_column("Price", justify="right", width=10)
    t.add_column("ROC(5)", justify="right", width=8)
    t.add_column("ROC(20)", justify="right", width=8)
    t.add_column("ROC(60)", justify="right", width=8)
    t.add_column("CRSI", justify="right", width=6)
    t.add_column("ADX", justify="right", width=6)
    t.add_column("Trend", width=6)
    t.add_column("Score", justify="right", style="bold", width=8)

    for i, r in enumerate(results, 1):
        c5 = "green" if r['roc_5'] >= 0 else "red"
        c20 = "green" if r['roc_20'] >= 0 else "red"
        c60 = "green" if r['roc_60'] >= 0 else "red"
        cs = "green" if r['score'] >= 0 else "red"
        trend = "[green]↑ Yes[/green]" if r['trending'] else "[yellow]— No[/yellow]"
        t.add_row(
            str(i), r['ticker'], f"${r['price']:.2f}",
            f"[{c5}]{r['roc_5']:+.1f}%[/{c5}]",
            f"[{c20}]{r['roc_20']:+.1f}%[/{c20}]",
            f"[{c60}]{r['roc_60']:+.1f}%[/{c60}]",
            f"{r['connors_rsi']:.0f}", f"{r['adx']:.0f}",
            trend, f"[{cs}]{r['score']:+.1f}[/{cs}]",
        )
    console.print(t)


@cli.command('tax-report')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def tax_report(as_json) -> None:
    """Capital gains tax report for portfolio positions.

    Shows unrealized gains/losses for current positions and realized
    gains from sold positions. Classifies as short-term (<1yr) or long-term.

    Example: stk tax-report
    """
    from cli.db import get_positions
    from cli.engine import portfolio_tax_report
    positions = get_positions()
    with console.status("[bold green]Computing tax report..."):
        report = portfolio_tax_report(positions)

    if as_json:
        import json
        click.echo(json.dumps(report, default=str))
        return

    # Unrealized gains
    if report['unrealized']:
        t = Table(title="Unrealized Gains/Losses", box=box.ROUNDED, show_header=True)
        t.add_column("Ticker", style="bold", width=6)
        t.add_column("Qty", justify="right", width=6)
        t.add_column("Cost Basis", justify="right", width=12)
        t.add_column("Current", justify="right", width=12)
        t.add_column("Gain/Loss", justify="right", width=12)
        t.add_column("%", justify="right", width=8)
        t.add_column("Term", width=6)
        t.add_column("Days", justify="right", width=6)
        for u in report['unrealized']:
            c = "green" if u['gain'] >= 0 else "red"
            t.add_row(
                u['ticker'], str(u['qty']),
                f"${u['cost_basis']:,.2f}", f"${u['current_value']:,.2f}",
                f"[{c}]${u['gain']:+,.2f}[/{c}]",
                f"[{c}]{u['gain_pct']:+.1f}%[/{c}]",
                u['term'].upper(), str(u['days_held']),
            )
        console.print(t)
    else:
        console.print("[dim]No open positions[/dim]")

    # Realized gains
    if report['realized']:
        t2 = Table(title="Realized Gains/Losses", box=box.ROUNDED, show_header=True)
        t2.add_column("Ticker", style="bold", width=6)
        t2.add_column("Qty", justify="right", width=6)
        t2.add_column("Cost", justify="right", width=12)
        t2.add_column("Proceeds", justify="right", width=12)
        t2.add_column("Gain/Loss", justify="right", width=12)
        t2.add_column("Term", width=6)
        for r in report['realized']:
            c = "green" if r['gain'] >= 0 else "red"
            t2.add_row(
                r['ticker'], str(r['qty']),
                f"${r['cost_basis']:,.2f}", f"${r['proceeds']:,.2f}",
                f"[{c}]${r['gain']:+,.2f}[/{c}]", r['term'].upper(),
            )
        console.print(t2)

    # Summary
    console.print()
    panel_lines = []
    tu = report['total_unrealized']
    tr = report['total_realized']
    cu = "green" if tu >= 0 else "red"
    cr = "green" if tr >= 0 else "red"
    panel_lines.append(f"Unrealized: [{cu}]${tu:+,.2f}[/{cu}]")
    panel_lines.append(f"Realized:   [{cr}]${tr:+,.2f}[/{cr}]")
    if report['short_term_gains'] != 0:
        panel_lines.append(f"  Short-term: ${report['short_term_gains']:+,.2f}")
    if report['long_term_gains'] != 0:
        panel_lines.append(f"  Long-term:  ${report['long_term_gains']:+,.2f}")
    console.print(Panel('\n'.join(panel_lines), title="Tax Summary", border_style="cyan"))


@cli.command()
@click.argument('ticker')
@click.option('--limit', '-n', default=30, help='Number of predictions to show')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def journal(ticker, limit, as_json) -> None:
    """Show prediction journal with outcomes for a ticker.

    Displays a timeline of past predictions, whether they were correct,
    and summary statistics including accuracy by horizon and conviction tier.

    Examples:
      stk journal TSLA
      stk journal AAPL --limit 50
      stk journal NVDA --json
    """
    from cli.engine import get_prediction_journal
    with console.status(f"[bold green]Loading journal for {ticker.upper()}..."):
        result = get_prediction_journal(ticker, limit=limit)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    entries = result.get('entries', [])
    stats = result.get('stats', {})

    if not entries:
        console.print(f"[yellow]No predictions found for {ticker.upper()}. Run 'stk analyze {ticker.upper()}' first.[/yellow]")
        return

    console.print(Panel(f"Prediction Journal — {len(entries)} entries", title=f"{result['ticker']}", border_style="cyan"))

    # Timeline table
    t = Table(box=box.ROUNDED)
    t.add_column("Date", style="dim")
    t.add_column("Horizon")
    t.add_column("Direction")
    t.add_column("Confidence", justify="right")
    t.add_column("Conviction")
    t.add_column("Outcome")
    t.add_column("Return", justify="right")
    t.add_column("Reason", style="dim")

    for e in entries[:limit]:
        date_str = e['date'][:10] if e['date'] else ''
        dir_icon = '🟢' if e['direction'] == 'bullish' else '🔴'
        conf_str = f"{e['confidence']*100:.0f}%" if e['confidence'] else ''
        tier = e.get('conviction_tier', '')
        outcome = e.get('outcome', '')
        outcome_str = {'correct': '✅', 'wrong': '❌', 'pending': '⏳'}.get(outcome, '')
        ret_str = f"{e['actual_return']:+.1f}%" if e.get('actual_return') is not None else ''
        reason = (e.get('top_reason') or '')[:25]
        t.add_row(date_str, e['horizon'], f"{dir_icon} {e['direction']}", conf_str, tier, outcome_str, ret_str, reason)

    console.print(t)

    # Summary stats
    if any(h in stats for h in ('short', 'medium', 'long')):
        console.print("\n[bold]Accuracy by Horizon:[/bold]")
        for h in ('short', 'medium', 'long'):
            if h in stats:
                s = stats[h]
                bar_len = int(s['accuracy'] * 20)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                console.print(f"  {h.capitalize():8s}: {s['accuracy']*100:.0f}% ({s['correct']}/{s['total']}) {bar}")

    # Conviction tier accuracy
    by_conv = stats.get('by_conviction', {})
    if by_conv:
        console.print("\n[bold]Accuracy by Conviction:[/bold]")
        for tier in ('HIGH', 'MODERATE', 'LOW'):
            if tier in by_conv:
                s = by_conv[tier]
                console.print(f"  {tier:10s}: {s['accuracy']*100:.0f}% ({s['correct']}/{s['total']})")

    # Confidence distribution histogram
    conf_dist = stats.get('confidence_distribution', {})
    if conf_dist:
        max_count = max(conf_dist.values()) if conf_dist else 1
        console.print("\n[bold]Confidence Distribution:[/bold]")
        for bucket, count in conf_dist.items():
            bar_len = int(count / max_count * 20)
            bar = '█' * bar_len
            console.print(f"  {bucket:8s}: {bar} ({count})")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def leaderboard(as_json) -> None:
    """Rank all trained models by prediction quality.

    Shows a leaderboard of tickers sorted by walk-forward accuracy,
    health grade, and calibration quality.

    Examples:
      stk leaderboard
      stk leaderboard --json
    """
    from cli.engine import model_status
    statuses = model_status()
    if not statuses:
        console.print("[yellow]No trained models found. Run 'stk retrain TICKER' first.[/yellow]")
        return

    if as_json:
        import json
        click.echo(json.dumps(statuses, indent=2, default=str))
        return

    # Sort by average WF accuracy (descending)
    def sort_key(s):
        vals = [s.get(f'wf_{h}') for h in ('short', 'medium', 'long') if s.get(f'wf_{h}') is not None]
        return sum(vals) / len(vals) if vals else 0

    statuses.sort(key=sort_key, reverse=True)

    t = Table(title="Model Leaderboard", box=box.ROUNDED)
    t.add_column("#", justify="right", style="dim")
    t.add_column("Ticker", style="bold")
    t.add_column("Health")
    t.add_column("Short WF", justify="right")
    t.add_column("Medium WF", justify="right")
    t.add_column("Long WF", justify="right")
    t.add_column("Avg WF", justify="right")
    t.add_column("Brier", justify="right")
    t.add_column("Features", justify="right")
    t.add_column("Age", justify="right")

    icons = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
    for i, s in enumerate(statuses, 1):
        grade = s.get('health_grade', '?')
        icon = icons.get(grade, '⚪')
        vals = [s.get(f'wf_{h}') for h in ('short', 'medium', 'long') if s.get(f'wf_{h}') is not None]
        avg_wf = f"{sum(vals)/len(vals)*100:.1f}%" if vals else '-'
        short_wf = f"{s['wf_short']*100:.1f}%" if s.get('wf_short') is not None else '-'
        med_wf = f"{s['wf_medium']*100:.1f}%" if s.get('wf_medium') is not None else '-'
        long_wf = f"{s['wf_long']*100:.1f}%" if s.get('wf_long') is not None else '-'
        brier = f"{s['avg_brier']:.3f}" if s.get('avg_brier') is not None else '-'
        feats = f"{s.get('features', '?')}/{s.get('total_features', '?')}"
        age = f"{s['age_days']}d" if s.get('age_days') is not None else '-'
        t.add_row(str(i), s['ticker'], f"{icon} {grade}", short_wf, med_wf, long_wf, avg_wf, brier, feats, age)

    console.print(t)


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def ablation(ticker, as_json) -> None:
    """Feature group ablation study — which feature groups matter most.

    Measures how much prediction confidence drops when each feature group
    is removed, revealing which groups the model relies on most.

    Examples:
      stk ablation TSLA
      stk ablation AAPL --json
    """
    from cli.engine import feature_ablation
    with console.status(f"[bold green]Running ablation study for {ticker.upper()}..."):
        result = feature_ablation(ticker)

    if result.get('error'):
        console.print(f"[red]{result['error']}[/red]")
        return

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    console.print(Panel(f"Feature Ablation Study — {result['total_selected']} selected features",
                        title=f"{result['ticker']}", border_style="cyan"))

    baseline = result.get('baseline_confidence', {})
    console.print(f"Baseline confidence: short={baseline.get('short', 0)*100:.1f}%, "
                  f"medium={baseline.get('medium', 0)*100:.1f}%, long={baseline.get('long', 0)*100:.1f}%\n")

    t = Table(box=box.ROUNDED)
    t.add_column("Group", style="bold")
    t.add_column("Features", justify="right")
    t.add_column("Short Δ", justify="right")
    t.add_column("Medium Δ", justify="right")
    t.add_column("Long Δ", justify="right")
    t.add_column("Avg Impact", justify="right")
    t.add_column("Importance")

    for g in result.get('groups', []):
        drops = g['confidence_drop']
        avg = g['avg_drop']
        # Bar visualization
        bar_len = min(20, int(abs(avg) * 200))
        bar = '█' * bar_len
        color = 'red' if avg > 0.01 else ('yellow' if avg > 0 else 'green')
        t.add_row(
            g['group'],
            str(g['feature_count']),
            f"{drops.get('short', 0)*100:+.1f}%",
            f"{drops.get('medium', 0)*100:+.1f}%",
            f"{drops.get('long', 0)*100:+.1f}%",
            f"{avg*100:+.1f}%",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(t)
    console.print("\n[dim]Positive Δ = confidence drops when group removed (group is important)[/dim]")


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def streak(ticker, as_json) -> None:
    """Show prediction streak stats — consecutive correct/wrong predictions.

    Tracks the current prediction streak and historical best/worst streaks
    to help assess model reliability over time.

    Examples:
      stk streak TSLA
      stk streak AAPL --json
    """
    from cli.engine import prediction_streak
    result = prediction_streak(ticker)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if result.get('evaluated', 0) == 0:
        console.print(f"[yellow]No evaluated predictions for {ticker.upper()}. "
                      f"Run 'stk analyze {ticker.upper()}' and wait for outcomes.[/yellow]")
        return

    console.print(Panel(f"Prediction Streaks — {result['evaluated']} evaluated",
                        title=f"{result['ticker']}", border_style="cyan"))

    streak_type = result.get('streak_type', 'none')
    streak_len = result.get('current_streak', 0)
    icon = '🔥' if streak_type == 'correct' else '❄️'
    console.print(f"Current streak: {icon} {streak_len} {streak_type}")
    console.print(f"Best correct streak: ✅ {result.get('best_correct_streak', 0)}")
    console.print(f"Worst wrong streak: ❌ {result.get('worst_wrong_streak', 0)}")

    recent = result.get('recent', [])
    if recent:
        console.print("\n[bold]Recent predictions:[/bold]")
        for e in recent:
            icon = '✅' if e['correct'] else '❌'
            console.print(f"  {e['date']} {e['horizon']:7s} {e['direction']:8s} "
                          f"{e.get('conviction', ''):10s} {icon}")


@cli.command('top-features')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--top', 'top_n', default=20, help='Number of features to show')
def top_features_cmd(as_json, top_n) -> None:
    """Show most important features across all trained tickers.

    Aggregates feature importance weighted by walk-forward accuracy
    and stability across WF windows.

    Examples:
      stk top-features
      stk top-features --top 10 --json
    """
    from cli.engine import global_feature_importance
    result = global_feature_importance(top_n)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if not result['features']:
        console.print("[yellow]No trained models found. Run 'stk retrain TICKER' first.[/yellow]")
        return

    console.print(Panel(f"Across {result['ticker_count']} trained tickers",
                        title="Global Feature Importance", border_style="cyan"))

    t = Table(box=box.SIMPLE)
    t.add_column("#", style="dim", width=3)
    t.add_column("Feature", style="bold")
    t.add_column("Score", justify="right")
    t.add_column("Tickers", justify="center")
    t.add_column("Coverage", justify="right")

    for i, f in enumerate(result['features'], 1):
        bar_len = int(f['score'] / max(result['features'][0]['score'], 0.001) * 15)
        bar = '█' * bar_len + '░' * (15 - bar_len)
        t.add_row(
            str(i),
            f['name'],
            f"{f['score']:.3f} {bar}",
            str(f['ticker_count']),
            f"{f['pct_tickers']:.0f}%",
        )

    console.print(t)


@cli.command('model-diff')
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def model_diff_cmd(ticker, as_json) -> None:
    """Compare current model with previous retrain for a ticker.

    Shows changes in accuracy, health, top features, and calibration
    between the last two retrains.

    Examples:
      stk model-diff TSLA
      stk model-diff AAPL --json
    """
    from cli.engine import get_model_explanation
    info = get_model_explanation(ticker)
    if 'error' in info:
        if as_json:
            import json
            click.echo(json.dumps(info))
        else:
            console.print(f"[red]{info['error']}[/red]")
        return

    diff = {'ticker': info['ticker']}

    # Health trend diff
    ht = info.get('health_trend', [])
    if len(ht) >= 2:
        prev, curr = ht[-2], ht[-1]
        delta = curr['score'] - prev['score']
        diff['health'] = {
            'previous': {'score': prev['score'], 'grade': prev['grade']},
            'current': {'score': curr['score'], 'grade': curr['grade']},
            'delta': round(delta, 1),
            'improved': delta > 0,
        }

    # Feature changelog diff
    cl = info.get('feature_changelog', [])
    if len(cl) >= 2:
        prev_feats = set(cl[-2].get('top_features', []))
        curr_feats = set(cl[-1].get('top_features', []))
        diff['features'] = {
            'added': list(curr_feats - prev_feats),
            'removed': list(prev_feats - curr_feats),
            'unchanged': len(prev_feats & curr_feats),
        }

    # WF accuracy (current only, but show it)
    wf = info.get('wf_accuracy', {})
    if wf:
        diff['wf_accuracy'] = wf

    if as_json:
        import json
        click.echo(json.dumps(diff, indent=2, default=str))
        return

    console.print(Panel(f"Model comparison for {diff['ticker']}", border_style="cyan"))

    h = diff.get('health')
    if h:
        arrow = '📈' if h['improved'] else '📉'
        console.print(f"  Health: {h['previous']['grade']}({h['previous']['score']:.0f}) → "
                      f"{h['current']['grade']}({h['current']['score']:.0f}) "
                      f"({h['delta']:+.0f}) {arrow}")
    else:
        console.print("  [dim]No health history (need ≥2 retrains)[/dim]")

    f = diff.get('features')
    if f:
        from backend.models.explain import _readable_name
        console.print(f"\n  Feature changes: {f['unchanged']} unchanged")
        if f['added']:
            names = [_readable_name(n) for n in f['added'][:5]]
            console.print(f"    [green]+ {', '.join(names)}[/green]")
        if f['removed']:
            names = [_readable_name(n) for n in f['removed'][:5]]
            console.print(f"    [red]- {', '.join(names)}[/red]")
    else:
        console.print("  [dim]No feature changelog (need ≥2 retrains)[/dim]")

    wf = diff.get('wf_accuracy', {})
    if wf:
        console.print("\n  Current WF accuracy:")
        for hz in ('short', 'medium', 'long'):
            if hz in wf:
                console.print(f"    {hz.capitalize()}: {wf[hz]*100:.1f}%")


@cli.command()
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def drift(ticker, as_json) -> None:
    """Analyze feature drift for a ticker vs training distribution.

    Shows which features have shifted significantly since the model was trained,
    helping identify when a retrain is needed.

    Examples:
      stk drift TSLA
      stk drift AAPL --json
    """
    from cli.engine import feature_drift_analysis
    result = feature_drift_analysis(ticker)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    risk_colors = {'HIGH': 'red', 'MODERATE': 'yellow', 'LOW': 'green'}
    risk_icons = {'HIGH': '🔴', 'MODERATE': '🟡', 'LOW': '🟢'}
    rc = risk_colors.get(result['risk'], 'white')
    ri = risk_icons.get(result['risk'], '⚪')

    header = (f"Drift Risk: [{rc}]{ri} {result['risk']}[/{rc}] — {result['risk_desc']}\n"
              f"Avg drift: {result['avg_drift']}σ  |  "
              f"Severe (>3σ): {result['severe_drift']}  |  "
              f"Moderate (2-3σ): {result['moderate_drift']}  |  "
              f"Mild (1-2σ): {result['mild_drift']}")
    console.print(Panel(header, title=f"{result['ticker']} Feature Drift", border_style="cyan"))

    if result['top_drifted']:
        from backend.models.explain import _readable_name
        t = Table(box=box.SIMPLE)
        t.add_column("Feature", style="bold")
        t.add_column("Z-Score", justify="right")
        t.add_column("Current", justify="right")
        t.add_column("Train Mean", justify="right")
        t.add_column("Direction", justify="center")

        for d in result['top_drifted']:
            z = d['z_score']
            z_color = 'red' if abs(z) > 3 else ('yellow' if abs(z) > 2 else 'white')
            direction = '↑' if z > 0 else '↓'
            t.add_row(
                _readable_name(d['feature']),
                f"[{z_color}]{z:+.1f}σ[/{z_color}]",
                f"{d['current']:.3f}",
                f"{d['train_mean']:.3f}",
                direction,
            )
        console.print(t)

    if result.get('trained_at'):
        console.print(f"\n[dim]Model trained: {result['trained_at']}[/dim]")


@cli.command('backtest-compare')
@click.argument('ticker')
@click.option('--days', default=365, help='Backtest period in days')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def backtest_compare_cmd(ticker, days, as_json) -> None:
    """Run backtest and compare with previous baseline.

    First run saves a baseline. Subsequent runs compare against it,
    showing which metrics improved or degraded after retraining.

    Examples:
      stk backtest-compare TSLA
      stk backtest-compare AAPL --days 180
    """
    from cli.engine import backtest_compare
    with console.status(f"[bold green]Running backtest comparison for {ticker.upper()}..."):
        result = backtest_compare(ticker, days)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    cur = result['current']
    console.print(Panel(f"Net Return: {cur['net_return']:+.2f}%  |  "
                        f"Sharpe: {cur['sharpe_ratio']:.2f}  |  "
                        f"Win Rate: {cur['win_rate']*100:.1f}%  |  "
                        f"Trades: {cur['total_trades']}",
                        title=f"{result['ticker']} Backtest", border_style="cyan"))

    comp = result.get('comparison')
    if comp is None:
        console.print("[dim]Baseline saved. Run again after retraining to compare.[/dim]")
        return

    t = Table(box=box.SIMPLE)
    t.add_column("Metric", style="bold")
    t.add_column("Current", justify="right")
    t.add_column("Baseline", justify="right")
    t.add_column("Delta", justify="right")
    t.add_column("", justify="center")

    labels = {
        'net_return': 'Net Return (%)', 'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown', 'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor', 'total_trades': 'Total Trades',
    }
    for m, label in labels.items():
        c = comp[m]
        icon = '✅' if c['improved'] else '❌'
        delta_str = f"{c['delta']:+.4f}"
        if m in ('net_return',):
            delta_str = f"{c['delta']:+.2f}%"
        elif m == 'win_rate':
            delta_str = f"{c['delta']*100:+.1f}%"
        t.add_row(label, f"{c['current']:.4f}", f"{c['baseline']:.4f}", delta_str, icon)

    console.print(t)
    console.print(f"\n[dim]Baseline from: {result['baseline'].get('saved_at', 'unknown')}[/dim]")


@cli.command('interactions')
@click.argument('ticker')
@click.option('--top', 'top_n', default=10, help='Number of interaction pairs')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def interactions_cmd(ticker, top_n, as_json) -> None:
    """Show top feature interaction pairs for a ticker's model.

    Reveals which feature combinations the model relies on most,
    helping understand complex prediction logic.

    Examples:
      stk interactions TSLA
      stk interactions AAPL --top 5 --json
    """
    from cli.engine import feature_interactions
    with console.status(f"[bold green]Computing feature interactions for {ticker.upper()}..."):
        result = feature_interactions(ticker, top_n)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    method = result.get('method', 'unknown')
    method_label = 'SHAP interaction values' if method == 'shap_interaction' else 'feature correlation'
    console.print(Panel(f"Method: {method_label}",
                        title=f"{result.get('ticker', ticker.upper())} Feature Interactions",
                        border_style="cyan"))

    if not result.get('interactions'):
        console.print("[yellow]No interactions found.[/yellow]")
        return

    t = Table(box=box.SIMPLE)
    t.add_column("#", style="dim", width=3)
    t.add_column("Feature 1", style="bold")
    t.add_column("Feature 2", style="bold")
    t.add_column("Strength", justify="right")

    max_strength = result['interactions'][0]['interaction_strength'] or 1
    for i, pair in enumerate(result['interactions'], 1):
        bar_len = int(pair['interaction_strength'] / max_strength * 15)
        bar = '█' * bar_len + '░' * (15 - bar_len)
        t.add_row(str(i), pair['feature_1'], pair['feature_2'],
                  f"{pair['interaction_strength']:.4f} {bar}")

    console.print(t)


@cli.command('feature-groups')
@click.argument('ticker')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def feature_groups_cmd(ticker, as_json) -> None:
    """Show feature importance by category for a ticker's model.

    Groups features into categories (technical, fundamental, sentiment, etc.)
    and shows which groups contribute most to predictions.

    Examples:
      stk feature-groups TSLA
      stk feature-groups AAPL --json
    """
    from cli.engine import feature_group_importance
    result = feature_group_importance(ticker)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    console.print(Panel(f"{result['total_features']} features across {len(result['groups'])} groups",
                        title=f"{result['ticker']} Feature Groups", border_style="cyan"))

    t = Table(box=box.SIMPLE)
    t.add_column("Group", style="bold")
    t.add_column("Share", justify="right")
    t.add_column("Count", justify="right")
    t.add_column("Top Features")

    for g in result['groups']:
        bar_len = int(g['pct'] / 100 * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        top = ', '.join(f['name'] for f in g['top_features'][:3])
        t.add_row(g['group'], f"{g['pct']:.1f}% {bar}", str(g['count']), top)

    console.print(t)


@cli.command('compare-models')
@click.argument('ticker1')
@click.argument('ticker2')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def compare_models_cmd(ticker1, ticker2, as_json) -> None:
    """Compare two ticker models side by side.

    Shows health, accuracy, features, and ensemble weights for both models.

    Examples:
      stk compare-models TSLA AAPL
      stk compare-models NVDA MSFT --json
    """
    from cli.engine import compare_models
    result = compare_models(ticker1, ticker2)

    if as_json:
        import json
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    m1, m2 = result['model1'], result['model2']

    t = Table(box=box.SIMPLE, title=f"{m1['ticker']} vs {m2['ticker']}")
    t.add_column("Metric", style="bold")
    t.add_column(m1['ticker'], justify="right")
    t.add_column(m2['ticker'], justify="right")
    t.add_column("Better", justify="center")

    def _row(label, v1, v2, fmt=".2f", higher_better=True):
        s1 = f"{v1:{fmt}}" if v1 is not None else "—"
        s2 = f"{v2:{fmt}}" if v2 is not None else "—"
        if v1 is not None and v2 is not None:
            better = m1['ticker'] if (v1 > v2) == higher_better else m2['ticker']
        else:
            better = "—"
        t.add_row(label, s1, s2, better)

    _row("Health Score", m1['health_score'], m2['health_score'])
    t.add_row("Health Grade", m1['health_grade'], m2['health_grade'], "")
    _row("WF Short", m1.get('wf_short'), m2.get('wf_short'), ".1%")
    _row("WF Medium", m1.get('wf_medium'), m2.get('wf_medium'), ".1%")
    _row("WF Long", m1.get('wf_long'), m2.get('wf_long'), ".1%")
    _row("Features", m1.get('features'), m2.get('features'), "d", False)
    _row("Samples", m1.get('samples'), m2.get('samples'), "d")
    t.add_row("Calibrated", "✅" if m1['calibrated'] else "❌",
              "✅" if m2['calibrated'] else "❌", "")

    console.print(t)


@cli.command('retrain-status')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def retrain_status_cmd(as_json) -> None:
    """Show which tickers need retraining and why.

    Analyzes model age, health grade, calibration status, and accuracy
    to recommend which models should be retrained first.

    Examples:
      stk retrain-status
      stk retrain-status --json
    """
    from cli.engine import retrain_recommendations
    recs = retrain_recommendations()

    if as_json:
        import json
        click.echo(json.dumps(recs, indent=2, default=str))
        return

    if not recs:
        console.print("[green]All models are healthy — no retraining needed.[/green]")
        return

    console.print(Panel(f"{len(recs)} models need attention", title="Retrain Recommendations",
                        border_style="cyan"))

    t = Table(box=box.SIMPLE)
    t.add_column("Ticker", style="bold")
    t.add_column("Urgency", justify="center")
    t.add_column("Grade", justify="center")
    t.add_column("Age", justify="right")
    t.add_column("Reasons")

    for r in recs:
        urg = r['urgency']
        urg_icon = '🔴' if urg >= 5 else ('🟡' if urg >= 3 else '⚪')
        age_str = f"{r['age_days']}d" if r['age_days'] is not None else "—"
        t.add_row(r['ticker'], f"{urg_icon} {urg}", r['health_grade'],
                  age_str, ', '.join(r['reasons']))

    console.print(t)


if __name__ == '__main__':
    main()
