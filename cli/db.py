"""SQLite storage for positions and watchlist."""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = Path.home() / '.stk' / 'data.db'


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS positions (
        ticker TEXT PRIMARY KEY, entry_price REAL, qty REAL,
        added_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        ticker TEXT PRIMARY KEY, added_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, direction TEXT, confidence REAL, price_at REAL,
        horizon TEXT, conviction_tier TEXT, top_reason TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Public accessor for the database connection."""
    return _conn()


def add_position(ticker: str, entry: float, qty: float) -> None:
    """Add or replace a portfolio position."""
    c = _conn()
    c.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?)",
              (ticker.upper(), entry, qty, datetime.now().isoformat()))
    c.commit()


def remove_position(ticker: str) -> None:
    """Remove a position and record it in sold_positions for tax reporting."""
    c = _conn()
    # Record in sold_positions for tax reporting
    c.execute("""CREATE TABLE IF NOT EXISTS sold_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, entry_price REAL, sell_price REAL, qty REAL,
        added_at TEXT, sold_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    row = c.execute("SELECT * FROM positions WHERE ticker=?", (ticker.upper(),)).fetchone()
    if row:
        try:
            from cli.engine import get_price
            sell_price = get_price(ticker)['price']
        except Exception:
            sell_price = row['entry_price']
        c.execute("INSERT INTO sold_positions (ticker, entry_price, sell_price, qty, added_at, sold_at) VALUES (?,?,?,?,?,?)",
                  (ticker.upper(), row['entry_price'], sell_price, row['qty'], row['added_at'], datetime.now().isoformat()))
    c.execute("DELETE FROM positions WHERE ticker=?", (ticker.upper(),))
    c.commit()


def get_positions() -> List[Dict]:
    """Return all portfolio positions."""
    c = _conn()
    return [dict(r) for r in c.execute("SELECT * FROM positions").fetchall()]


def add_watch(ticker: str) -> None:
    """Add a ticker to the watchlist."""
    c = _conn()
    c.execute("INSERT OR IGNORE INTO watchlist VALUES (?,?)",
              (ticker.upper(), datetime.now().isoformat()))
    c.commit()


def remove_watch(ticker: str) -> None:
    """Remove a ticker from the watchlist."""
    c = _conn()
    c.execute("DELETE FROM watchlist WHERE ticker=?", (ticker.upper(),))
    c.commit()


def get_watchlist() -> List[Dict]:
    """Return all watchlist entries."""
    c = _conn()
    return [dict(r) for r in c.execute("SELECT * FROM watchlist").fetchall()]


def save_prediction(ticker: str, direction: str, confidence: float, price: float, horizon: str,
                    conviction_tier: str = None, top_reason: str = None) -> None:
    """Save a prediction record for history tracking."""
    c = _conn()
    c.execute("INSERT INTO predictions (ticker, direction, confidence, price_at, horizon, conviction_tier, top_reason, created_at) VALUES (?,?,?,?,?,?,?,?)",
              (ticker.upper(), direction, confidence, price, horizon, conviction_tier, top_reason, datetime.now().isoformat()))
    c.commit()


def get_predictions(ticker: str, limit: int = 20) -> List[Dict]:
    """Return recent predictions for a ticker."""
    c = _conn()
    return [dict(r) for r in c.execute(
        "SELECT * FROM predictions WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
        (ticker.upper(), limit)).fetchall()]


def evaluate_predictions(ticker: str, current_price: float) -> Dict:
    """Evaluate past predictions against actual price movement.

    Only evaluates predictions where enough time has elapsed for the horizon.
    Returns accuracy stats for each horizon.
    """
    preds = get_predictions(ticker, limit=100)
    if not preds:
        return {}
    horizon_days = {'short': 1, 'medium': 5, 'long': 20}
    now = datetime.now()
    results = {}
    for h, days in horizon_days.items():
        h_preds = [p for p in preds if p.get('horizon') == h]
        if not h_preds:
            continue
        correct = 0
        total = 0
        for p in h_preds:
            price_at = p.get('price_at', 0)
            if price_at <= 0:
                continue
            # Only evaluate if enough time has elapsed for this horizon
            try:
                created = datetime.fromisoformat(p['created_at'])
                elapsed = (now - created).days
                if elapsed < days:
                    continue  # Too soon to evaluate this prediction
            except (ValueError, KeyError, TypeError):
                continue
            actual_return = (current_price - price_at) / price_at
            predicted_up = p.get('direction') == 'bullish'
            actual_up = actual_return > 0
            if predicted_up == actual_up:
                correct += 1
            total += 1
        if total > 0:
            results[h] = {'correct': correct, 'total': total,
                          'accuracy': round(correct / total, 4)}
    # Accuracy by conviction tier
    tier_stats = {}
    for p in preds:
        tier = p.get('conviction_tier')
        if not tier:
            continue
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
        actual_return = (current_price - price_at) / price_at
        predicted_up = p.get('direction') == 'bullish'
        actual_up = actual_return > 0
        if tier not in tier_stats:
            tier_stats[tier] = {'correct': 0, 'total': 0}
        tier_stats[tier]['total'] += 1
        if predicted_up == actual_up:
            tier_stats[tier]['correct'] += 1
    for tier, s in tier_stats.items():
        if s['total'] > 0:
            s['accuracy'] = round(s['correct'] / s['total'], 4)
    if tier_stats:
        results['by_conviction'] = tier_stats
    return results


# --- Alerts ---

def _ensure_alerts_table() -> sqlite3.Connection:
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, condition TEXT, threshold REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        triggered INTEGER DEFAULT 0)""")
    c.commit()
    return c


def add_alert(ticker: str, condition: str, threshold: float) -> None:
    """Create a price alert for a ticker."""
    c = _ensure_alerts_table()
    c.execute("INSERT INTO alerts (ticker, condition, threshold, created_at) VALUES (?,?,?,?)",
              (ticker.upper(), condition, threshold, datetime.now().isoformat()))
    c.commit()


def get_alerts(ticker: Optional[str] = None) -> List[Dict]:
    """Return active (non-triggered, non-paused) alerts."""
    c = _ensure_alerts_table()
    # Ensure paused column exists
    try:
        c.execute("SELECT paused FROM alerts LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE alerts ADD COLUMN paused INTEGER DEFAULT 0")
        c.commit()
    if ticker:
        return [dict(r) for r in c.execute(
            "SELECT * FROM alerts WHERE ticker=? AND triggered=0 AND (paused IS NULL OR paused=0) ORDER BY created_at DESC",
            (ticker.upper(),)).fetchall()]
    return [dict(r) for r in c.execute(
        "SELECT * FROM alerts WHERE triggered=0 AND (paused IS NULL OR paused=0) ORDER BY created_at DESC").fetchall()]


def trigger_alert(alert_id: int) -> None:
    """Mark an alert as triggered."""
    c = _ensure_alerts_table()
    c.execute("UPDATE alerts SET triggered=1 WHERE id=?", (alert_id,))
    c.commit()


def get_all_active_alerts(ticker: Optional[str] = None) -> List[Dict]:
    """Get all non-triggered alerts including paused ones."""
    c = _ensure_alerts_table()
    try:
        c.execute("SELECT paused FROM alerts LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE alerts ADD COLUMN paused INTEGER DEFAULT 0")
        c.commit()
    if ticker:
        return [dict(r) for r in c.execute(
            "SELECT * FROM alerts WHERE ticker=? AND triggered=0 ORDER BY created_at DESC",
            (ticker.upper(),)).fetchall()]
    return [dict(r) for r in c.execute(
        "SELECT * FROM alerts WHERE triggered=0 ORDER BY created_at DESC").fetchall()]


def remove_alert(alert_id: int) -> None:
    """Delete an alert by ID."""
    c = _ensure_alerts_table()
    c.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    c.commit()


def get_triggered_alerts(ticker: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """Return previously triggered alerts."""
    c = _ensure_alerts_table()
    if ticker:
        return [dict(r) for r in c.execute(
            "SELECT * FROM alerts WHERE ticker=? AND triggered=1 ORDER BY created_at DESC LIMIT ?",
            (ticker.upper(), limit)).fetchall()]
    return [dict(r) for r in c.execute(
        "SELECT * FROM alerts WHERE triggered=1 ORDER BY created_at DESC LIMIT ?",
        (limit,)).fetchall()]


def clear_alerts(ticker: Optional[str] = None) -> int:
    """Remove all active alerts. If ticker given, only that ticker's alerts."""
    c = _ensure_alerts_table()
    if ticker:
        cur = c.execute("DELETE FROM alerts WHERE ticker=? AND triggered=0", (ticker.upper(),))
    else:
        cur = c.execute("DELETE FROM alerts WHERE triggered=0")
    c.commit()
    return cur.rowcount


# --- Position Snapshots (cumulative return tracking) ---

def _ensure_snapshots_table() -> sqlite3.Connection:
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS position_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, price REAL, pnl_pct REAL,
        snapshot_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.commit()
    return c


def save_snapshot(ticker: str, price: float, pnl_pct: float) -> None:
    """Save a position snapshot for cumulative return tracking."""
    c = _ensure_snapshots_table()
    c.execute("INSERT INTO position_snapshots (ticker, price, pnl_pct, snapshot_at) VALUES (?,?,?,?)",
              (ticker.upper(), price, pnl_pct, datetime.now().isoformat()))
    c.commit()


def get_snapshots(ticker: str, limit: int = 30) -> List[Dict]:
    """Return recent position snapshots for a ticker."""
    c = _ensure_snapshots_table()
    return [dict(r) for r in c.execute(
        "SELECT * FROM position_snapshots WHERE ticker=? ORDER BY snapshot_at DESC LIMIT ?",
        (ticker.upper(), limit)).fetchall()]


# --- Custom Scan Presets ---

def _ensure_custom_presets_table() -> sqlite3.Connection:
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS custom_presets (
        name TEXT PRIMARY KEY, filters TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.commit()
    return c


def save_custom_preset(name: str, filters: str) -> None:
    """Save or update a custom scan preset."""
    c = _ensure_custom_presets_table()
    c.execute("INSERT OR REPLACE INTO custom_presets VALUES (?,?,?)",
              (name.lower(), filters, datetime.now().isoformat()))
    c.commit()


def get_custom_presets() -> List[Dict]:
    """Return all saved custom scan presets."""
    c = _ensure_custom_presets_table()
    return [dict(r) for r in c.execute("SELECT * FROM custom_presets ORDER BY name").fetchall()]


def get_custom_preset(name: str) -> Optional[Dict]:
    """Return a single custom preset by name, or None."""
    c = _ensure_custom_presets_table()
    row = c.execute("SELECT * FROM custom_presets WHERE name=?", (name.lower(),)).fetchone()
    return dict(row) if row else None


def delete_custom_preset(name: str) -> bool:
    """Delete a custom preset. Returns True if it existed."""
    c = _ensure_custom_presets_table()
    cur = c.execute("DELETE FROM custom_presets WHERE name=?", (name.lower(),))
    c.commit()
    return cur.rowcount > 0


# --- Alert pause/resume ---

def _ensure_alerts_paused_column() -> sqlite3.Connection:
    c = _ensure_alerts_table()
    try:
        c.execute("SELECT paused FROM alerts LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE alerts ADD COLUMN paused INTEGER DEFAULT 0")
        c.commit()
    return c


def pause_alert(alert_id: int) -> bool:
    """Pause an active alert. Returns True if updated."""
    c = _ensure_alerts_paused_column()
    cur = c.execute("UPDATE alerts SET paused=1 WHERE id=? AND triggered=0", (alert_id,))
    c.commit()
    return cur.rowcount > 0


def resume_alert(alert_id: int) -> bool:
    """Resume a paused alert. Returns True if updated."""
    c = _ensure_alerts_paused_column()
    cur = c.execute("UPDATE alerts SET paused=0 WHERE id=? AND triggered=0", (alert_id,))
    c.commit()
    return cur.rowcount > 0


def pause_alerts_by_ticker(ticker: str) -> int:
    """Pause all active alerts for a ticker. Returns count updated."""
    c = _ensure_alerts_paused_column()
    cur = c.execute("UPDATE alerts SET paused=1 WHERE ticker=? AND triggered=0", (ticker.upper(),))
    c.commit()
    return cur.rowcount


def resume_alerts_by_ticker(ticker: str) -> int:
    """Resume all paused alerts for a ticker. Returns count updated."""
    c = _ensure_alerts_paused_column()
    cur = c.execute("UPDATE alerts SET paused=0 WHERE ticker=? AND triggered=0", (ticker.upper(),))
    c.commit()
    return cur.rowcount


# --- Portfolio snapshots (aggregate) ---

def _ensure_portfolio_snapshots_table() -> sqlite3.Connection:
    c = _conn()
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        total_value REAL, total_cost REAL, pnl REAL, pnl_pct REAL,
        num_positions INTEGER,
        snapshot_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.commit()
    return c


def save_portfolio_snapshot(total_value: float, total_cost: float, pnl: float, pnl_pct: float, num_positions: int) -> None:
    """Save an aggregate portfolio snapshot."""
    c = _ensure_portfolio_snapshots_table()
    c.execute("INSERT INTO portfolio_snapshots (total_value, total_cost, pnl, pnl_pct, num_positions, snapshot_at) VALUES (?,?,?,?,?,?)",
              (total_value, total_cost, pnl, pnl_pct, num_positions, datetime.now().isoformat()))
    c.commit()


def get_portfolio_snapshots(limit: int = 90) -> List[Dict]:
    """Return recent aggregate portfolio snapshots."""
    c = _ensure_portfolio_snapshots_table()
    return [dict(r) for r in c.execute(
        "SELECT * FROM portfolio_snapshots ORDER BY snapshot_at DESC LIMIT ?",
        (limit,)).fetchall()]
