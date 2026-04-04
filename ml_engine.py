"""
ml_engine.py v2.0 — Апгрейди:
  ✅ Ensemble: XGBoost + RandomForest + LogisticRegression
  ✅ Розширені features (15 ознак замість 10)
  ✅ Feature Engineering (return candles, vol_std, regime, time)
  ✅ Правильний лейбл: profit > fee*2 (не просто win/loss)
  ✅ Sharpe Ratio, Max Drawdown, Expectancy метрики
  ✅ Збереження ensemble в БД
"""

import logging, pickle, math
from typing import Optional, Tuple, Dict, Any, List

log = logging.getLogger("NeuralTrade.ML")

# ── Розширений список features ──────────────────────────────────
FEATURES = [
    # Технічні
    "rsi",          # RSI(14)
    "macd",         # MACD значення
    "bb_pct",       # Bollinger Band %
    "atr_pct",      # ATR / price  (волатильність в %)
    "volume_ratio", # Volume / MA20 Volume
    "ma_dist",      # (price - MA20) / MA20
    "ma50_dist",    # (price - MA50) / MA50
    "vol_std",      # Realized volatility (rolling std)
    # Цінові рухи
    "ret1",         # Доходність останньої свічки
    "ret3",         # Доходність за 3 свічки
    "ret5",         # Доходність за 5 свічок
    # Сентимент
    "fear_greed",   # Fear & Greed Index
    "funding_rate", # Binance Funding Rate
    "sent_score",   # Зведений сентимент
    # Час
    "hour_of_day",  # Час доби (0-23)
]

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    HAS_ML = True
except ImportError:
    HAS_ML = False
    log.warning("ML: pip install xgboost scikit-learn pandas numpy")


class MLEngine:
    """
    Ensemble ML:
      1. XGBoost       — нелінійні патерни
      2. RandomForest  — стабільність
      3. LogisticReg   — лінійні тренди

    final_prob = mean(xgb, rf, lr)
    """

    def __init__(self, db, cfg):
        self.db  = db
        self.cfg = cfg
        # pair → {"xgb": ..., "rf": ..., "lr": ..., "scaler": ..., "accuracy": ...}
        self._models: Dict[str, Dict] = {}

    def load(self, pair: str, blob: bytes):
        if not HAS_ML:
            return
        try:
            data = pickle.loads(blob)
            self._models[pair] = data
            log.info(f"🧠 Ensemble {pair}: acc={data.get('accuracy',0):.1%} "
                     f"n={data.get('n_samples',0)}")
        except Exception as e:
            log.error(f"ML load [{pair}]: {e}")

    def predict(self, pair: str, features: Dict[str, Any]) -> float:
        if not HAS_ML or pair not in self._models:
            return self._heuristic(features)

        m = self._models[pair]
        scaler = m.get("scaler")
        if not scaler:
            return self._heuristic(features)

        try:
            row = self._build_row(features)
            X   = scaler.transform([row])
            probs = []
            for key in ("xgb", "rf", "lr"):
                model = m.get(key)
                if model:
                    try:
                        p = float(model.predict_proba(X)[0][1])
                        probs.append(p)
                    except Exception:
                        pass
            return float(np.mean(probs)) if probs else self._heuristic(features)
        except Exception as e:
            log.error(f"ML predict [{pair}]: {e}")
            return self._heuristic(features)

    def train(self, pair: str) -> Optional[Tuple[float, int]]:
        if not HAS_ML:
            return None

        trades = self.db.get_closed_with_features(pair)
        min_n  = self.cfg.min_trades_for_train or 30
        if len(trades) < min_n:
            log.info(f"🧠 {pair}: {len(trades)}/{min_n} — пропуск")
            return None

        log.info(f"🧠 Навчання ensemble [{pair}] ({len(trades)} угод)...")

        rows, labels = [], []
        fee_rate = self.cfg.effective_fee or 0.001

        for t in trades:
            row = self._build_row(t)
            # ── Покращений лейбл: profit > fee*2 ─────────────────
            net = t.get("net_pnl")
            amount = t.get("amount_usd", 1)
            if net is None:
                continue
            # Win = net PnL перевищує 2× комісії (реальний прибуток)
            label = 1 if net > amount * fee_rate * 2 else 0

            if any(v is None for v in row):
                continue
            rows.append(row)
            labels.append(label)

        if len(rows) < min_n:
            log.warning(f"🧠 {pair}: тільки {len(rows)} повних записів")
            return None

        X = np.array(rows, dtype=float)
        y = np.array(labels)

        # Стратифікований split
        if len(X) >= 50:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_tr, X_te, y_tr, y_te = X, X, y, y

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # ── Навчання трьох моделей ──────────────────────────────
        models = {}

        # XGBoost
        try:
            xgb = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )
            xgb.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
            models["xgb"] = xgb
        except Exception as e:
            log.warning(f"XGBoost: {e}")

        # RandomForest
        try:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=2,
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr_s, y_tr)
            models["rf"] = rf
        except Exception as e:
            log.warning(f"RandomForest: {e}")

        # LogisticRegression
        try:
            lr = LogisticRegression(
                C=0.5, max_iter=500, random_state=42,
                class_weight="balanced",
            )
            lr.fit(X_tr_s, y_tr)
            models["lr"] = lr
        except Exception as e:
            log.warning(f"LogisticReg: {e}")

        if not models:
            return None

        # ── Ensemble predict ────────────────────────────────────
        probs_te = []
        for m in models.values():
            try:
                probs_te.append(m.predict_proba(X_te_s)[:, 1])
            except Exception:
                pass
        if not probs_te:
            return None

        ens_probs = np.mean(probs_te, axis=0)
        ens_preds = (ens_probs > 0.5).astype(int)

        acc = float(accuracy_score(y_te, ens_preds))
        f1  = float(f1_score(y_te, ens_preds, zero_division=0))
        pr  = float(precision_score(y_te, ens_preds, zero_division=0))
        rc  = float(recall_score(y_te, ens_preds, zero_division=0))

        log.info(f"🧠 Ensemble [{pair}]: acc={acc:.1%} f1={f1:.2f} "
                 f"prec={pr:.2f} rec={rc:.2f} n={len(rows)}")

        # Feature importance (XGBoost)
        if "xgb" in models:
            try:
                imp = dict(zip(FEATURES, models["xgb"].feature_importances_))
                top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
                log.info("🧠 Top features: " + " | ".join(f"{k}={v:.3f}" for k,v in top5))
            except Exception:
                pass

        # ── Метрики ─────────────────────────────────────────────
        metrics = self._compute_metrics(trades)
        log.info(f"📊 [{pair}] Sharpe={metrics['sharpe']:.2f} "
                 f"MaxDD={metrics['max_drawdown']:.1%} "
                 f"Expect=${metrics['expectancy']:.2f}")

        # ── Зберегти ────────────────────────────────────────────
        ensemble_data = {
            "scaler": scaler,
            "accuracy": acc, "f1": f1,
            "n_samples": len(rows),
            "features": FEATURES,
            "metrics": metrics,
            **models,
        }
        blob = pickle.dumps(ensemble_data)
        self.db.save_model(pair, acc, len(rows), blob)
        self._models[pair] = ensemble_data

        return acc, len(rows)

    def train_all(self):
        results = {}
        for pair in self.cfg.pairs:
            r = self.train(pair)
            if r:
                results[pair] = r
        return results

    def get_metrics(self, pair: str) -> Dict[str, Any]:
        """Повернути метрики для дашборду."""
        if pair in self._models:
            return self._models[pair].get("metrics", {})
        trades = self.db.get_closed_with_features(pair)
        return self._compute_metrics(trades) if trades else {}

    def _compute_metrics(self, trades: List[dict]) -> Dict[str, Any]:
        """Sharpe Ratio, Max Drawdown, Expectancy, Winrate."""
        closed = [t for t in trades if t.get("net_pnl") is not None]
        if not closed:
            return {"sharpe": 0, "max_drawdown": 0, "expectancy": 0, "winrate": 0}

        pnls = [float(t["net_pnl"]) for t in closed]
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p <= 0]

        winrate = len(wins) / len(pnls) if pnls else 0

        # Sharpe Ratio (спрощений — без безризикової ставки)
        import statistics as stats
        if len(pnls) >= 2:
            mean_r = stats.mean(pnls)
            std_r  = stats.stdev(pnls)
            sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe = 0

        # Max Drawdown
        equity = [0.0]
        for p in pnls:
            equity.append(equity[-1] + p)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Expectancy: avg_win * winrate - avg_loss * (1 - winrate)
        avg_win  = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(loss) / len(loss)) if loss else 0
        expect   = avg_win * winrate - avg_loss * (1 - winrate)

        return {
            "sharpe":       round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "expectancy":   round(expect, 2),
            "winrate":      round(winrate, 4),
            "total_trades": len(closed),
            "avg_win":      round(avg_win, 2),
            "avg_loss":     round(avg_loss, 2),
        }

    def _build_row(self, t: dict) -> list:
        price = t.get("entry_price") or t.get("price") or 1
        atr   = t.get("atr") or 0
        return [
            t.get("rsi")          or 50,
            t.get("macd")         or 0,
            t.get("bb_pct")       or 0.5,
            (atr / price * 100)   if price > 0 else 0,   # atr_pct
            t.get("volume_ratio") or 1,
            t.get("ma_dist")      or 0,
            t.get("ma50_dist")    or (t.get("ma50_dist") or 0),
            t.get("vol_std")      or 1,
            t.get("ret1")         or 0,
            t.get("ret3")         or 0,
            t.get("ret5")         or 0,
            t.get("fear_greed")   or 50,
            t.get("funding_rate") or 0,
            t.get("sent_score")   or 0,
            t.get("hour_of_day")  or 12,
        ]

    @staticmethod
    def _heuristic(f: Dict[str, Any]) -> float:
        score = 0.5
        rsi = f.get("rsi", 50)
        if rsi < 35:   score += 0.12
        elif rsi > 65: score -= 0.12
        if f.get("macd", 0) > 0: score += 0.08
        else:                    score -= 0.08
        price = f.get("price", 1)
        ma20  = f.get("ma20", price)
        if ma20 and price > ma20: score += 0.06
        elif ma20:                score -= 0.06
        vol   = f.get("volume_ratio", 1)
        if vol > 1.5: score += 0.04
        sent  = f.get("sent_score", 0)
        score += sent * 0.08
        ret3  = f.get("ret3", 0)
        if ret3 > 0.01: score += 0.05
        elif ret3 < -0.01: score -= 0.05
        return max(0.1, min(0.9, score))
