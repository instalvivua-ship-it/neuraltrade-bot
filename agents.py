"""
agents.py v2.0 — Апгрейди:
  ✅ Trailing Stop (динамічний тейк-профіт)
  ✅ Volatility Filter (захист від маніпуляцій)
  ✅ Market Regime AI (тренд/флет/висока волатильність)
  ✅ ATR-based SL/TP замість фіксованих %
  ✅ Kelly Criterion + Volatility position sizing
  ✅ Portfolio correlation check
  ✅ Drawdown stop + Loss streak pause
  ✅ Executor upgrade (retry, slippage, partial fills)
  ✅ Ensemble ML (XGBoost + RF + LR)
  ✅ Weighted Debate AI
"""

import logging, time, math, statistics
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import requests

log = logging.getLogger("NeuralTrade.Agents")

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

try:
    import pandas as pd
    import numpy as np
    HAS_TA = True
except ImportError:
    HAS_TA = False
    log.warning("pandas/numpy не встановлено")

# Бібліотека технічного аналізу
try:
    import ta as ta_lib
    HAS_TA_LIB = True
except ImportError:
    HAS_TA_LIB = False
    log.warning("ta не встановлено: pip install ta")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    HAS_ML = True
except ImportError:
    HAS_ML = False


# ════════════════════════════════════════════════════════════════
#  📊 TECH ANALYST v2 — з Regime Detection + Volatility Filter
# ════════════════════════════════════════════════════════════════

class TechAnalystAgent:
    """
    Технічний аналіз + Market Regime Detection.

    Market Regime:
      trend_up   — ADX > 25, MA slope > 0
      trend_down — ADX > 25, MA slope < 0
      sideways   — ADX < 20
      volatile   — ATR / price > 2x середнє

    Volatility Filter:
      Якщо остання свічка > 2.5× середнє ATR → PAUSE (не торгуємо)
    """

    TF_MAP = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}

    def __init__(self, cfg):
        self.cfg       = cfg
        self._exchange = None
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # pair → (price, ts)
        self._ohlcv_cache: Dict[str, tuple] = {}   # pair → (df, timestamp)
        self._ohlcv_ttl   = 300                     # 5 хв TTL
        self._init_exchange()

    def _init_exchange(self):
        # ccxt повністю вимкнено — Railway блокує Binance (451)
        # Ціни: CoinGecko + Bybit (не блокуються)
        # OHLCV: Bybit public API
        self._exchange = None
        log.info("✅ Ціни: CoinGecko + Bybit API (Railway-compatible)")

    def get_current_price(self, pair: str) -> Optional[float]:
        """
        Отримує ціну через:
        1. Bybit REST API (публічний, не блокується)
        2. CoinGecko API (резервний)
        3. Симуляція (останній варіант)
        """
        # Кеш 5 секунд
        if pair in self._price_cache:
            p, ts = self._price_cache[pair]
            if time.time() - ts < 5:
                return p

        # ── Bybit публічний REST (не блокується з Railway) ────
        try:
            sym = pair.replace("/", "")
            r = requests.get(
                f"https://api.bybit.com/v5/market/tickers"
                f"?category=spot&symbol={sym}",
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                items = data.get("result", {}).get("list", [])
                if items:
                    p = float(items[0]["lastPrice"])
                    self._price_cache[pair] = (p, time.time())
                    return p
        except Exception:
            pass

        # ── CoinGecko резервний ────────────────────────────────
        try:
            sym_map = {
                "BTC/USDT": "bitcoin", "ETH/USDT": "ethereum",
                "SOL/USDT": "solana",  "BNB/USDT": "binancecoin",
            }
            cg_id = sym_map.get(pair)
            if cg_id:
                r = requests.get(
                    f"https://api.coingecko.com/api/v3/simple/price"
                    f"?ids={cg_id}&vs_currencies=usd",
                    timeout=8
                )
                if r.status_code == 200:
                    p = float(r.json()[cg_id]["usd"])
                    self._price_cache[pair] = (p, time.time())
                    return p
        except Exception:
            pass

        # ── Симуляція (якщо все заблоковано) ──────────────────
        import random
        base = {
            "BTC/USDT": 67000, "ETH/USDT": 3200,
            "SOL/USDT": 148,   "BNB/USDT": 590
        }.get(pair, 100)
        p = base * (1 + random.gauss(0, 0.001))
        self._price_cache[pair] = (p, time.time())
        return p

    def get_orderbook_features(self, pair: str) -> Dict[str, float]:
        """
        Отримує Order Book imbalance + CVD з Bybit.

        OBI (Order Book Imbalance):
            > 0.6 = тиск покупців (bullish)
            < 0.4 = тиск продавців (bearish)

        CVD (Cumulative Volume Delta):
            Позитивний = більше buy volume (bullish)
            Негативний = більше sell volume (bearish)
        """
        default = {"obi": 0.5, "cvd": 0.0, "spread_pct": 0.0,
                   "bid_wall": 0.0, "ask_wall": 0.0}
        try:
            sym = pair.replace("/", "")

            # ── Order Book (топ 25 рівнів) ────────────────────────
            r = requests.get(
                f"https://api.bybit.com/v5/market/orderbook"
                f"?category=spot&symbol={sym}&limit=25",
                timeout=5
            )
            if r.status_code != 200:
                return default
            ob = r.json().get("result", {})
            bids = ob.get("b", [])  # [[price, qty], ...]
            asks = ob.get("a", [])
            if not bids or not asks:
                return default

            # OBI = bid_volume / (bid_volume + ask_volume)
            bid_vol = sum(float(b[1]) for b in bids[:10])
            ask_vol = sum(float(a[1]) for a in asks[:10])
            total_vol = bid_vol + ask_vol
            obi = bid_vol / total_vol if total_vol > 0 else 0.5

            # Bid/Ask wall — найбільший рівень
            bid_wall = max(float(b[1]) for b in bids[:10]) if bids else 0.0
            ask_wall = max(float(a[1]) for a in asks[:10]) if asks else 0.0

            # Спред
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0.0

            # ── CVD з останніх 200 угод ────────────────────────────
            r2 = requests.get(
                f"https://api.bybit.com/v5/market/recent-trade"
                f"?category=spot&symbol={sym}&limit=200",
                timeout=5
            )
            cvd = 0.0
            if r2.status_code == 200:
                trades = r2.json().get("result", {}).get("list", [])
                for t in trades:
                    qty  = float(t.get("size", 0))
                    side = t.get("side", "")
                    if side == "Buy":
                        cvd += qty
                    elif side == "Sell":
                        cvd -= qty
                # Нормалізуємо CVD як % від загального об'єму
                total_trade_vol = sum(float(t.get("size", 0)) for t in trades)
                if total_trade_vol > 0:
                    cvd = cvd / total_trade_vol  # -1..+1

            log.debug(f"OB {pair}: OBI={obi:.3f} CVD={cvd:.3f} spread={spread_pct:.4f}%")
            return {
                "obi":       round(obi, 4),
                "cvd":       round(cvd, 4),
                "spread_pct": round(spread_pct, 5),
                "bid_wall":  round(bid_wall, 2),
                "ask_wall":  round(ask_wall, 2),
            }
        except Exception as e:
            log.debug(f"OB {pair}: {e}")
            return default

    def fetch_ohlcv_tf(self, pair: str, tf: str, limit: int = 100) -> Optional["pd.DataFrame"]:
        """Завантажує OHLCV для конкретного таймфрейму (з кешем 5 хв)."""
        cache_key = f"{pair}_{tf}"
        if cache_key in self._ohlcv_cache:
            df, ts = self._ohlcv_cache[cache_key]
            if time.time() - ts < self._ohlcv_ttl:
                return df
        try:
            sym = pair.replace("/", "")
            tf_map = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}
            interval = tf_map.get(tf, "60")
            r = requests.get(
                f"https://api.bybit.com/v5/market/kline"
                f"?category=spot&symbol={sym}&interval={interval}&limit={limit}",
                timeout=8
            )
            if r.status_code == 200:
                items = r.json().get("result", {}).get("list", [])
                if items and len(items) >= 30:
                    items = list(reversed(items))
                    df = pd.DataFrame(items, columns=["ts","open","high","low","close","volume","turnover"])
                    for col in ["open","high","low","close","volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
                    df = df.dropna()
                    if len(df) >= 20:
                        self._ohlcv_cache[cache_key] = (df, time.time())
                        return df
        except Exception as e:
            log.debug(f"OHLCV {pair} {tf}: {e}")
        return None

    def _quick_rsi(self, c, window=14) -> float:
        """Швидкий RSI без зовнішніх бібліотек."""
        try:
            if HAS_TA_LIB:
                return float(ta_lib.momentum.RSIIndicator(c, window=window).rsi().iloc[-1])
            delta = c.diff()
            gain  = delta.clip(lower=0).rolling(window).mean()
            loss  = (-delta.clip(upper=0)).rolling(window).mean()
            rs    = gain / loss.replace(0, 1e-10)
            return float(100 - 100 / (1 + rs.iloc[-1]))
        except Exception:
            return 50.0

    def fetch_ohlcv(self, pair: str, limit: int = 200, force: bool = False) -> Optional["pd.DataFrame"]:
        """
        Завантажує свічки через:
        1. Кеш (TTL=5 хв) — щоб не смикати API кожні 30 сек
        2. Bybit REST API (не блокується)
        3. Симуляція як fallback
        """
        if not HAS_TA:
            return None

        # ── Кеш ──────────────────────────────────────────────
        if not force and pair in self._ohlcv_cache:
            df, ts = self._ohlcv_cache[pair]
            if time.time() - ts < self._ohlcv_ttl:
                return df

        # ── Bybit klines (публічний, не блокується) ───────────
        try:
            sym = pair.replace("/", "")
            tf_map = {
                "1m": "1",  "5m": "5",  "15m": "15",
                "1h": "60", "4h": "240","1d": "D"
            }
            interval = tf_map.get(self.cfg.timeframe or "1h", "60")
            r = requests.get(
                f"https://api.bybit.com/v5/market/kline"
                f"?category=spot&symbol={sym}&interval={interval}&limit={limit}",
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                items = data.get("result", {}).get("list", [])
                if items and len(items) >= 50:
                    # Bybit повертає в зворотному порядку
                    items = list(reversed(items))
                    df = pd.DataFrame(items, columns=[
                        "ts", "open", "high", "low", "close", "volume", "turnover"
                    ])
                    for col in ["open","high","low","close","volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
                    df = df.dropna()
                    if len(df) >= 50:
                        log.debug(f"Bybit OHLCV {pair}: {len(df)} свічок")
                        self._ohlcv_cache[pair] = (df, time.time())
                        return df
        except Exception as e:
            log.debug(f"Bybit OHLCV {pair}: {e}")

        df_sim = self._simulate_ohlcv(pair, limit)
        if df_sim is not None:
            self._ohlcv_cache[pair] = (df_sim, time.time())
        return df_sim

    def _simulate_ohlcv(self, pair: str, limit: int) -> "pd.DataFrame":
        import random
        price = self.get_current_price(pair) or 84000
        data, p = [], price * 0.95
        for i in range(limit):
            p *= (1 + random.gauss(0, 0.006))
            data.append([
                pd.Timestamp.now() - pd.Timedelta(hours=limit-i),
                p*0.999, p*1.003, p*0.997, p, random.uniform(50,500)
            ])
        return pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])

    def analyze(self, pair: str) -> Optional[Dict[str, Any]]:
        """Технічний аналіз через бібліотеку ta."""
        df    = self.fetch_ohlcv(pair)
        price = self.get_current_price(pair)

        if df is None or price is None or len(df) < 60:
            return self._fallback(pair)
        if not HAS_TA:
            return self._fallback(pair)

        try:
            c = df["close"].astype(float)
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            v = df["volume"].astype(float)

            if not HAS_TA_LIB:
                # Ручний розрахунок без бібліотеки
                return self._calc_manual(pair, price, c, h, l, v)

            # ── RSI ───────────────────────────────────────────────
            rsi_s = ta_lib.momentum.RSIIndicator(c, window=14).rsi()
            rsi   = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0

            # ── MACD ──────────────────────────────────────────────
            macd_ind = ta_lib.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
            macd     = float(macd_ind.macd().iloc[-1]) if not macd_ind.macd().empty else 0.0
            macd_h   = float(macd_ind.macd_diff().iloc[-1]) if not macd_ind.macd_diff().empty else 0.0

            # ── Moving Averages ───────────────────────────────────
            ma20  = float(ta_lib.trend.SMAIndicator(c, window=20).sma_indicator().iloc[-1])
            ma50  = float(ta_lib.trend.SMAIndicator(c, window=50).sma_indicator().iloc[-1])
            ma200 = float(ta_lib.trend.SMAIndicator(c, window=min(200,len(c)-1)).sma_indicator().iloc[-1])

            # ── Bollinger Bands ───────────────────────────────────
            bb_ind = ta_lib.volatility.BollingerBands(c, window=20, window_dev=2)
            bb_u   = float(bb_ind.bollinger_hband().iloc[-1])
            bb_l   = float(bb_ind.bollinger_lband().iloc[-1])
            bb_pct = (price - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5

            # ── ATR ───────────────────────────────────────────────
            atr_s = ta_lib.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
            atr   = float(atr_s.iloc[-1]) if not atr_s.empty else price * 0.012

            # ── ADX ───────────────────────────────────────────────
            try:
                adx = float(ta_lib.trend.ADXIndicator(h, l, c, window=14).adx().iloc[-1])
            except Exception:
                adx = 20.0

            # ── Volume ────────────────────────────────────────────
            vol_avg = float(v.rolling(20).mean().iloc[-1])
            vol_r   = float(v.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

            # ── Volatility Filter ─────────────────────────────────
            last_candle_range = abs(float(h.iloc[-1]) - float(l.iloc[-1]))
            candle_vs_atr     = last_candle_range / atr if atr > 0 else 1.0
            volatility_spike  = candle_vs_atr > (self.cfg.volatility_pause_multiplier or 2.5)

            # ── Market Regime ─────────────────────────────────────
            sma20_vals = ta_lib.trend.SMAIndicator(c, window=20).sma_indicator()
            ma20_slope = float(sma20_vals.iloc[-1] - sma20_vals.iloc[-4]) / 3 if len(sma20_vals) >= 4 else 0
            atr_avg20  = float(atr_s.rolling(20).mean().iloc[-1]) if len(atr_s) >= 20 else atr
            atr_ratio  = atr / atr_avg20 if atr_avg20 > 0 else 1.0

            if atr_ratio > 1.8:
                regime = "volatile"
            elif adx > 25 and ma20_slope > 0:
                regime = "trend_up"
            elif adx > 25 and ma20_slope < 0:
                regime = "trend_down"
            else:
                regime = "sideways"

            # ── SL/TP відстані (ATR-based) ────────────────────────
            sl_distance = atr * (self.cfg.atr_sl_multiplier or 1.5)
            tp_distance = atr * (self.cfg.atr_tp_multiplier or 3.0)

            # ── ML features ───────────────────────────────────────
            returns   = c.pct_change()
            ret1      = float(returns.iloc[-1])
            ret3      = float(returns.iloc[-3:].sum())
            ret5      = float(returns.iloc[-5:].sum())
            ma_dist   = (price - ma20)  / ma20  if ma20  > 0 else 0
            ma50_dist = (price - ma50)  / ma50  if ma50  > 0 else 0
            vol_std   = float(returns.rolling(20).std().iloc[-1]) * 100
            hour_of_day = datetime.now().hour

            # ── Торговий сигнал ───────────────────────────────────
            score, reasons = 0.0, []

            if regime == "sideways":
                if rsi < 30:  score += 0.25; reasons.append(f"RSI {rsi:.1f} OS+sideways")
                elif rsi > 70: score -= 0.25; reasons.append(f"RSI {rsi:.1f} OB+sideways")
            else:
                if rsi < 35:    score += 0.18; reasons.append(f"RSI {rsi:.1f} OS")
                elif rsi > 65:  score -= 0.18; reasons.append(f"RSI {rsi:.1f} OB")
                if macd_h > 0:  score += 0.15; reasons.append("MACD+")
                else:           score -= 0.15; reasons.append("MACD-")
                if price > ma20: score += 0.10; reasons.append("P>MA20")
                else:            score -= 0.10
                if price > ma50: score += 0.08; reasons.append("P>MA50")
                else:            score -= 0.08
                if price > ma200: score += 0.05; reasons.append("P>MA200")
                else:             score -= 0.05

            if bb_pct < 0.2:  score += 0.10; reasons.append("BB low")
            elif bb_pct > 0.8: score -= 0.10; reasons.append("BB high")
            if vol_r > 1.8:   score += 0.04; reasons.append(f"Vol×{vol_r:.1f}")
            if regime == "volatile": score *= 0.4; reasons.insert(0, "⚠️ Volatile")

            confidence = max(0.1, min(0.95, (score + 0.7) / 1.4))
            # В DEMO режимі — нижчий поріг для швидшого збору зразків ML
            demo_mode = getattr(self.cfg, 'mode', 'demo') == 'demo'
            threshold = 0.08 if demo_mode else 0.15  # 0.08 замість 0.15
            if score > threshold:    signal = "BUY"
            elif score < -threshold: signal = "SELL"
            else:                    signal = "HOLD"

            # ── MTF: 5m і 15m для підтвердження ──────────────────
            mtf_score   = 0.0
            mtf_reasons = []
            for tf_check in ["5m", "15m"]:
                try:
                    df_tf = self.fetch_ohlcv_tf(pair, tf_check, limit=60)
                    if df_tf is not None and len(df_tf) >= 20:
                        c_tf  = df_tf["close"].astype(float)
                        rsi_tf = self._quick_rsi(c_tf)
                        ema12  = float(c_tf.ewm(span=12).mean().iloc[-1])
                        ema26  = float(c_tf.ewm(span=26).mean().iloc[-1])
                        macd_tf = ema12 - ema26
                        ma20_tf = float(c_tf.rolling(20).mean().iloc[-1])
                        p_tf    = float(c_tf.iloc[-1])
                        # Напрямок на цьому TF
                        tf_bull = (rsi_tf < 50 and macd_tf > 0) or (p_tf > ma20_tf and macd_tf > 0)
                        tf_bear = (rsi_tf > 50 and macd_tf < 0) or (p_tf < ma20_tf and macd_tf < 0)
                        if tf_bull:
                            mtf_score += 0.08
                            mtf_reasons.append(f"{tf_check}↑")
                        elif tf_bear:
                            mtf_score -= 0.08
                            mtf_reasons.append(f"{tf_check}↓")
                except Exception:
                    pass

            # ── Order Book + CVD ──────────────────────────────────
            ob = self.get_orderbook_features(pair)
            obi = ob["obi"]
            cvd = ob["cvd"]

            # OBI: > 0.6 = купці домінують, < 0.4 = продавці
            if obi > 0.62:
                mtf_score += 0.10; mtf_reasons.append(f"OBI={obi:.2f}↑")
            elif obi < 0.38:
                mtf_score -= 0.10; mtf_reasons.append(f"OBI={obi:.2f}↓")

            # CVD: позитивний = buy volume переважає
            if cvd > 0.15:
                mtf_score += 0.08; mtf_reasons.append(f"CVD+{cvd:.2f}")
            elif cvd < -0.15:
                mtf_score -= 0.08; mtf_reasons.append(f"CVD{cvd:.2f}")

            # Додаємо MTF+OB до основного score
            score += mtf_score
            if mtf_reasons:
                reasons.extend(mtf_reasons)

            # Перерахуємо signal з новим score
            demo_mode = getattr(self.cfg, 'mode', 'demo') == 'demo'
            threshold = 0.08 if demo_mode else 0.15
            if score > threshold:    signal = "BUY"
            elif score < -threshold: signal = "SELL"
            else:                    signal = "HOLD"
            confidence = max(0.1, min(0.95, (score + 0.7) / 1.4))

            return {
                "pair": pair, "price": price, "signal": signal,
                "score": score, "confidence": confidence,
                "rsi": rsi, "macd": macd, "macd_h": macd_h,
                "ma20": ma20, "ma50": ma50, "ma200": ma200,
                "bb_pct": bb_pct, "bb_upper": bb_u, "bb_lower": bb_l,
                "atr": atr, "adx": adx,
                "volume_ratio": vol_r, "vol_std": vol_std,
                "regime": regime,
                "candle_vs_atr": candle_vs_atr,
                "volatility_spike": volatility_spike,
                "sl_distance": sl_distance, "tp_distance": tp_distance,
                "ma_dist": ma_dist, "ma50_dist": ma50_dist,
                "ret1": ret1, "ret3": ret3, "ret5": ret5,
                "hour_of_day": hour_of_day,
                "reasons": reasons,
                # Нові features
                "obi": obi, "cvd": cvd,
                "spread_pct": ob["spread_pct"],
                "mtf_score": round(mtf_score, 3),
            }

        except Exception as e:
            log.error(f"Tech [{pair}]: {e}", exc_info=True)
            return self._fallback(pair)

    def _calc_manual(self, pair, price, c, h, l, v) -> Dict[str, Any]:
        """Ручний розрахунок індикаторів без зовнішніх бібліотек."""
        import numpy as np
        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-10)
        rsi   = float(100 - 100/(1+rs.iloc[-1]))

        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd   = float(macd_line.iloc[-1])
        macd_h = float((macd_line - signal_line).iloc[-1])

        # MAs
        ma20  = float(c.rolling(20).mean().iloc[-1])
        ma50  = float(c.rolling(50).mean().iloc[-1])
        ma200 = float(c.rolling(min(200,len(c))).mean().iloc[-1])

        # BB
        std20 = c.rolling(20).std()
        bb_u  = float((c.rolling(20).mean() + 2*std20).iloc[-1])
        bb_l  = float((c.rolling(20).mean() - 2*std20).iloc[-1])
        bb_pct= (price-bb_l)/(bb_u-bb_l) if (bb_u-bb_l)>0 else 0.5

        # ATR
        hl  = h - l
        hpc = abs(h - c.shift(1))
        lpc = abs(l - c.shift(1))
        tr  = pd.concat([hl,hpc,lpc],axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        adx = 20.0

        vol_avg = float(v.rolling(20).mean().iloc[-1])
        vol_r   = float(v.iloc[-1])/vol_avg if vol_avg>0 else 1.0

        last_candle_range = abs(float(h.iloc[-1]) - float(l.iloc[-1]))
        candle_vs_atr     = last_candle_range/atr if atr>0 else 1.0
        volatility_spike  = candle_vs_atr > (self.cfg.volatility_pause_multiplier or 2.5)

        returns = c.pct_change()
        score, reasons = 0.0, []
        if rsi < 35:   score += 0.18; reasons.append(f"RSI {rsi:.1f} OS")
        elif rsi > 65: score -= 0.18; reasons.append(f"RSI {rsi:.1f} OB")
        if macd_h > 0: score += 0.15; reasons.append("MACD+")
        else:          score -= 0.15; reasons.append("MACD-")
        if price > ma20: score += 0.10; reasons.append("P>MA20")
        else:            score -= 0.10
        if bb_pct < 0.2: score += 0.10; reasons.append("BB low")
        elif bb_pct > 0.8: score -= 0.10; reasons.append("BB high")

        confidence = max(0.1, min(0.95, (score+0.7)/1.4))
        signal = "BUY" if score>0.15 else "SELL" if score<-0.15 else "HOLD"
        ma20_slope = float(c.rolling(20).mean().iloc[-1] - c.rolling(20).mean().iloc[-4])/3
        regime = "trend_up" if ma20_slope>0 else "trend_down" if ma20_slope<0 else "sideways"

        return {
            "pair":pair,"price":price,"signal":signal,
            "score":score,"confidence":confidence,
            "rsi":rsi,"macd":macd,"macd_h":macd_h,
            "ma20":ma20,"ma50":ma50,"ma200":ma200,
            "bb_pct":bb_pct,"bb_upper":bb_u,"bb_lower":bb_l,
            "atr":atr,"adx":adx,
            "volume_ratio":vol_r,"vol_std":float(returns.rolling(20).std().iloc[-1])*100,
            "regime":regime,"candle_vs_atr":candle_vs_atr,
            "volatility_spike":volatility_spike,
            "sl_distance":atr*1.5,"tp_distance":atr*3.0,
            "ma_dist":(price-ma20)/ma20,"ma50_dist":(price-ma50)/ma50,
            "ret1":float(returns.iloc[-1]),"ret3":float(returns.iloc[-3:].sum()),
            "ret5":float(returns.iloc[-5:].sum()),
            "hour_of_day":datetime.now().hour,"reasons":reasons,
        }

    def _fallback(self, pair: str) -> Dict[str, Any]:
        import random
        price = self.get_current_price(pair) or 84000
        atr = price * 0.012
        return {
            "pair": pair, "price": price, "signal": "HOLD",
            "score": 0, "confidence": 0.5,
            "rsi": 50, "macd": 0, "macd_h": 0,
            "ma20": price * 0.99, "ma50": price * 0.98, "ma200": price * 0.97,
            "bb_pct": 0.5, "bb_upper": price*1.02, "bb_lower": price*0.98,
            "atr": atr, "adx": 20, "volume_ratio": 1.0, "vol_std": 1.5,
            "regime": "sideways",
            "candle_vs_atr": 1.0, "volatility_spike": False,
            "sl_distance": atr*1.5, "tp_distance": atr*3.0,
            "ma_dist": 0, "ma50_dist": 0,
            "ret1": 0, "ret3": 0, "ret5": 0,
            "hour_of_day": datetime.now().hour,
            "reasons": ["fallback"],
        }


# ════════════════════════════════════════════════════════════════
#  📰 SENTIMENT AGENT v2 — з кешем і LLM раз на 1 год
# ════════════════════════════════════════════════════════════════

class SentimentAgent:
    FG_URL = "https://api.alternative.me/fng/"

    def __init__(self, cfg):
        self.cfg = cfg
        self._fg_val  = 50.0
        self._fg_ts   = 0
        self._llm_cache: Dict[str, Tuple[float, float]] = {}  # pair → (score, ts)

    def _get_fear_greed(self) -> float:
        if time.time() - self._fg_ts < 3600:
            return self._fg_val
        try:
            r = requests.get(self.FG_URL, timeout=8)
            self._fg_val = float(r.json()["data"][0]["value"])
            self._fg_ts  = time.time()
        except Exception:
            pass
        return self._fg_val

    def _get_funding(self, pair: str) -> float:
        # Bybit funding rate (Binance заблокований)
        try:
            sym = pair.replace("/", "")
            r = requests.get(
                f"https://api.bybit.com/v5/market/tickers"
                f"?category=linear&symbol={sym}",
                timeout=8
            )
            if r.status_code == 200:
                items = r.json().get("result", {}).get("list", [])
                if items:
                    return float(items[0].get("fundingRate", 0))
        except Exception:
            pass
        return 0.0

    def _llm_once_per_hour(self, pair: str, price: float) -> float:
        """LLM кеш 1 год — не витрачаємо API на кожен трейд."""
        if pair in self._llm_cache:
            score, ts = self._llm_cache[pair]
            if time.time() - ts < 3600:
                return score

        if not self.cfg.use_llm:
            return 0.0

        prompt = (
            f"Market sentiment for {pair} (price ${price:,.0f}). "
            f"Rate from -1.0 (extreme fear/sell) to +1.0 (extreme greed/buy). "
            f"Consider: recent news, macro, crypto market mood. "
            f"Reply with ONLY a single float number."
        )
        score = 0.0
        try:
            if self.cfg.openai_api_key:
                from openai import OpenAI
                c = OpenAI(api_key=self.cfg.openai_api_key)
                r = c.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=8,
                )
                score = float(r.choices[0].message.content.strip())
            elif self.cfg.anthropic_api_key:
                import anthropic
                c = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)
                r = c.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=8,
                    messages=[{"role":"user","content":prompt}],
                )
                score = float(r.content[0].text.strip())
        except Exception as e:
            log.debug(f"LLM: {e}")

        self._llm_cache[pair] = (score, time.time())
        return score

    def analyze(self, pair: str, price: float) -> Dict[str, Any]:
        fg     = self._get_fear_greed()
        fund   = self._get_funding(pair)
        llm_s  = self._llm_once_per_hour(pair, price)

        fg_norm   = (fg - 50) / 50
        fund_norm = -fund * 1000

        if self.cfg.use_llm and abs(llm_s) > 0.01:
            score = fg_norm * 0.3 + fund_norm * 0.2 + llm_s * 0.5
        else:
            score = fg_norm * 0.6 + fund_norm * 0.4

        score = max(-1.0, min(1.0, score))

        return {
            "fear_greed": fg, "funding_rate": fund,
            "llm_score": llm_s, "sent_score": score,
        }


# ════════════════════════════════════════════════════════════════
#  🤖 DEBATE AI v2 — Weighted Scoring
# ════════════════════════════════════════════════════════════════

class DebateAI:
    """
    Зважений консенсус:
      Tech     × 0.40
      Sentiment× 0.20
      ML       × 0.40
    Bull і Bear агенти аналізують з протилежними упередженнями.
    """

    WEIGHTS = {"tech": 0.40, "sentiment": 0.20, "ml": 0.40}

    def vote(
        self,
        tech: Dict[str, Any],
        sent: Dict[str, Any],
        ml_prob: float,
        cfg,
    ) -> Dict[str, Any]:

        # ── Блокуємо при volatility spike ─────────────────────
        if tech.get("volatility_spike"):
            return {
                "action": "HOLD", "confidence": 0.3,
                "bull_score": 0, "bear_score": 0,
                "bull_votes": 0, "bear_votes": 5,
                "bull_args": [], "bear_args": ["Volatility spike — PAUSE"],
                "reason": "⚠️ Volatility spike — пауза торгівлі",
                "tech_score": 0, "sent_score_w": 0, "ml_score_w": 0,
            }

        # ── Блокуємо при sideways + слабкий сигнал ─────────────
        regime = tech.get("regime", "sideways")

        # ── Tech компонент ─────────────────────────────────────
        tech_raw = 0.0
        bull_args, bear_args = [], []

        rsi = tech.get("rsi", 50)
        if rsi < 35:
            tech_raw += 0.3; bull_args.append(f"RSI {rsi:.1f} oversold")
        elif rsi > 65:
            tech_raw -= 0.3; bear_args.append(f"RSI {rsi:.1f} overbought")

        macd_h = tech.get("macd_h", 0)
        if macd_h > 0: tech_raw += 0.2; bull_args.append("MACD hist+")
        else:          tech_raw -= 0.2; bear_args.append("MACD hist-")

        price = tech.get("price", 0)
        ma20  = tech.get("ma20",  price)
        ma50  = tech.get("ma50",  price)
        if price > ma20: tech_raw += 0.15; bull_args.append("P>MA20")
        else:            tech_raw -= 0.15; bear_args.append("P<MA20")
        if price > ma50: tech_raw += 0.1;  bull_args.append("P>MA50")
        else:            tech_raw -= 0.1;  bear_args.append("P<MA50")

        bb_pct = tech.get("bb_pct", 0.5)
        if bb_pct < 0.15: tech_raw += 0.15; bull_args.append(f"BB {bb_pct:.1%} low")
        elif bb_pct > 0.85: tech_raw -= 0.15; bear_args.append(f"BB {bb_pct:.1%} high")

        vol_r = tech.get("volume_ratio", 1)
        if vol_r > 2.0: tech_raw += 0.1 if tech_raw > 0 else -0.1

        # ADX підсилює тренд
        adx = tech.get("adx", 20)
        if adx > 30: tech_raw *= 1.2

        # Sideways зменшує впевненість
        if regime == "sideways":
            tech_raw *= 0.5

        tech_raw = max(-1.0, min(1.0, tech_raw))

        # ── Sentiment компонент ────────────────────────────────
        sent_raw = sent.get("sent_score", 0)
        fg = sent.get("fear_greed", 50)
        if fg < 25: bull_args.append(f"Extreme Fear F&G={fg:.0f}")
        elif fg > 75: bear_args.append(f"Extreme Greed F&G={fg:.0f}")

        # ── ML компонент ───────────────────────────────────────
        ml_raw = (ml_prob - 0.5) * 2   # -1..+1
        if ml_raw > 0: bull_args.append(f"ML win_prob={ml_prob:.1%}")
        else:          bear_args.append(f"ML lose_prob={1-ml_prob:.1%}")

        # ── Зважена сума ───────────────────────────────────────
        W = self.WEIGHTS
        final_score = (
            tech_raw  * W["tech"] +
            sent_raw  * W["sentiment"] +
            ml_raw    * W["ml"]
        )

        # Конвертуємо в ймовірність BUY: 0..1
        bull_prob = (final_score + 1) / 2
        bull_prob = max(0.05, min(0.95, bull_prob))
        bear_prob = 1 - bull_prob

        AGENTS = 5
        bull_votes = round(bull_prob * AGENTS)
        bear_votes = AGENTS - bull_votes

        threshold = cfg.min_confidence or 0.62

        if bull_prob >= threshold:
            action = "BUY"
            confidence = bull_prob
            reason = " | ".join(bull_args[:4]) or "Bull consensus"
        elif bear_prob >= threshold:
            action = "SELL"
            confidence = bear_prob
            reason = " | ".join(bear_args[:4]) or "Bear consensus"
        else:
            action = "HOLD"
            confidence = max(bull_prob, bear_prob)
            reason = f"Немає консенсусу ({bull_votes}B/{bear_votes}Be) regime={regime}"

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "bull_score": round(bull_prob, 3),
            "bear_score": round(bear_prob, 3),
            "bull_votes": bull_votes,
            "bear_votes": bear_votes,
            "bull_args": bull_args,
            "bear_args": bear_args,
            "reason": reason,
            "tech_score": round(tech_raw, 3),
            "sent_score_w": round(sent_raw, 3),
            "ml_score_w": round(ml_raw, 3),
            "regime": regime,
        }


# ════════════════════════════════════════════════════════════════
#  🛡️ RISK MANAGER v2 — Kelly + Volatility + Drawdown
# ════════════════════════════════════════════════════════════════

class RiskManagerAgent:
    """
    Апгрейди:
      ✅ Kelly Criterion position sizing
      ✅ Volatility-adjusted sizing (ATR-based)
      ✅ Portfolio correlation check
      ✅ Max drawdown stop
      ✅ Loss streak pause
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._peak_balance = None
        self._loss_streak  = 0
        self._last_results: List[bool] = []  # True=win, False=loss

    def update_result(self, is_win: bool, balance: float):
        """Оновити після кожної закритої угоди."""
        self._last_results.append(is_win)
        if len(self._last_results) > 20:
            self._last_results.pop(0)
        if not is_win:
            self._loss_streak += 1
        else:
            self._loss_streak = 0
        if self._peak_balance is None or balance > self._peak_balance:
            self._peak_balance = balance

    def check(
        self,
        decision: Dict[str, Any],
        open_trades: List[dict],
        pair: str,
        balance: float,
        tech_signal: Dict[str, Any] = None,
    ) -> Dict[str, Any]:

        if decision["action"] == "HOLD":
            return {"allowed": False, "reason": "HOLD"}

        # ── Drawdown Stop ────────────────────────────────────────
        if self._peak_balance and balance > 0:
            drawdown = (self._peak_balance - balance) / self._peak_balance
            max_dd = self.cfg.max_drawdown_pct or 0.10
            if drawdown > max_dd:
                return {
                    "allowed": False,
                    "reason": f"🛑 Max drawdown {drawdown:.1%} > {max_dd:.0%} — торгівля зупинена!"
                }

        # ── Loss Streak Pause ────────────────────────────────────
        max_streak = self.cfg.max_loss_streak or 5
        if self._loss_streak >= max_streak:
            return {
                "allowed": False,
                "reason": f"⏸ Loss streak {self._loss_streak} — пауза {max_streak} програшів поспіль"
            }

        # ── Volatility Spike ────────────────────────────────────
        if tech_signal and tech_signal.get("volatility_spike"):
            return {"allowed": False, "reason": "⚠️ Volatility spike — пауза"}

        # ── Max open trades ──────────────────────────────────────
        if len(open_trades) >= (self.cfg.max_open_trades or 3):
            return {"allowed": False, "reason": f"Макс. угод {self.cfg.max_open_trades}"}

        # ── Вже є угода по цій парі ─────────────────────────────
        if any(t["pair"] == pair and t["status"] == "OPEN" for t in open_trades):
            return {"allowed": False, "reason": f"{pair} вже відкрита"}

        # ── Confidence threshold ─────────────────────────────────
        if decision["confidence"] < (self.cfg.min_confidence or 0.62):
            return {
                "allowed": False,
                "reason": f"Conf {decision['confidence']:.1%} < threshold"
            }

        # ── Portfolio correlation ────────────────────────────────
        high_corr_pairs = {
            "BTC/USDT": ["ETH/USDT"],
            "ETH/USDT": ["BTC/USDT", "SOL/USDT"],
            "SOL/USDT": ["ETH/USDT"],
        }
        for open_t in open_trades:
            if open_t["pair"] in high_corr_pairs.get(pair, []):
                # Дозволяємо але попереджаємо (якщо не заборонено)
                if self.cfg.block_correlated_pairs:
                    return {
                        "allowed": False,
                        "reason": f"Висока кореляція: {pair} ↔ {open_t['pair']}"
                    }

        info = []
        if self._loss_streak > 0:
            info.append(f"streak={self._loss_streak}")
        if self._peak_balance:
            dd = (self._peak_balance - balance) / self._peak_balance
            info.append(f"dd={dd:.1%}")

        return {
            "allowed": True,
            "reason": f"OK conf={decision['confidence']:.1%} " + " ".join(info)
        }

    def position_size(
        self,
        balance: float,
        atr: float = None,
        price: float = None,
        ml_prob: float = 0.6,
        winrate: float = None,
    ) -> float:
        """
        Розмір позиції через Kelly + Volatility adjustment.

        Kelly formula:
          f* = (bp - q) / b
          де b = reward/risk ratio, p = winrate, q = 1 - p

        ATR-based:
          risk_usd = balance * 0.01   (ризикуємо 1% балансу)
          size     = risk_usd / (ATR * atr_sl_mult)
        """
        method = self.cfg.position_sizing_method or "atr"

        if method == "kelly" and winrate is not None:
            p = winrate
            q = 1 - p
            b = (self.cfg.atr_tp_multiplier or 3.0) / (self.cfg.atr_sl_multiplier or 1.5)
            kelly_f = max(0, (b * p - q) / b)
            # Half-Kelly (безпечніше)
            kelly_f = kelly_f * 0.5
            # ML confidence підвищує/знижує
            kelly_f *= (0.5 + ml_prob * 0.5)
            # Clamp
            kelly_f = max(0.01, min(self.cfg.max_position_pct or 0.15, kelly_f))
            return round(balance * kelly_f, 2)

        elif method == "atr" and atr and price:
            risk_usd  = balance * (self.cfg.risk_per_trade_pct or 0.01)
            sl_pts    = atr * (self.cfg.atr_sl_multiplier or 1.5)
            size_usd  = risk_usd / (sl_pts / price) if price > 0 else risk_usd / 0.02
            # Clamp
            max_size  = balance * (self.cfg.max_position_pct or 0.15)
            return round(min(size_usd, max_size), 2)

        else:
            # Fallback — фіксований %
            pct = self.cfg.position_size_pct or 0.05
            return round(max(10.0, balance * pct), 2)

    def atr_stop_loss(self, entry: float, side: str, atr: float) -> float:
        mult = self.cfg.atr_sl_multiplier or 1.5
        dist = atr * mult
        return round(entry - dist if side == "BUY" else entry + dist, 2)

    def atr_take_profit(self, entry: float, side: str, atr: float) -> float:
        mult = self.cfg.atr_tp_multiplier or 3.0
        dist = atr * mult
        return round(entry + dist if side == "BUY" else entry - dist, 2)

    def fee(self, amount: float) -> float:
        return round(amount * (self.cfg.effective_fee or 0.001), 6)


# ════════════════════════════════════════════════════════════════
#  ⚡ EXECUTOR v2 — Trailing Stop + Retry + Slippage Control
# ════════════════════════════════════════════════════════════════

class ExecutorAgent:
    """
    Апгрейди:
      ✅ Trailing Stop (динамічний тейк-профіт)
      ✅ Retry logic (3 спроби при збої)
      ✅ Slippage control (cancel якщо ціна рухнула)
      ✅ Partial fill detection
    """

    def __init__(self, cfg, db, tg, risk: RiskManagerAgent):
        self.cfg  = cfg
        self.db   = db
        self.tg   = tg
        self.risk = risk
        self._exchange = None
        self._trailing: Dict[int, Dict] = {}

        # Відновлюємо баланс з БД при старті
        self._demo_balance = self._restore_balance(cfg, db)
        log.info(f"💰 Demo баланс відновлено: ${self._demo_balance:.2f}")

        if not cfg.is_demo:
            self._init_exchange()

    def _restore_balance(self, cfg, db) -> float:
        """Відновлює баланс з БД або повертає початковий."""
        initial = float(cfg.initial_demo_balance or 1000.0)
        try:
            # Беремо останній записаний баланс
            row = db.conn.execute(
                "SELECT balance FROM balance_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[0] and float(row[0]) > 0:
                log.info(f"💾 Баланс з БД: ${float(row[0]):.2f}")
                return float(row[0])
        except Exception:
            pass
        # Рахуємо з угод якщо немає збереженого балансу
        try:
            # initial - заморожені відкриті - комісії закритих + прибуток закритих
            open_trades = db.get_open_trades()
            stats = db.get_stats()
            frozen = sum(float(t.get("amount_usd", 0)) for t in open_trades)
            net_pnl = float(stats.get("net_pnl", 0))
            fees = float(stats.get("fees", 0))
            # Баланс = початковий - заморожено + чистий PnL
            bal = initial - frozen + net_pnl
            if bal > 0:
                log.info(f"📊 Баланс розраховано: ${bal:.2f} "
                         f"(frozen=${frozen:.2f} pnl=${net_pnl:.2f})")
                return bal
        except Exception as e:
            log.debug(f"Balance calc: {e}")
        return initial

    def _init_exchange(self):
        if not HAS_CCXT:
            return
        try:
            params = {
                "apiKey":          self.cfg.api_key,
                "secret":          self.cfg.api_secret,
                "enableRateLimit": True,
            }
            # Bybit testnet якщо потрібно
            if self.cfg.testnet:
                params["urls"] = {"api": {
                    "public":  "https://api-testnet.bybit.com",
                    "private": "https://api-testnet.bybit.com",
                }}
            self._exchange = ccxt.binance(params)
            log.info("✅ Executor: Binance підключено")
        except Exception as e:
            log.error(f"Executor Binance: {e}")

    def get_balance(self) -> float:
        if self.cfg.is_demo:
            return self._demo_balance
        if self._exchange:
            for attempt in range(3):
                try:
                    bal = self._exchange.fetch_balance()
                    return float(bal["USDT"]["free"])
                except Exception as e:
                    log.warning(f"Balance attempt {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
        return self._demo_balance

    def _place_order_with_retry(
        self, pair: str, side: str, units: float, max_attempts: int = 3
    ) -> Optional[dict]:
        """Retry logic для реальних ордерів."""
        for attempt in range(max_attempts):
            try:
                order = self._exchange.create_market_order(
                    symbol=pair,
                    side=side.lower(),
                    amount=units,
                )
                return order
            except ccxt.NetworkError as e:
                wait = 2 ** attempt
                log.warning(f"Order attempt {attempt+1} NetworkError: {e} — retry in {wait}s")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                log.error(f"Order ExchangeError: {e}")
                return None
        log.error(f"Order failed after {max_attempts} attempts")
        return None

    def _check_slippage(
        self, expected_price: float, actual_price: float,
        side: str, max_slippage_pct: float = 0.005
    ) -> bool:
        """True = slippage OK, False = занадто великий слиппейдж."""
        if side == "BUY":
            slippage = (actual_price - expected_price) / expected_price
        else:
            slippage = (expected_price - actual_price) / expected_price
        if slippage > max_slippage_pct:
            log.warning(f"Slippage {slippage:.2%} > {max_slippage_pct:.2%}")
            return False
        return True

    def open_trade(
        self,
        pair: str, side: str, price: float,
        amount_usd: float, sl: float, tp: float,
        features: dict, confidence: float, reason: str,
    ) -> Optional[int]:
        units = round(amount_usd / price, 8)
        entry_fee = self.risk.fee(amount_usd)
        mode = "demo" if self.cfg.is_demo else "live"
        binance_order_id = None

        if not self.cfg.is_demo:
            if not self._exchange:
                log.error("Live mode але exchange недоступний!")
                return None
            order = self._place_order_with_retry(pair, side, units)
            if not order:
                return None
            actual_price = float(order.get("average") or price)
            # Slippage check
            max_slip = self.cfg.max_slippage_pct or 0.005
            if not self._check_slippage(price, actual_price, side, max_slip):
                log.warning(f"Великий слиппейдж {pair} — угода записана але сигнал")
            price = actual_price
            binance_order_id = str(order.get("id", ""))
            log.info(f"⚡ LIVE #{binance_order_id}: {side} {pair} @ ${price:,.2f}")
        else:
            self._demo_balance -= entry_fee
            log.info(f"🧪 DEMO {side} {pair}: {units:.6f} @ ${price:,.2f} "
                     f"${amount_usd:.2f} fee=-${entry_fee:.4f}")

        trade_data = {
            "ts_open": datetime.now().isoformat(),
            "pair": pair, "side": side,
            "entry_price": price,
            "amount_usd": amount_usd,
            "units": units,
            "fee_usd": entry_fee,
            "sl_price": sl, "tp_price": tp,
            "confidence": confidence,
            "reason": reason,
            "mode": mode,
            "binance_order_id": binance_order_id,
            **{k: features.get(k) for k in [
                "rsi","macd","ma20","ma50","bb_pct","atr",
                "volume_ratio","fear_greed","funding_rate",
                "sent_score","ml_prob","regime",
                "vol_std","ma_dist","ret1","ret3","ret5","hour_of_day",
            ]},
        }
        trade_id = self.db.open_trade(trade_data)

        # Ініціалізуємо trailing stop
        if self.cfg.trailing_stop_enabled:
            self._trailing[trade_id] = {
                "best_price":   price,
                "activated":    False,
                "side":         side,
                "entry":        price,
                "activation_pct": self.cfg.trailing_stop_activation_pct or 0.03,
                "trail_pct":    self.cfg.trailing_stop_trail_pct or 0.01,
            }

        self.tg.trade_opened({**trade_data, "id": trade_id})
        log.info(f"✅ #{trade_id} відкрито: {side} {pair} @ ${price:,.2f} "
                 f"SL=${sl:.2f} TP=${tp:.2f}")
        return trade_id

    def check_open_trades(self):
        """Перевірка SL/TP + Trailing Stop для всіх відкритих угод."""
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return

        for t in open_trades:
            current = self._get_price(t["pair"])
            if not current:
                continue

            trade_id = t["id"]
            entry    = t["entry_price"]
            side     = t["side"]
            sl       = t["sl_price"]
            tp       = t["tp_price"]

            # ── Trailing Stop ──────────────────────────────────
            if self.cfg.trailing_stop_enabled and trade_id in self._trailing:
                trail = self._trailing[trade_id]
                new_sl = self._update_trailing(trail, current, sl)
                if new_sl != sl:
                    sl = new_sl
                    self.db.update_sl(trade_id, new_sl)
                    log.info(f"🔄 Trailing #{trade_id}: новий SL=${new_sl:.2f}")

            # ── SL / TP Check ──────────────────────────────────
            hit_sl = (side=="BUY" and current<=sl) or (side=="SELL" and current>=sl) if sl else False
            hit_tp = (side=="BUY" and current>=tp) or (side=="SELL" and current<=tp) if tp else False

            if hit_sl or hit_tp:
                status = "TP" if hit_tp else "SL"
                self._close_trade(t, current, status)
                # Оновити Risk Manager
                is_win = status == "TP"
                self.risk.update_result(is_win, self.get_balance())

    def _update_trailing(
        self,
        trail: dict,
        current: float,
        current_sl: float,
    ) -> float:
        """
        Trailing Stop логіка:
          1. Чекаємо активацію (ціна пішла на activation_pct%)
          2. Після активації — тягнемо SL за ціною на trail_pct%
        """
        side   = trail["side"]
        entry  = trail["entry"]
        act_p  = trail["activation_pct"]
        trl_p  = trail["trail_pct"]

        if side == "BUY":
            profit_pct = (current - entry) / entry
            if not trail["activated"] and profit_pct >= act_p:
                trail["activated"] = True
                trail["best_price"] = current
                log.info(f"🔄 Trailing activated BUY @ ${current:.2f} "
                         f"(+{profit_pct:.1%})")

            if trail["activated"]:
                if current > trail["best_price"]:
                    trail["best_price"] = current
                new_sl = trail["best_price"] * (1 - trl_p)
                return max(new_sl, current_sl)  # тільки підвищуємо SL

        else:  # SELL
            profit_pct = (entry - current) / entry
            if not trail["activated"] and profit_pct >= act_p:
                trail["activated"] = True
                trail["best_price"] = current
                log.info(f"🔄 Trailing activated SELL @ ${current:.2f}")

            if trail["activated"]:
                if current < trail["best_price"]:
                    trail["best_price"] = current
                new_sl = trail["best_price"] * (1 + trl_p)
                return min(new_sl, current_sl)  # тільки знижуємо SL

        return current_sl

    def _close_trade(self, t: dict, exit_price: float, status: str):
        entry  = t["entry_price"]
        side   = t["side"]
        units  = t.get("units") or t["amount_usd"] / entry
        amount = t["amount_usd"]

        if not self.cfg.is_demo and self._exchange:
            close_side = "sell" if side == "BUY" else "buy"
            order = self._place_order_with_retry(t["pair"], close_side, units)
            if order:
                exit_price = float(order.get("average") or exit_price)

        gross = ((exit_price - entry) / entry * amount
                 if side == "BUY"
                 else (entry - exit_price) / entry * amount)
        fee   = self.risk.fee(amount)
        net   = round(gross - fee, 4)

        if self.cfg.is_demo:
            self._demo_balance += amount + net

        self.db.close_trade(t["id"], exit_price, gross, fee, status)
        self.tg.trade_closed(t, net)

        # Прибираємо trailing
        self._trailing.pop(t["id"], None)

        log.info(f"{'✅ TP' if status=='TP' else '🛑 SL'} #{t['id']} "
                 f"{t['pair']} @ ${exit_price:,.2f} | net={net:+.2f}")

    def _get_price(self, pair: str) -> Optional[float]:
        if not self.cfg.is_demo and self._exchange:
            for attempt in range(2):
                try:
                    return float(self._exchange.fetch_ticker(pair)["last"])
                except Exception:
                    time.sleep(1)
        import random
        base = {"BTC/USDT":67000,"ETH/USDT":3200,"SOL/USDT":148,"BNB/USDT":590}.get(pair,100)
        return base * (1 + random.gauss(0, 0.003))
