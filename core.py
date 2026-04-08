"""
core.py v2.0 — Апгрейди:
  ✅ Telegram-команди: /status /panic_close /mode_swap /pause /resume /stats
  ✅ Автобекап БД в Telegram раз на добу
  ✅ Нові поля конфігу: trailing stop, ATR SL/TP, Kelly, drawdown
  ✅ update_sl() в Database для trailing stop
"""

import json, os, sqlite3, threading, requests, logging, pickle, time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

log = logging.getLogger("NeuralTrade.Core")


# ════════════════════════════════════════════════════════════════
#  CONFIG v2
# ════════════════════════════════════════════════════════════════

class Config:
    DEFAULTS = {
        # Binance
        "api_key": "", "api_secret": "", "testnet": True, "mode": "demo",

        # Пари та таймфрейм
        "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "timeframe": "1h",
        "cycle_interval_minutes": 60,

        # Ризик — базовий
        "min_confidence":      0.62,
        "max_open_trades":     3,
        "max_drawdown_pct":    0.10,    # 10% drawdown → стоп
        "max_loss_streak":     5,       # 5 програшів поспіль → пауза
        "block_correlated_pairs": False,

        # Position Sizing
        "position_sizing_method": "atr", # "atr" | "kelly" | "fixed"
        "position_size_pct":  0.05,      # fixed fallback
        "risk_per_trade_pct": 0.01,      # 1% балансу під ризик (ATR метод)
        "max_position_pct":   0.15,      # макс. 15% балансу в угоді

        # ATR-based SL/TP
        "atr_sl_multiplier":  1.5,
        "atr_tp_multiplier":  3.0,

        # Trailing Stop
        "trailing_stop_enabled":          True,
        "trailing_stop_activation_pct":   0.03,   # активація при +3%
        "trailing_stop_trail_pct":        0.01,   # відстань трейлу 1%

        # Volatility Filter
        "volatility_pause_multiplier":    2.5,    # пауза якщо свічка > 2.5× ATR

        # Комісії
        "fee_rate":    0.001,   # 0.10% spot
        "use_bnb":     False,   # BNB знижка

        # ML
        "retrain_every_hours":   24,
        "min_trades_for_train":  30,

        # LLM (опціонально)
        "use_llm":           False,
        "openai_api_key":    "",
        "anthropic_api_key": "",

        # Telegram
        "telegram_token":   "",
        "telegram_chat_id": "",
        "telegram_backup_db": True,   # щоденний бекап БД

        # Сервер
        "server_host":  "0.0.0.0",
        "server_port":  8000,

        # Демо
        "initial_demo_balance": 1000.0,

        # DB
        "db_path": "neuraltrade.db",
    }

    def __init__(self, path: str = "config.json"):
        self._data = dict(self.DEFAULTS)
        # 1. Завантажуємо config.json
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if not k.startswith("//"):
                    self._data[k] = v
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULTS, f, indent=2, ensure_ascii=False)
            log.info(f"✅ config.json створено: {path}")

        # 2. ENV змінні Railway перезаписують config.json
        #    Це головне джерело секретів (токени, ключі)
        env_map = {
            "TELEGRAM_TOKEN":    "telegram_token",
            "TELEGRAM_CHAT_ID":  "telegram_chat_id",
            "BINANCE_API_KEY":   "api_key",
            "BINANCE_SECRET":    "api_secret",
            "OPENAI_API_KEY":    "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "TRADING_MODE":      "mode",
            "DB_PATH":           "db_path",
        }
        for env_key, cfg_key in env_map.items():
            val = os.environ.get(env_key, "").strip()
            if val:
                self._data[cfg_key] = val
                log.info(f"✅ ENV {env_key} → cfg.{cfg_key}")

        log.info(f"🔧 Config: mode={self._data.get('mode')} "
                 f"tg={'✅' if self._data.get('telegram_token') else '❌'} "
                 f"db={self._data.get('db_path')}")


    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    @property
    def effective_fee(self) -> float:
        r = self._data.get("fee_rate", 0.001)
        if self._data.get("use_bnb"):
            r *= 0.75
        return r

    @property
    def is_demo(self) -> bool:
        return self._data.get("mode", "demo") == "demo"

    def save(self, path: str = "config.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        safe = dict(self._data)
        for key in ("api_key","api_secret","openai_api_key","anthropic_api_key"):
            safe.pop(key, None)
        return safe


# ════════════════════════════════════════════════════════════════
#  DATABASE v2
# ════════════════════════════════════════════════════════════════

class Database:
    def __init__(self, path: str):
        self.path  = path
        self._lock = threading.Lock()
        self.conn  = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()
        log.info(f"📦 БД: {path}")

    def _init(self):
        with self._lock:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_open          TEXT    NOT NULL,
                ts_close         TEXT,
                pair             TEXT    NOT NULL,
                side             TEXT    NOT NULL,
                entry_price      REAL    NOT NULL,
                exit_price       REAL,
                amount_usd       REAL    NOT NULL,
                units            REAL,
                gross_pnl        REAL,
                fee_usd          REAL    DEFAULT 0,
                net_pnl          REAL,
                status           TEXT    DEFAULT 'OPEN',
                sl_price         REAL,
                tp_price         REAL,
                confidence       REAL,
                reason           TEXT,
                mode             TEXT    DEFAULT 'demo',
                regime           TEXT,
                rsi              REAL, macd        REAL,
                ma20             REAL, ma50        REAL,
                bb_pct           REAL, atr         REAL,
                volume_ratio     REAL, vol_std     REAL,
                ma_dist          REAL, ma50_dist   REAL,
                ret1             REAL, ret3        REAL, ret5 REAL,
                fear_greed       REAL, funding_rate REAL,
                sent_score       REAL, ml_prob     REAL,
                hour_of_day      INTEGER,
                outcome          INTEGER,
                binance_order_id TEXT,
                timeframe        TEXT    DEFAULT '1h'
            );

            CREATE TABLE IF NOT EXISTS signals (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                pair       TEXT NOT NULL,
                action     TEXT NOT NULL,
                confidence REAL, price       REAL,
                executed   INTEGER DEFAULT 0,
                reason     TEXT,
                bull_votes INTEGER DEFAULT 0,
                bear_votes INTEGER DEFAULT 0,
                regime     TEXT,
                mode       TEXT DEFAULT 'demo'
            );

            CREATE TABLE IF NOT EXISTS ml_models (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                pair       TEXT NOT NULL,
                accuracy   REAL, n_samples  INTEGER,
                model_blob BLOB
            );

            CREATE TABLE IF NOT EXISTS balance_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance   REAL NOT NULL,
                mode      TEXT DEFAULT 'demo'
            );
            """)
            self.conn.commit()

    # ── TRADES ──────────────────────────────────────────────────
    def open_trade(self, t: dict) -> int:
        cols = [
            "ts_open","pair","side","entry_price","amount_usd","units",
            "fee_usd","status","sl_price","tp_price","confidence","reason","mode",
            "regime","rsi","macd","ma20","ma50","bb_pct","atr",
            "volume_ratio","vol_std","ma_dist","ma50_dist",
            "ret1","ret3","ret5",
            "fear_greed","funding_rate","sent_score","ml_prob","hour_of_day",
            "timeframe",
        ]
        vals = [t.get(c) for c in cols]
        vals[0] = vals[0] or datetime.now().isoformat()
        vals[7] = "OPEN"
        ph = ",".join(["?"]*len(cols))
        with self._lock:
            c = self.conn.execute(
                f"INSERT INTO trades ({','.join(cols)}) VALUES ({ph})", vals
            )
            self.conn.commit()
            return c.lastrowid

    def close_trade(self, tid: int, exit_price: float,
                    gross: float, fee: float, status: str):
        # ── БАГ-ФІКС: отримуємо вже збережену entry fee,
        #    додаємо exit fee, рахуємо total fee і net ──────────
        row = self.conn.execute(
            "SELECT fee_usd FROM trades WHERE id=?", (tid,)
        ).fetchone()
        entry_fee   = float(row["fee_usd"]) if row and row["fee_usd"] else 0.0
        total_fee   = round(entry_fee + fee, 6)   # entry + exit fee
        net         = round(gross - total_fee, 4)
        outcome     = 1 if net > 0 else 0
        with self._lock:
            self.conn.execute("""
                UPDATE trades SET
                    ts_close=?,exit_price=?,gross_pnl=?,
                    fee_usd=?,net_pnl=?,status=?,outcome=?
                WHERE id=?
            """, (datetime.now().isoformat(), exit_price,
                  gross, total_fee, net, status, outcome, tid))
            self.conn.commit()

    def update_sl(self, trade_id: int, new_sl: float):
        """Оновити SL (для trailing stop)."""
        with self._lock:
            self.conn.execute(
                "UPDATE trades SET sl_price=? WHERE id=?",
                (new_sl, trade_id)
            )
            self.conn.commit()

    def get_open_trades(self) -> List[dict]:
        try:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT * FROM trades WHERE status='OPEN'"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_trades(self, limit: int = 500, pair: str = None) -> List[dict]:
        if pair:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE pair=? ORDER BY id DESC LIMIT ?",
                (pair, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_closed_with_features(self, pair: str = None) -> List[dict]:
        q = "SELECT * FROM trades WHERE outcome IS NOT NULL"
        p = []
        if pair:
            q += " AND pair=?"
            p.append(pair)
        return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def get_stats(self) -> dict:
        c = self.conn
        # Тільки ЗАКРИТІ угоди для статистики
        total = c.execute("SELECT COUNT(*) FROM trades WHERE status NOT IN ('OPEN')").fetchone()[0]
        wins  = c.execute("SELECT COUNT(*) FROM trades WHERE outcome=1 AND status NOT IN ('OPEN')").fetchone()[0]
        # Комісії тільки закритих угод (відкриті ще не реалізовані)
        fees  = c.execute("SELECT COALESCE(SUM(fee_usd),0) FROM trades WHERE status NOT IN ('OPEN')").fetchone()[0]
        gross = c.execute("SELECT COALESCE(SUM(gross_pnl),0) FROM trades WHERE status NOT IN ('OPEN')").fetchone()[0]
        best  = c.execute("SELECT COALESCE(MAX(net_pnl),0) FROM trades WHERE status NOT IN ('OPEN')").fetchone()[0]
        worst = c.execute("SELECT COALESCE(MIN(net_pnl),0) FROM trades WHERE status NOT IN ('OPEN')").fetchone()[0]
        # net_pnl = gross - fees (тільки закриті)
        net   = round(float(gross) - float(fees), 2)
        return {
            "total":     total,
            "wins":      wins,
            "losses":    total - wins,
            "winrate":   round(wins / total * 100, 1) if total else 0,
            "gross_pnl": round(float(gross), 2),
            "fees":      round(float(fees), 4),
            "net_pnl":   net,
            "best":      round(float(best), 2),
            "worst":     round(float(worst), 2),
        }

    def get_winrate(self) -> float:
        s = self.get_stats()
        return s["winrate"] / 100 if s["total"] else 0.5

    # ── SIGNALS ─────────────────────────────────────────────────
    def save_signal(self, s: dict):
        with self._lock:
            self.conn.execute("""
                INSERT INTO signals
                (timestamp,pair,action,confidence,price,executed,
                 reason,bull_votes,bear_votes,regime,mode)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                s.get("timestamp", datetime.now().isoformat()),
                s["pair"], s["action"], s.get("confidence"),
                s.get("price"), s.get("executed", 0),
                s.get("reason"),
                s.get("bull_votes", 0), s.get("bear_votes", 0),
                s.get("regime"), s.get("mode","demo"),
            ))
            self.conn.commit()

    # ── ML ──────────────────────────────────────────────────────
    def save_model(self, pair: str, acc: float, n: int, blob: bytes):
        with self._lock:
            self.conn.execute("""
                INSERT INTO ml_models
                (timestamp,pair,accuracy,n_samples,model_blob)
                VALUES (?,?,?,?,?)
            """, (datetime.now().isoformat(), pair, acc, n, blob))
            self.conn.commit()

    def load_model(self, pair: str) -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT model_blob FROM ml_models WHERE pair=? ORDER BY id DESC LIMIT 1",
            (pair,)
        ).fetchone()
        return row["model_blob"] if row else None

    # ── BALANCE ─────────────────────────────────────────────────
    def save_balance(self, balance: float, mode: str = "demo"):
        with self._lock:
            self.conn.execute(
                "INSERT INTO balance_history (timestamp,balance,mode) VALUES (?,?,?)",
                (datetime.now().isoformat(), balance, mode)
            )
            self.conn.commit()

    def export_csv(self, path: str = "trades_export.csv"):
        try:
            import pandas as pd
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY id", self.conn)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            log.info(f"📊 CSV: {path} ({len(df)} угод)")
            return path
        except ImportError:
            log.warning("pandas не встановлено")


# ════════════════════════════════════════════════════════════════
#  TELEGRAM NOTIFIER v2 — з командами
# ════════════════════════════════════════════════════════════════

class TelegramNotifier:
    """
    Сповіщення + обробка команд:
      /status      — баланс, відкриті угоди, режим
      /stats       — статистика всіх угод
      /pause       — призупинити торгівлю
      /resume      — відновити
      /mode_swap   — demo ↔ live
      /panic_close — закрити всі угоди + зупинити бота
      /backup      — надіслати БД файлом
    """

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        self._base   = f"https://api.telegram.org/bot{token}"
        self._paused = False
        self._bot_ref = None     # встановлює бот після ініціалізації

    @property
    def is_paused(self) -> bool:
        return self._paused

    def set_bot(self, bot):
        """Посилання на головний бот-об'єкт (для panic_close)."""
        self._bot_ref = bot

    def send(self, text: str):
        if not self.enabled:
            return
        try:
            r = requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self.chat_id,
                      "text": text, "parse_mode": "HTML"},
                timeout=8,
            )
            if not r.ok:
                log.warning(f"Telegram send {r.status_code}: {r.text[:100]}")
        except Exception as e:
            log.error(f"Telegram send: {e}")

    def send_file(self, filepath: str, caption: str = ""):
        """Надіслати файл у Telegram (для бекапу БД)."""
        if not self.enabled or not os.path.exists(filepath):
            return
        try:
            with open(filepath, "rb") as f:
                requests.post(
                    f"{self._base}/sendDocument",
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"document": f},
                    timeout=30,
                )
            log.info(f"📤 Надіслано файл: {filepath}")
        except Exception as e:
            log.error(f"Telegram file: {e}")

    def poll_commands(self, db, cfg):
        """
        Polling команд — запускається в окремому потоці.
        Перевіряє нові повідомлення кожні 5 секунд.
        """
        # Чекаємо поки token і chat_id з'являться (можуть завантажитись пізніше)
        for _ in range(30):  # чекаємо до 30 сек
            if self.token and self.chat_id:
                break
            time.sleep(1)

        if not self.token:
            log.warning("Telegram: TELEGRAM_TOKEN не встановлено в Railway Variables!")
            return
        if not self.chat_id:
            log.warning("Telegram: TELEGRAM_CHAT_ID не встановлено — чекаємо першого повідомлення")

        self.enabled = True  # оновлюємо
        log.info(f"📱 Telegram polling запущено | token={'✅' if self.token else '❌'} | chat_id={'✅' if self.chat_id else '⏳'}")
        offset = 0

        # Логуємо що маємо
        log.info(f"📱 Telegram: token={'✅' if self.token else '❌'} chat_id='{self.chat_id}'")

        while True:
            try:
                r = requests.get(
                    f"{self._base}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                    timeout=35,
                )
                if not r.ok:
                    log.error(f"Telegram getUpdates HTTP {r.status_code}: {r.text[:100]}")
                    time.sleep(10)
                    continue

                updates = r.json().get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    msg    = upd.get("message", {})
                    text   = msg.get("text", "").strip()
                    chat   = str(msg.get("chat", {}).get("id", ""))

                    log.info(f"📱 Telegram повідомлення від chat={chat}: '{text}'")

                    # Авто-оновлення chat_id при розбіжності
                    stored = str(self.chat_id).strip()
                    incoming = chat.strip()
                    if not stored:
                        # Перший раз — зберігаємо
                        self.chat_id = incoming
                        log.info(f"📱 chat_id встановлено: {incoming}")
                    elif stored != incoming:
                        # Розбіжність — оновлюємо і повідомляємо
                        log.warning(f"📱 chat_id змінився: '{stored}' → '{incoming}'")
                        self.chat_id = incoming
                        self.send(
                            f"⚠️ Chat ID оновлено: <code>{incoming}</code>\n"
                            f"Додай в Railway Variables:\n"
                            f"TELEGRAM_CHAT_ID = <code>{incoming}</code>"
                        )

                    # Виконуємо команду для будь-якого chat
                    self._handle_command(text, db, cfg)

            except Exception as e:
                log.error(f"Telegram poll помилка: {e}", exc_info=True)
            time.sleep(3)

    def _handle_command(self, text: str, db, cfg):
        """Обробляє команди Telegram бота."""
        cmd = text.lower().split()[0] if text else ""
        log.info(f"📱 Telegram команда: {cmd}")

        try:
            # ── /start, /help ─────────────────────────────────────
            if cmd in ("/start", "/help"):
                self.send(
                    "🤖 <b>NeuralTrade AI v4.0</b>\n\n"
                    "📊 /status — баланс та угоди\n"
                    "📈 /stats — статистика\n"
                    "💰 /balance — поточний баланс\n"
                    "📋 /positions — всі відкриті позиції детально\n"
                    "📜 /history — останні 20 закритих угод\n"
                    "📊 /report — виписка по всіх угодах\n"
                    "⏸ /pause — зупинити торгівлю\n"
                    "▶️ /resume — продовжити\n"
                    "🔄 /mode_swap — demo/live\n"
                    "🚨 /panic_close — закрити всі угоди\n"
                    "📋 /positions — відкриті позиції детально\n"
                    "📁 /closed — останні 20 закритих\n"
                    "📊 /report — повна виписка по угодам\n"
                    "💾 /backup — файл БД\n"
                )

            # ── /status ───────────────────────────────────────────
            elif cmd == "/status":
                try:
                    stats = db.get_stats()
                    opens = db.get_open_trades()
                except Exception:
                    stats = {"total": 0, "winrate": 0, "net_pnl": 0, "fees": 0}
                    opens = []
                mode   = cfg.mode.upper()
                paused = "⏸ ПАУЗА" if self._paused else "▶️ Активний"
                try:
                    row = db.conn.execute(
                        "SELECT COALESCE(balance,?) FROM balance_history ORDER BY id DESC LIMIT 1",
                        (cfg.initial_demo_balance or 1000,)
                    ).fetchone()
                    bal = float(row[0]) if row else (cfg.initial_demo_balance or 1000)
                except Exception:
                    bal = cfg.initial_demo_balance or 1000
                msg = (
                    f"📊 <b>NeuralTrade {mode}</b> | {paused}\n"
                    f"━━━━━━━━━━━━━━\n"
                    f"💰 Баланс: <code>${bal:.2f}</code>\n"
                    f"📈 Закрито: {stats.get('total', 0)} | WR: {stats.get('winrate', 0)}%\n"
                    f"💹 Net P&L: <code>${stats.get('net_pnl', 0):+.2f}</code>\n"
                    f"📋 Відкрито: {len(opens)} угод\n"
                )
                if opens:
                    msg += "━━━━━━━━━━━━━━\n"
                    for t in opens:
                        tf    = t.get('timeframe','1h')
                        side  = t.get('side','')
                        pair  = t.get('pair','')
                        entry = float(t.get('entry_price',0))
                        amt   = float(t.get('amount_usd',0))
                        e2    = '🟢' if side == 'BUY' else '🔴'
                        msg  += f"{e2} {side} {pair} [{tf}] @ ${entry:,.2f} (${amt:.0f})\n"
                self.send(msg)

            # ── /balance ──────────────────────────────────────────
            elif cmd == "/balance":
                try:
                    stats   = db.get_stats()
                    opens   = db.get_open_trades()
                    initial = float(cfg.initial_demo_balance or 1000)
                    closed_net  = float(stats.get('net_pnl', 0))
                    open_frozen = sum(float(t.get('amount_usd',0)) for t in opens)
                    open_fees   = sum(float(t.get('fee_usd',0)) for t in opens)
                    # Правильний баланс = початковий + закриті P&L - заморожені
                    real_bal = initial + closed_net - open_frozen - open_fees
                    self.send(
                        f"💰 <b>Баланс: ${real_bal:.2f}</b>\n"
                        f"💵 Початковий: <code>${initial:.2f}</code>\n"
                        f"📈 P&L закритих: <code>${closed_net:+.2f}</code>\n"
                        f"🔒 Заморожено: <code>-${open_frozen:.2f}</code> ({len(opens)} угод)\n"
                        f"💸 Комісії (закр.): <code>-${stats.get('fees',0):.4f}</code>\n"
                        f"📊 Вільно: <code>${real_bal:.2f}</code>"
                    )
                except Exception as e:
                    self.send(f"❌ /balance: {e}")

            # ── /stats ────────────────────────────────────────────
            elif cmd == "/stats":
                self.daily_report(db.get_stats())

            # ── /pause ────────────────────────────────────────────
            elif cmd == "/pause":
                self._paused = True
                self.send("⏸ <b>Торгівля призупинена</b>\n/resume для відновлення")
                log.info("⏸ Торгівля призупинена через Telegram")

            # ── /resume ───────────────────────────────────────────
            elif cmd == "/resume":
                self._paused = False
                self.send("▶️ <b>Торгівля відновлена!</b>")
                log.info("▶️ Торгівля відновлена через Telegram")

            # ── /panic_close ──────────────────────────────────────
            elif cmd == "/panic_close":
                self._paused = True
                try:
                    opens = db.get_open_trades()
                    count = len(opens)
                    closed_count = 0
                    for t in opens:
                        try:
                            # close_trade(tid, exit_price, gross, fee, status)
                            entry = float(t.get("entry_price", 0))
                            amt   = float(t.get("amount_usd", 0))
                            fee   = float(t.get("fee_usd", 0))
                            # Закриваємо по ціні входу (нульовий P&L)
                            db.close_trade(
                                t["id"],
                                entry,   # exit_price = entry (нейтрально)
                                0.0,     # gross = 0
                                fee,     # fee
                                "PANIC"  # status
                            )
                            closed_count += 1
                            log.info(f"🚨 PANIC: закрито #{t['id']} {t.get('pair')}")
                        except Exception as e:
                            log.error(f"PANIC close #{t.get('id')}: {e}")
                    self.send(
                        f"🚨 <b>PANIC CLOSE виконано!</b>\n"
                        f"Закрито угод: {closed_count}/{count}\n"
                        f"Торгівля зупинена. /resume для відновлення"
                    )
                    log.critical(f"PANIC CLOSE: закрито {closed_count}/{count} угод")
                except Exception as e:
                    self.send(f"🚨 Panic close помилка: {e}")
                    log.error(f"panic_close: {e}", exc_info=True)

            # ── /mode_swap ────────────────────────────────────────
            elif cmd == "/mode_swap":
                new_mode = "live" if cfg.mode == "demo" else "demo"
                if new_mode == "live":
                    self.send("⚠️ Підтверди: /confirm_live")
                    return
                cfg._data["mode"] = new_mode
                self.send(f"🔄 Режим: <b>{new_mode.upper()}</b>")

            elif cmd == "/confirm_live":
                cfg._data["mode"] = "live"
                self.send("💰 <b>LIVE режим активовано!</b>")
                log.warning("LIVE режим через Telegram!")

            # ── /positions ────────────────────────────────────────
            elif cmd == "/positions":
                try:
                    opens = db.get_open_trades()
                    if not opens:
                        self.send("📭 Відкритих позицій немає\n/resume щоб відновити торгівлю")
                    else:
                        msg = f"📊 <b>Відкриті позиції: {len(opens)}</b>\n━━━━━━━━━━━━━━\n"
                        for t in opens:
                            entry  = float(t.get('entry_price', 0))
                            amount = float(t.get('amount_usd', 0))
                            sl     = float(t.get('sl_price') or 0)
                            tp     = float(t.get('tp_price') or 0)
                            fee    = float(t.get('fee_usd') or 0)
                            side   = t.get('side','')
                            pair   = t.get('pair','')
                            ts     = (t.get('ts_open','')[:16] or '—')
                            conf   = float(t.get('confidence') or 0)
                            reason = (t.get('reason') or '')[:60]
                            e2     = '🟢' if side == 'BUY' else '🔴'
                            tf2    = t.get('timeframe', '1h')
                            regime = t.get('regime', '—')
                            msg += (
                                f"\n{e2} <b>{side} {pair}</b> [{tf2}]\n"
                                f"  💵 Вхід: <code>${entry:,.2f}</code>\n"
                                f"  💰 Сума: <code>${amount:.2f}</code>\n"
                                f"  🛑 SL: <code>${sl:,.2f}</code>  ✅ TP: <code>${tp:,.2f}</code>\n"
                                f"  🎯 Conf: <code>{conf:.0%}</code> | Режим: <code>{regime}</code>\n"
                                f"  🕐 Відкрито: <code>{ts}</code>\n"
                                f"  💬 <i>{reason}</i>\n"
                            )
                        # Telegram max = 4096 символів → шлемо по 5 позицій
                        chunk_size = 5
                        chunks = [opens[i:i+chunk_size] for i in range(0, len(opens), chunk_size)]
                        for ci, chunk in enumerate(chunks):
                            chunk_msg = f"📊 <b>Позиції {ci*chunk_size+1}-{ci*chunk_size+len(chunk)} з {len(opens)}:</b>\n━━━━━━━━━━━━━━\n"
                            for t in chunk:
                                entry  = float(t.get('entry_price', 0))
                                amount = float(t.get('amount_usd', 0))
                                sl     = float(t.get('sl_price') or 0)
                                tp     = float(t.get('tp_price') or 0)
                                side   = t.get('side','')
                                pair   = t.get('pair','')
                                ts     = (t.get('ts_open','')[:16] or '—')
                                conf   = float(t.get('confidence') or 0)
                                reason = (t.get('reason') or '')[:50]
                                tf2    = t.get('timeframe','1h')
                                e2     = '🟢' if side == 'BUY' else '🔴'
                                chunk_msg += (
                                    f"\n{e2} <b>{side} {pair}</b> [{tf2}]\n"
                                    f"  💵 ${entry:,.2f} | 💰 ${amount:.0f}\n"
                                    f"  🛑 ${sl:,.2f}  ✅ ${tp:,.2f}\n"
                                    f"  🎯 {conf:.0%} | 🕐 {ts}\n"
                                    f"  <i>{reason}</i>\n"
                                )
                            self.send(chunk_msg)
                except Exception as e:
                    self.send(f"❌ /positions: {e}")

            # ── /closed — закриті позиції ────────────────────────
            elif cmd == "/closed":
                try:
                    rows = db.conn.execute(
                        "SELECT * FROM trades WHERE status NOT IN ('OPEN') ORDER BY id DESC LIMIT 20"
                    ).fetchall()
                    closed_list = [dict(r) for r in rows]
                    if not closed_list:
                        self.send("📭 Закритих позицій ще немає")
                    else:
                        msg = f"📋 <b>Останні {len(closed_list)} закритих угод:</b>\n━━━━━━━━━━━━━━\n"
                        for t in closed_list:
                            side   = t.get('side','')
                            pair   = t.get('pair','')
                            tf     = t.get('timeframe','1h')
                            entry  = float(t.get('entry_price') or 0)
                            exit_p = float(t.get('exit_price') or 0)
                            amt    = float(t.get('amount_usd') or 0)
                            net    = float(t.get('net_pnl') or 0)
                            status = t.get('status','')
                            ts     = (t.get('ts_open','')[:16] or '—')
                            pct    = (net/amt*100) if amt else 0
                            e2     = '✅' if net > 0 else '❌'
                            msg += (
                                f"\n{e2} <b>{side} {pair}</b> [{tf}] [{status}]\n"
                                f"  Вхід: <code>${entry:,.2f}</code> → Вихід: <code>${exit_p:,.2f}</code>\n"
                                f"  Сума: <code>${amt:.2f}</code>\n"
                                f"  P&L: <code>${net:+.2f}</code> ({pct:+.1f}%)\n"
                                f"  {ts}\n"
                            )
                        stat = db.get_stats()
                        msg += f"\n━━━━━━━━━━━━━━\n"
                        msg += f"💹 Net P&L всього: <code>${stat.get('net_pnl',0):+.2f}</code>\n"
                        msg += f"🏆 WR: {stat.get('winrate',0)}%"
                        self.send(msg)
                except Exception as e:
                    self.send(f"❌ /closed: {e}")

            # ── /history — закриті угоди ──────────────────────────
            elif cmd == "/history":
                try:
                    all_t  = db.get_trades(limit=500)
                    closed = [t for t in all_t if t.get('status','') not in ('','OPEN')][-20:]
                    if not closed:
                        self.send("📭 Закритих угод ще немає")
                    else:
                        msg = f"📜 <b>Останні {len(closed)} закритих:</b>\n━━━━━━━━━━━━━━\n"
                        for t in closed:
                            net    = float(t.get('net_pnl') or 0)
                            amount = float(t.get('amount_usd') or 0)
                            entry  = float(t.get('entry_price') or 0)
                            tf2    = t.get('timeframe','—')
                            status = t.get('status','—')
                            ts     = (t.get('ts_open','')[:16] or '—')
                            e2     = '✅' if net > 0 else '❌'
                            pct    = (net / amount * 100) if amount else 0
                            msg += (
                                f"{e2} {t.get('side','')} {t.get('pair','')} [{tf2}] [{status}]\n"
                                f"  Вхід: ${entry:,.2f} | Сума: ${amount:.0f}\n"
                                f"  P&L: ${net:+.2f} ({pct:+.1f}%) | {ts}\n"
                            )
                        self.send(msg)
                except Exception as e:
                    self.send(f"❌ /history: {e}")

            # ── /report — повна виписка ──────────────────────────────
            elif cmd == "/report":
                try:
                    stats   = db.get_stats()
                    all_t   = db.get_trades(limit=1000)
                    closed  = [t for t in all_t if t.get('status','') not in ('','OPEN')]
                    opens   = db.get_open_trades()
                    total_g = sum(float(t.get('gross_pnl') or 0) for t in closed)
                    total_f = sum(float(t.get('fee_usd') or 0) for t in closed)
                    wins    = [t for t in closed if float(t.get('net_pnl') or 0) > 0]
                    losses  = [t for t in closed if float(t.get('net_pnl') or 0) <= 0]
                    avg_w   = (sum(float(t.get('net_pnl',0)) for t in wins) / len(wins)) if wins else 0
                    avg_l   = (sum(float(t.get('net_pnl',0)) for t in losses) / len(losses)) if losses else 0
                    # По парах
                    pairs_stats = {}
                    for t in closed:
                        p  = t.get('pair','?')
                        if p not in pairs_stats:
                            pairs_stats[p] = {'n':0,'pnl':0.0,'w':0}
                        pairs_stats[p]['n']   += 1
                        pairs_stats[p]['pnl'] += float(t.get('net_pnl') or 0)
                        if float(t.get('net_pnl') or 0) > 0:
                            pairs_stats[p]['w'] += 1
                    # По таймфреймах
                    tf_stats = {}
                    for t in closed:
                        tf3 = t.get('timeframe','1h')
                        if tf3 not in tf_stats:
                            tf_stats[tf3] = {'n':0,'pnl':0.0}
                        tf_stats[tf3]['n']   += 1
                        tf_stats[tf3]['pnl'] += float(t.get('net_pnl') or 0)

                    ini_bal  = float(cfg.initial_demo_balance or 1000)
                    net_pnl  = float(stats.get('net_pnl', 0))
                    frozen   = sum(float(t.get('amount_usd',0)) + float(t.get('fee_usd',0)) for t in opens)
                    real_bal = ini_bal + net_pnl - frozen
                    msg = (
                        f"📊 <b>ВИПИСКА NeuralTrade DEMO</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"💰 Баланс: <code>${real_bal:.2f}</code>\n"
                        f"📈 Всього угод: {len(closed)} закрито | {len(opens)} відкрито\n"
                        f"🎯 WR: {stats.get('winrate',0):.1f}%\n"
                        f"━━━━━━━━━━━━━━\n"
                        f"P&L Gross: <code>${total_g:+.2f}</code>\n"
                        f"Комісії:   <code>-${total_f:.4f}</code>\n"
                        f"Net P&L:   <code>${total_g-total_f:+.2f}</code>\n"
                        f"Сер. профіт: <code>${avg_w:+.2f}</code>\n"
                        f"Сер. збиток: <code>${avg_l:+.2f}</code>\n"
                        f"━━━━━━━━━━━━━━\n"
                        f"<b>По парах:</b>\n"
                    )
                    for p, s in sorted(pairs_stats.items(), key=lambda x: -x[1]['pnl']):
                        wr2 = s['w']/s['n']*100 if s['n'] else 0
                        msg += f"  {p}: {s['n']} угод | P&L ${s['pnl']:+.2f} | WR {wr2:.0f}%\n"
                    msg += "━━━━━━━━━━━━━━\n<b>По таймфреймах:</b>\n"
                    for tf3, s in sorted(tf_stats.items()):
                        msg += f"  [{tf3}]: {s['n']} угод | P&L ${s['pnl']:+.2f}\n"
                    self.send(msg)
                except Exception as e:
                    self.send(f"❌ /report: {e}")

            # ── /backup ───────────────────────────────────────────
            elif cmd == "/backup":
                self.send_file(db.path, f"💾 Бекап БД: {db.path}")

            else:
                self.send(f"❓ Невідома команда: {cmd}\n/help — список команд")

        except Exception as e:
            log.error(f"_handle_command {cmd}: {e}", exc_info=True)
            self.send(f"❌ Помилка: {e}")


    def trade_opened(self, t: dict):
        side = t.get("side", "BUY")
        e = "🟢" if side == "BUY" else "🔴"
        mode_tag = "🧪" if t.get("mode") == "demo" else "💰"
        regime  = t.get("regime", "—")
        tf      = t.get("timeframe", "1h")
        reason  = t.get("reason", "")
        conf    = t.get("confidence", 0)
        sl      = t.get("sl_price") or t.get("sl") or 0
        tp      = t.get("tp_price") or t.get("tp") or 0
        amount  = t.get("amount_usd", 0)
        entry   = t.get("entry_price", 0)
        self.send(
            f"{e} <b>{side} {t['pair']}</b> {mode_tag}\n"
            f"━━━━━━━━━━━━\n"
            f"💵 Ціна входу: <code>${entry:,.2f}</code>\n"
            f"💰 Сума угоди: <code>${amount:.2f}</code>\n"
            f"📊 Таймфрейм: <code>{tf}</code>\n"
            f"🎯 Впевненість: <code>{conf:.0%}</code>\n"
            f"🛑 SL: <code>${sl:,.2f}</code> | ✅ TP: <code>${tp:,.2f}</code>\n"
            f"📈 Режим: <code>{regime}</code>\n"
            f"💬 <i>{reason[:80]}</i>"
        )

    def trade_closed(self, t: dict, net: float):
        e = "✅" if net > 0 else "❌"
        status   = t.get("status", "CLOSED")
        entry    = t.get("entry_price", 0)
        amount   = t.get("amount_usd", 0)
        fee      = t.get("fee_usd", 0)
        tf       = t.get("timeframe", "1h")
        pct      = (net / amount * 100) if amount else 0
        self.send(
            f"{e} <b>ЗАКРИТО {t.get('side','')} {t['pair']}</b> [{status}]\n"
            f"━━━━━━━━━━━━\n"
            f"💵 Ціна входу: <code>${entry:,.2f}</code>\n"
            f"💰 Сума: <code>${amount:.2f}</code> | TF: <code>{tf}</code>\n"
            f"{'✅' if net > 0 else '❌'} Net P&L: <code>${net:+.2f}</code> ({pct:+.1f}%)\n"
            f"💸 Комісія: <code>-${fee:.4f}</code>"
        )

    def ml_update(self, pair: str, acc: float, n: int):
        self.send(
            f"🧠 <b>ML оновлено: {pair}</b>\n"
            f"Точність: <code>{acc:.1%}</code> | "
            f"Зразків: <code>{n}</code>"
        )

    def daily_report(self, stats: dict):
        self.send(
            f"📊 <b>Щоденний звіт NeuralTrade</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"Угод: {stats['total']} | WR: {stats['winrate']}%\n"
            f"Net P&L: <b><code>${stats['net_pnl']:+.2f}</code></b>\n"
            f"Комісії: <code>-${stats['fees']:.4f}</code>\n"
            f"Краща: +${stats['best']:.2f} | "
            f"Гірша: ${stats['worst']:.2f}"
        )

    def daily_backup(self, db_path: str):
        """Щоденний авто-бекап БД."""
        if os.path.exists(db_path):
            ts = datetime.now().strftime("%Y-%m-%d")
            self.send_file(db_path, f"📦 Щоденний бекап {ts}")
            log.info(f"📤 Бекап надіслано: {ts}")
        # Міграція: додаємо нові колонки якщо відсутні
        for col_sql in [
            "ALTER TABLE trades ADD COLUMN timeframe TEXT DEFAULT '1h'",
        ]:
            try:
                self.conn.execute(col_sql)
                self.conn.commit()
            except Exception:
                pass  # Вже існує
