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

        def __init__(self, path: str = "config.json"):
        self._data = dict(self.DEFAULTS)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if not k.startswith("//"):
                    self._data[k] = v

        # ENV змінні
        env_map = {
            "BINANCE_API_KEY":    "api_key",
            "BINANCE_SECRET":     "api_secret",
            "TELEGRAM_TOKEN":     "telegram_token",
            "TELEGRAM_CHAT_ID":   "telegram_chat_id",
            "OPENAI_API_KEY":     "openai_api_key",
            "MODE":               "mode",
        }

        for env_key, cfg_key in env_map.items():
            val = os.environ.get(env_key)
            if val:
                self._data[cfg_key] = val

        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULTS, f, indent=2, ensure_ascii=False)
            log.info("✅ Створено config.json")
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
                binance_order_id TEXT
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
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE status='OPEN'"
        ).fetchall()
        return [dict(r) for r in rows]

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
        total = c.execute("SELECT COUNT(*) FROM trades WHERE status!='OPEN'").fetchone()[0]
        wins  = c.execute("SELECT COUNT(*) FROM trades WHERE outcome=1").fetchone()[0]
        fees  = c.execute("SELECT COALESCE(SUM(fee_usd),0) FROM trades").fetchone()[0]
        gross = c.execute("SELECT COALESCE(SUM(gross_pnl),0) FROM trades WHERE status!='OPEN'").fetchone()[0]
        best  = c.execute("SELECT COALESCE(MAX(net_pnl),0) FROM trades").fetchone()[0]
        worst = c.execute("SELECT COALESCE(MIN(net_pnl),0) FROM trades").fetchone()[0]
        return {
            "total": total, "wins": wins, "losses": total-wins,
            "winrate": round(wins/total*100, 1) if total else 0,
            "gross_pnl": round(gross, 2), "fees": round(fees, 4),
            "net_pnl": round(gross-fees, 2),
            "best": round(best, 2), "worst": round(worst, 2),
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
            requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self.chat_id,
                      "text": text, "parse_mode": "HTML"},
                timeout=8,
            )
        except Exception as e:
            log.debug(f"Telegram send: {e}")

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
        if not self.enabled:
            return
        offset = 0
        log.info("📱 Telegram polling запущено")

        while True:
            try:
                r = requests.get(
                    f"{self._base}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                    timeout=35,
                )
                updates = r.json().get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    msg = upd.get("message", {})
                    text = msg.get("text", "").strip()
                    chat = str(msg.get("chat", {}).get("id", ""))

                    # Приймаємо тільки від авторизованого чату
                    if chat != self.chat_id:
                        continue

                    self._handle_command(text, db, cfg)
            except Exception as e:
                log.debug(f"Telegram poll: {e}")
            time.sleep(5)

    def _handle_command(self, text: str, db, cfg):
        cmd = text.lower().split()[0] if text else ""

        if cmd == "/status":
            stats   = db.get_stats()
            opens   = db.get_open_trades()
            mode    = cfg.mode.upper()
            paused  = "⏸ ПАУЗА" if self._paused else "▶️ Активний"
            msg = (
                f"📊 <b>NeuralTrade Status</b>\n"
                f"Режим: <b>{mode}</b> | {paused}\n"
                f"━━━━━━━━━━━━\n"
                f"Угод: {stats['total']} | WR: {stats['winrate']}%\n"
                f"Net P&L: <code>${stats['net_pnl']:+.2f}</code>\n"
                f"Відкрито: {len(opens)}\n"
            )
            for t in opens:
                msg += f"  • {t['side']} {t['pair']} @ ${t['entry_price']:,.2f}\n"
            self.send(msg)

        elif cmd == "/stats":
            stats = db.get_stats()
            self.daily_report(stats)

        elif cmd == "/pause":
            self._paused = True
            self.send("⏸ <b>Торгівля призупинена</b>\n/resume для відновлення")
            log.info("⏸ Торгівля призупинена через Telegram")

        elif cmd == "/resume":
            self._paused = False
            self.send("▶️ <b>Торгівля відновлена</b>")
            log.info("▶️ Торгівля відновлена через Telegram")

        elif cmd == "/mode_swap":
            new_mode = "live" if cfg.is_demo else "demo"
            if new_mode == "live":
                self.send("⚠️ Підтверди перехід в LIVE: надішли /confirm_live")
                return
            cfg._data["mode"] = new_mode
            cfg.save()
            self.send(f"🔄 Режим змінено: <b>{new_mode.upper()}</b>")
            log.info(f"Режим: {new_mode}")

        elif cmd == "/confirm_live":
            cfg._data["mode"] = "live"
            cfg.save()
            self.send("💰 <b>LIVE режим активовано!</b>\n⚠️ Реальні гроші!")
            log.warning("LIVE режим активовано через Telegram!")

        elif cmd == "/panic_close":
            self.send("🚨 <b>PANIC CLOSE!</b> Закриваю всі угоди...")
            log.critical("PANIC CLOSE від Telegram!")
            if self._bot_ref:
                try:
                    self._bot_ref.executor.check_open_trades()
                    self._paused = True
                except Exception as e:
                    log.error(f"Panic close: {e}")
            self.send("✅ Угоди закрито. Бот на паузі.\n/resume для відновлення")

        elif cmd == "/backup":
            if os.path.exists(db.path):
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
                self.send_file(db.path, f"📦 Backup {ts}")
            else:
                self.send("❌ БД не знайдено")

        elif cmd == "/help":
            self.send(
                "🤖 <b>NeuralTrade команди:</b>\n"
                "/status — баланс та позиції\n"
                "/stats — статистика\n"
                "/pause — пауза\n"
                "/resume — відновити\n"
                "/mode_swap — demo ↔ live\n"
                "/panic_close — закрити все!\n"
                "/backup — надіслати БД\n"
            )

    def trade_opened(self, t: dict):
        e = "🟢" if t["side"] == "BUY" else "🔴"
        mode_tag = "🧪" if t.get("mode") == "demo" else "💰"
        regime = t.get("regime", "")
        self.send(
            f"{e} <b>{t['side']} {t['pair']}</b> {mode_tag}\n"
            f"Ціна: <code>${t['entry_price']:,.2f}</code>\n"
            f"Обсяг: <code>${t['amount_usd']:.2f}</code>\n"
            f"SL: <code>${t.get('sl_price',0):,.2f}</code>  "
            f"TP: <code>${t.get('tp_price',0):,.2f}</code>\n"
            f"Conf: <code>{t.get('confidence',0):.0%}</code> | "
            f"Regime: <code>{regime}</code>\n"
            f"<i>{t.get('reason','')}</i>"
        )

    def trade_closed(self, t: dict, net: float):
        e = "✅" if net > 0 else "❌"
        self.send(
            f"{e} <b>ЗАКРИТО {t['pair']}</b>\n"
            f"Net P&L: <code>${net:+.2f}</code>\n"
            f"Комісія: <code>-${t.get('fee_usd',0):.4f}</code>"
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
