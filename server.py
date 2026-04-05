"""
server.py v2.0 — Апгрейди:
  ✅ /api/metrics — Sharpe, MaxDD, Expectancy по парах
  ✅ /api/regime  — поточний Market Regime
  ✅ WS: trailing_update, drawdown_alert, loss_streak_alert
  ✅ Telegram polling у фоновому потоці
  ✅ Щоденний бекап БД
"""

import asyncio, json, logging, os, threading, time, sys, secrets, hashlib
from datetime import datetime
from typing import Set, Optional
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import Config, Database, TelegramNotifier
from agents import (TechAnalystAgent, SentimentAgent,
                    RiskManagerAgent, ExecutorAgent, DebateAI)
from ml_engine import MLEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("neuraltrade.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("NeuralTrade.Server")

# ─── Глобальний стан ─────────────────────────────────────────
cfg      = Config()
db       = Database(cfg.db_path)
tg       = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
ml       = MLEngine(db, cfg)
risk     = RiskManagerAgent(cfg)
tech     = TechAnalystAgent(cfg)
sentiment= SentimentAgent(cfg)
debate   = DebateAI()
executor = ExecutorAgent(cfg, db, tg, risk)
tg.set_bot(type('Bot', (), {'executor': executor})())

_ws_clients: Set[WebSocket] = set()
_ws_lock     = asyncio.Lock()
_msg_queue   = asyncio.Queue(maxsize=300)

# Кеш поточних режимів по парах
_current_regimes: dict = {}

app = FastAPI(title="NeuralTrade AI v2", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════
#  🔐 БЕЗПЕКА — API Key Auth + Rate Limiting
# ════════════════════════════════════════════════════════════════

_API_KEY_FILE = "api_key.secret"

def _load_or_create_api_key() -> str:
    # Спочатку перевіряємо змінну середовища
    env_key = os.environ.get("NEURALTRADE_API_KEY")
    if env_key:
        return env_key
    # Потім файл
    if os.path.exists(_API_KEY_FILE):
        with open(_API_KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    # Генеруємо новий
    key = secrets.token_urlsafe(32)
    with open(_API_KEY_FILE, "w") as f:
        f.write(key)
    try:
        import os as _os
        _os.chmod(_API_KEY_FILE, 0o600)
    except Exception:
        pass
    log.info(f"🔑 Новий API ключ → {_API_KEY_FILE}")
    log.info(f"🔑 API KEY: {key}")
    return key

_SERVER_API_KEY = _load_or_create_api_key()

# Rate limiting: ip → список timestamp
_rate_data: dict = defaultdict(list)
_RATE_LIMIT = 60   # макс. запитів
_RATE_WINDOW = 60  # за 60 секунд

def _check_rate_limit(ip: str) -> bool:
    """True = дозволено, False = перевищено ліміт."""
    now = time.time()
    _rate_data[ip] = [t for t in _rate_data[ip] if now - t < _RATE_WINDOW]
    if len(_rate_data[ip]) >= _RATE_LIMIT:
        return False
    _rate_data[ip].append(now)
    return True

async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
):
    """
    Перевірка API ключа + Rate Limiting.
    Ключ передається в заголовку: X-API-Key: <ключ>
    """
    ip = request.client.host if request.client else "unknown"

    # Rate limiting
    if not _check_rate_limit(ip):
        log.warning(f"Rate limit перевищено: {ip}")
        raise HTTPException(429, "Too Many Requests — зачекай хвилину")

    # API Key перевірка
    if not x_api_key:
        raise HTTPException(401, "Потрібен X-API-Key заголовок")

    # Constant-time порівняння (захист від timing attack)
    if not secrets.compare_digest(x_api_key.strip(), _SERVER_API_KEY):
        log.warning(f"Невірний API ключ від {ip}")
        raise HTTPException(403, "Невірний API ключ")

    return True

# Публічні endpoints (без auth): тільки /health
# Всі інші — захищені через Depends(verify_api_key)


# ════════════════════════════════════════════════════════════════
#  REST API
# ════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Публічний — перевірка чи сервер живий."""
    return {
        "status":  "ok",
        "version": "2.0",
        "mode":    cfg.mode,
        "pairs":   cfg.pairs,
        "paused":  tg.is_paused,
        "ts":      datetime.now().isoformat(),
    }


@app.get("/api/key-info")
def key_info():
    """Підказка де знайти ключ (без самого ключа)."""
    return {"message": f"API ключ знаходиться у файлі '{_API_KEY_FILE}'",
            "hint": "Передавай у заголовку: X-API-Key: <ключ>"}


@app.get("/api/stats")
def get_stats(_auth=Depends(verify_api_key)):
    return {**db.get_stats(), "balance": executor.get_balance(),
            "mode": cfg.mode, "paused": tg.is_paused}


@app.get("/api/trades")
def get_trades(limit: int = 100, pair: str = None, _auth=Depends(verify_api_key)):
    return db.get_trades(limit=limit, pair=pair)


@app.get("/api/metrics")
def get_metrics(_auth=Depends(verify_api_key)):
    """Sharpe, MaxDD, Expectancy для кожної пари."""
    result = {}
    for pair in cfg.pairs:
        result[pair] = ml.get_metrics(pair)
    return result


@app.get("/api/regime")
def get_regime(_auth=Depends(verify_api_key)):
    """Поточний Market Regime по парах."""
    return _current_regimes


@app.get("/api/signals")
def get_signals(_auth=Depends(verify_api_key)):
    rows = db.conn.execute(
        "SELECT * FROM signals ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/config")
def get_config(_auth=Depends(verify_api_key)):
    return cfg.to_dict()


class ConfigUpdate(BaseModel):
    data: dict

@app.post("/api/config")
def update_config(body: ConfigUpdate, _auth=Depends(verify_api_key)):
    for k, v in body.data.items():
        if k not in ("api_key","api_secret"):
            cfg._data[k] = v
    cfg.save()
    _queue_msg({"type": "config_updated", "config": cfg.to_dict()})
    return {"ok": True}


class ModeUpdate(BaseModel):
    mode: str

@app.post("/api/mode")
def set_mode(body: ModeUpdate, _auth=Depends(verify_api_key)):
    if body.mode not in ("demo","live"):
        raise HTTPException(400, "mode must be demo or live")
    cfg._data["mode"] = body.mode
    cfg.save()
    _queue_msg({"type": "mode_changed", "mode": body.mode})
    return {"ok": True, "mode": body.mode}


@app.post("/api/retrain")
def trigger_retrain(_auth=Depends(verify_api_key)):
    def _do():
        for pair in cfg.pairs:
            r = ml.train(pair)
            if r:
                acc, n = r
                tg.ml_update(pair, acc, n)
                _queue_msg({"type": "ml_retrained",
                            "pair": pair, "accuracy": acc,
                            "n_samples": n,
                            "metrics": ml.get_metrics(pair)})
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}


@app.post("/api/pause")
def pause_bot(_auth=Depends(verify_api_key)):
    tg._paused = True
    _queue_msg({"type": "bot_paused"})
    return {"ok": True}


@app.post("/api/resume")
def resume_bot(_auth=Depends(verify_api_key)):
    tg._paused = False
    _queue_msg({"type": "bot_resumed"})
    return {"ok": True}


@app.get("/api/export")
def export_csv(_auth=Depends(verify_api_key)):
    path = db.export_csv("trades_export.csv")
    if path and os.path.exists(path):
        return FileResponse(
            path, media_type="text/csv",
            filename=f"neuraltrade_{datetime.now().strftime('%Y-%m-%d')}.csv"
        )
    raise HTTPException(500, "Помилка експорту")


@app.get("/api/balance_history")
def balance_history(_auth=Depends(verify_api_key)):
    rows = db.conn.execute(
        "SELECT * FROM balance_history ORDER BY id DESC LIMIT 200"
    ).fetchall()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async with _ws_lock:
        _ws_clients.add(ws)
    log.info(f"WS підключено | клієнтів: {len(_ws_clients)}")

    stats = db.get_stats()
    await ws.send_json({
        "type":    "init",
        "stats":   stats,
        "balance": executor.get_balance(),
        "mode":    cfg.mode,
        "paused":  tg.is_paused,
        "config":  cfg.to_dict(),
        "trades":  db.get_trades(limit=50),
        "regimes": _current_regimes,
        "metrics": {p: ml.get_metrics(p) for p in cfg.pairs},
        "ml_stats": {
            "n_samples": stats.get("total", 0),
            "accuracy":  stats.get("accuracy", None),
        },
        "open_count": len(db.get_open_trades()),
    })

    try:
        while True:
            await asyncio.sleep(2)
            await ws.send_json({"type":"ping","ts":datetime.now().isoformat()})
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(ws)


def _queue_msg(msg: dict):
    try:
        _msg_queue.put_nowait({**msg, "ts": datetime.now().isoformat()})
    except asyncio.QueueFull:
        pass


async def _broadcaster():
    while True:
        try:
            msg = await asyncio.wait_for(_msg_queue.get(), timeout=1.0)
            if _ws_clients:
                dead = set()
                for ws in list(_ws_clients):
                    try:
                        await ws.send_json(msg)
                    except Exception:
                        dead.add(ws)
                async with _ws_lock:
                    _ws_clients -= dead
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            log.debug(f"Broadcaster: {e}")


# ════════════════════════════════════════════════════════════════
#  ТОРГОВИЙ ЦИКЛ
# ════════════════════════════════════════════════════════════════

def run_trading_cycle():
    if tg.is_paused:
        _queue_msg({"type":"agent_log","level":"warn",
                    "message":"⏸ Торгівля на паузі (Telegram /pause)"})
        return

    _queue_msg({"type":"cycle_start",
                "message": f"Цикл {datetime.now().strftime('%H:%M:%S')}"})

    for pair in cfg.pairs:
        try:
            _run_pair(pair)
        except Exception as e:
            log.error(f"[{pair}]: {e}", exc_info=True)
            _queue_msg({"type":"agent_log","level":"error",
                        "message":f"Помилка [{pair}]: {e}"})

    executor.check_open_trades()
    _stats = db.get_stats()
    _open  = db.get_open_trades()
    _queue_msg({
        "type":       "stats_update",
        **_stats,
        "balance":    executor.get_balance(),
        "open_trades": _open,
        "open_count":  len(_open),
        "ml_samples":  db.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
        "paused":     tg.is_paused,
    })

    # Перевірка drawdown
    stats  = db.get_stats()
    bal    = executor.get_balance()
    if risk._peak_balance and bal > 0:
        dd = (risk._peak_balance - bal) / risk._peak_balance
        if dd > (cfg.max_drawdown_pct or 0.10):
            _queue_msg({"type": "drawdown_alert",
                        "drawdown": dd, "balance": bal})


# ════════════════════════════════════════════════════════════════
#  ⚡ ШВИДКИЙ ЦИКЛ — кожні 30 сек (ціни + SL/TP + агент-статус)
# ════════════════════════════════════════════════════════════════

_last_prices: dict = {}       # пара → остання ціна
_price_move_threshold = 0.004  # 0.4% руху → повний аналіз

def run_fast_cycle():
    """
    Легкий цикл кожні 30 сек:
    - Оновлює ціни через Bybit/CoinGecko
    - Транслює prices на дашборд
    - Перевіряє SL/TP відкритих угод
    - Якщо ціна рухається > 0.4% → запускає повний аналіз
    """
    if tg.is_paused:
        return

    prices = {}
    trigger_full = False

    for pair in cfg.pairs:
        try:
            price = tech.get_current_price(pair)
            if price and price > 0:
                prices[pair] = round(price, 4)

                # Перевіряємо чи ціна рухається суттєво
                prev = _last_prices.get(pair, price)
                move = abs(price - prev) / prev if prev > 0 else 0
                if move > _price_move_threshold:
                    trigger_full = True
                    log.info(f"⚡ {pair}: рух ціни {move*100:.2f}% → повний аналіз")
                _last_prices[pair] = price
        except Exception:
            pass

    if prices:
        _queue_msg({"type": "prices", "prices": prices})

    # Перевірка SL/TP кожні 30 сек
    try:
        executor.check_open_trades()
        open_t = db.get_open_trades()
        if open_t:
            _queue_msg({
                "type": "stats_update",
                **db.get_stats(),
                "balance": executor.get_balance(),
                "open_trades": open_t,
                "paused": tg.is_paused,
            })
    except Exception as e:
        log.debug(f"fast_cycle check_trades: {e}")

    # Якщо великий рух — запускаємо повний аналіз
    if trigger_full:
        try:
            run_trading_cycle()
        except Exception as e:
            log.error(f"trigger_full error: {e}")


def _run_pair(pair: str):
    _queue_msg({"type":"agent_log","level":"info",
                "message":f"🔍 Аналіз {pair}..."})

    # ── Tech ─────────────────────────────────────────────────
    tech_s = tech.analyze(pair)
    if not tech_s:
        return

    regime = tech_s.get("regime", "sideways")
    _current_regimes[pair] = {
        "regime": regime,
        "adx": round(tech_s.get("adx", 0), 1),
        "atr_pct": round(tech_s.get("atr", 0) / tech_s.get("price", 1) * 100, 3),
        "volatility_spike": tech_s.get("volatility_spike", False),
    }
    _queue_msg({
        "type":"tech_result","pair":pair,
        "price": round(tech_s["price"], 2),
        "signal": tech_s["signal"],
        "rsi":    round(tech_s["rsi"], 1),
        "macd":   round(tech_s["macd"], 2),
        "regime": regime,
        "adx":    round(tech_s.get("adx",0), 1),
        "volatility_spike": tech_s.get("volatility_spike", False),
    })

    # Якщо volatility spike — стоп
    if tech_s.get("volatility_spike"):
        _queue_msg({"type":"agent_log","level":"warn",
                    "message":f"⚠️ {pair}: Volatility spike {tech_s.get('candle_vs_atr',0):.1f}×ATR → PAUSE"})
        return

    # ── Sentiment ────────────────────────────────────────────
    sent_s = sentiment.analyze(pair, tech_s["price"])
    _queue_msg({
        "type":"sentiment_result","pair":pair,
        "score":       round(sent_s["sent_score"], 3),
        "fear_greed":  sent_s["fear_greed"],
        "funding":     round(sent_s["funding_rate"], 5),
    })

    # ── ML ───────────────────────────────────────────────────
    features  = {**tech_s, **sent_s}
    ml_prob   = ml.predict(pair, features)
    _queue_msg({"type":"ml_result","pair":pair,
                "win_prob": round(ml_prob, 3)})

    # ── Debate ───────────────────────────────────────────────
    decision = debate.vote(tech_s, sent_s, ml_prob, cfg)
    _queue_msg({
        "type":"debate_result","pair":pair,
        "action":      decision["action"],
        "confidence":  round(decision["confidence"], 3),
        "bull_votes":  decision["bull_votes"],
        "bear_votes":  decision["bear_votes"],
        "reason":      decision["reason"],
        "regime":      regime,
        "tech_score":  decision.get("tech_score",0),
        "sent_score_w":decision.get("sent_score_w",0),
        "ml_score_w":  decision.get("ml_score_w",0),
    })

    db.save_signal({
        "pair": pair, "action": decision["action"],
        "confidence": decision["confidence"],
        "price": tech_s["price"],
        "reason": decision["reason"],
        "bull_votes": decision["bull_votes"],
        "bear_votes": decision["bear_votes"],
        "regime": regime,
        "mode": cfg.mode,
    })

    # ── Risk ─────────────────────────────────────────────────
    open_trades = db.get_open_trades()
    balance     = executor.get_balance()
    risk_ok     = risk.check(decision, open_trades, pair, balance, tech_s)
    _queue_msg({
        "type":"risk_result","pair":pair,
        "allowed": risk_ok["allowed"],
        "reason":  risk_ok["reason"],
        "loss_streak": risk._loss_streak,
        "peak_balance": risk._peak_balance,
    })

    if not risk_ok["allowed"]:
        return

    # ── Execute ──────────────────────────────────────────────
    atr    = tech_s.get("atr", tech_s["price"] * 0.012)
    amount = risk.position_size(
        balance  = balance,
        atr      = atr,
        price    = tech_s["price"],
        ml_prob  = ml_prob,
        winrate  = db.get_winrate(),
    )
    sl = risk.atr_stop_loss(tech_s["price"], decision["action"], atr)
    tp = risk.atr_take_profit(tech_s["price"], decision["action"], atr)

    trade_id = executor.open_trade(
        pair       = pair,
        side       = decision["action"],
        price      = tech_s["price"],
        amount_usd = amount,
        sl         = sl, tp = tp,
        features   = {**features, "ml_prob": ml_prob},
        confidence = decision["confidence"],
        reason     = decision["reason"],
    )

    if trade_id:
        _queue_msg({
            "type":  "trade_opened",
            "trade_id": trade_id,
            "pair":  pair,
            "side":  decision["action"],
            "price": round(tech_s["price"], 2),
            "amount": amount,
            "sl": sl, "tp": tp,
            "atr": round(atr, 2),
            "trailing_enabled": cfg.trailing_stop_enabled,
            "mode":  cfg.mode,
        })


# ════════════════════════════════════════════════════════════════
#  ФОНОВІ ЗАДАЧІ
# ════════════════════════════════════════════════════════════════

async def _price_ticker():
    while True:
        try:
            prices = {}
            for pair in cfg.pairs:
                p = tech.get_current_price(pair)
                if p:
                    prices[pair] = round(p, 2)
            _queue_msg({"type": "prices", "prices": prices})
        except Exception:
            pass
        await asyncio.sleep(3)


async def _agent_ticker():
    import random
    STATUS = {
        "tech":    ["Аналізую RSI...","MA перетин!","Regime: {regime}","BB squeeze","MACD hist+","ATR calc...","Volatility check"],
        "sent":    ["Fear&Greed: {fg}","Funding rate...","LLM кеш 1h","Позитивний сент.","Нейтральний","Whale alert!"],
        "risk":    ["Моніторинг","SL оновлено","Drawdown OK","Kelly sizing","Ліміт OK","Loss streak: {streak}"],
        "exec":    ["Очікування","Ордер розміщено","Trailing active","TP досягнуто","SL спрацював","Retry 2/3..."],
        "debate":  ["Bull vs Bear...","Bull перемагає!","Bear!","Консенсус HOLD","Голосую {b}B/{be}Be","Режим {regime}"],
    }
    fg_val, streak = 50, 0
    while True:
        try:
            fg_val = sentiment._fg_val or 50
            streak = risk._loss_streak
            # Беремо режим першої пари
            first_pair = cfg.pairs[0] if cfg.pairs else "BTC/USDT"
            regime = _current_regimes.get(first_pair, {}).get("regime","sideways")

            agents = {}
            for name, msgs in STATUS.items():
                msg = random.choice(msgs)
                msg = msg.replace("{fg}", str(int(fg_val)))
                msg = msg.replace("{regime}", regime)
                msg = msg.replace("{streak}", str(streak))
                bull = random.randint(2,4)
                msg = msg.replace("{b}", str(bull)).replace("{be}", str(5-bull))
                st = random.choice(["thinking","active","idle"])
                agents[name] = {"status": msg, "state": st}
            _queue_msg({"type":"agent_status","agents":agents,
                        "regime": regime, "loss_streak": streak})
        except Exception:
            pass
        await asyncio.sleep(8)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_broadcaster())
    asyncio.create_task(_price_ticker())
    asyncio.create_task(_agent_ticker())
    asyncio.create_task(_self_ping())


async def _self_ping():
    """
    Самопінг кожні 5 хв — запобігає засинанню Railway.
    Railway засинає сервіс після ~10 хв без HTTP запитів.
    """
    import aiohttp
    await asyncio.sleep(60)  # Чекаємо старту
    port = int(os.environ.get("PORT", 8000))
    url  = f"http://localhost:{port}/health"
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        log.debug("🏓 Self-ping OK")
        except Exception:
            pass
        await asyncio.sleep(300)  # Кожні 5 хвилин

    # Завантаження ML
    for pair in cfg.pairs:
        blob = db.load_model(pair)
        if blob:
            ml.load(pair, blob)

    log.info("🚀 FastAPI v2 запущено")


def _scheduler():
    import schedule, traceback

    log.info("🕐 Scheduler thread запущено")

    interval = cfg.cycle_interval_minutes or 5

    # Повний аналіз (RSI/MACD/ML/Debate) — кожні N хвилин
    schedule.every(interval).minutes.do(run_trading_cycle)

    # ⚡ Швидкий цикл — ціни + SL/TP кожні 30 секунд
    schedule.every(30).seconds.do(run_fast_cycle)

    # ML навчання кожні 24 год
    schedule.every(cfg.retrain_every_hours or 24).hours.do(
        lambda: [ml.train(p) for p in cfg.pairs]
    )
    schedule.every().day.at("09:00").do(
        lambda: tg.daily_report(db.get_stats())
    )
    schedule.every(10).minutes.do(
        lambda: db.save_balance(executor.get_balance(), cfg.mode)
    )
    if cfg.telegram_backup_db:
        schedule.every().day.at("03:00").do(
            lambda: tg.daily_backup(db.path)
        )

    # Перший запуск через 3 сек
    time.sleep(3)
    log.info("🚀 Перший цикл запускається...")
    try:
        run_trading_cycle()
    except Exception as e:
        log.error(f"❌ Цикл ПОМИЛКА: {e}")
        log.error(traceback.format_exc())

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            log.error(f"Scheduler error: {e}")
        time.sleep(5)


if __name__ == "__main__":
    # Telegram polling
    if tg.enabled:
        t_tg = threading.Thread(
            target=tg.poll_commands, args=(db, cfg), daemon=True
        )
        t_tg.start()
        log.info("📱 Telegram polling запущено")

    # Trading scheduler
    t_sched = threading.Thread(target=_scheduler, daemon=True)
    t_sched.start()
    log.info(f"▶️  Scheduler запущено | mode={cfg.mode} | pairs={cfg.pairs}")

    port = int(os.environ.get("PORT", cfg.server_port or 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    log_level="warning",
)
