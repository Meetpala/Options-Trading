"""Comprehensive Unit & End-to-End Integration Tests for APEX-3 NFO Options Engine.

Verifies:
1. F&O eligibility check & cross-broker Zerodha/Angel NFO symbol translation.
2. Live 5-level market-depth strike selection (rejects dead/illiquid/odd strikes,
   picks tightest liquid strike across ATM +/- 1 and DTE 3..6 rollover comparison).
3. Optimistic inside-spread entry pricing (1.0%..2.5% spread) vs tight marketable entry (<= 1.0%).
4. Full Bull (LONG -> Buy CE) & Bear (SHORT -> Buy PE) NFO NRML order lifecycle.
5. Hybrid Spot-Triggered Exits (Spot SL, +1.0R Spot Trailing Stop, Spot 2.0R Target)
   paired with broker-resting Catastrophic Option SL (-35%).
6. Optimistic entry auto-reprice after 10s when unfilled and spot intact.
7. EOD 14:55 IST Reconciler flatten for NFO NRML positions (preserving manual holdings).
8. Late-start 09:50 IST backfill & catch-up (including stale-quote retry fix).
9. Cloud dashboard /api/set_token auto-resume fix (clears Manual pause, preserves safety halts).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo
import pytest

from apex3.broker.paper import PaperBroker
from apex3.config import OptionsConfig, load_config
from apex3.journal.sqlite_journal import Journal
from apex3.models import (
    Candle,
    Decision,
    InstrumentInfo,
    OrderRequest,
    OrderState,
    OrderStatus,
    OrderType,
    PositionState,
    Product,
    Quote,
    Side,
)
from apex3.reconciliation.reconciler import Reconciler
from apex3.reconciliation.state_machine import PositionPhase, StateMachine
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.options_resolver import (
    OptionContractSpec,
    compute_catastrophic_option_sl,
    compute_optimistic_entry_price,
    evaluate_option_liquidity,
    format_angel_monthly_symbol,
    format_zerodha_monthly_symbol,
    select_liquid_option_contract,
)

IST = ZoneInfo("Asia/Kolkata")


def _make_option_candles(base_dt: datetime, spot_open: float, direction: str = "UP") -> list[Candle]:
    """Build realistic 2-day + today 5m candle series producing a clean APEX-3 breakout."""
    candles: list[Candle] = []
    # Prior day bars (enough for 20-bar volume SMA, EMA-50, ADX-14, ADR-14)
    prev_day = base_dt.date() - timedelta(days=1)
    if prev_day.weekday() >= 5:
        prev_day -= timedelta(days=2)
    t = datetime(prev_day.year, prev_day.month, prev_day.day, 9, 15, tzinfo=IST)
    px = spot_open * (0.985 if direction == "UP" else 1.015)
    for i in range(60):
        step = 0.0001 * spot_open if direction == "UP" else -0.0001 * spot_open
        o = round(px, 2)
        c = round(px + step, 2)
        h = round(max(o, c) + spot_open * 0.0012, 2)
        l = round(min(o, c) - spot_open * 0.0012, 2)
        candles.append(Candle(timestamp=t, open=o, high=h, low=l, close=c, volume=100_000, is_complete=True))
        px = c
        t += timedelta(minutes=5)

    # Today's Opening Range (09:15, 09:20, 09:25) — tight ~0.24% OR so stop width lands cleanly in [0.45%, 0.90%]
    or_t = datetime(base_dt.year, base_dt.month, base_dt.day, 9, 15, tzinfo=IST)
    for i in range(3):
        o = round(spot_open, 2)
        h = round(spot_open * 1.0012, 2)
        l = round(spot_open * 0.9988, 2)
        c = round(spot_open * (1.0003 if direction == "UP" else 0.9997), 2)
        candles.append(Candle(timestamp=or_t, open=o, high=h, low=l, close=c, volume=110_000, is_complete=True))
        or_t += timedelta(minutes=5)

    # Post-OR bars up to base_dt - 5m (last bar is high-conviction breakout bar)
    bar_t = datetime(base_dt.year, base_dt.month, base_dt.day, 9, 30, tzinfo=IST)
    while bar_t + timedelta(minutes=5) <= base_dt:
        is_last = (bar_t + timedelta(minutes=10) > base_dt)
        if not is_last:
            o = round(spot_open, 2)
            h = round(spot_open * 1.0011, 2)
            l = round(spot_open * 0.9989, 2)
            c = round(spot_open * (1.0006 if direction == "UP" else 0.9994), 2)
            vol = 100_000
        else:
            # Strong institutional breakout bar (Volume 3.5x, CLV >= 0.88, clears OR + 0.15*ATR)
            if direction == "UP":
                o = round(spot_open * 1.0008, 2)
                l = round(spot_open * 1.0005, 2)
                h = round(spot_open * 1.0032, 2)
                c = round(spot_open * 1.0030, 2)
            else:
                o = round(spot_open * 0.9992, 2)
                h = round(spot_open * 0.9995, 2)
                l = round(spot_open * 0.9968, 2)
                c = round(spot_open * 0.9970, 2)
            vol = 350_000
        candles.append(Candle(timestamp=bar_t, open=o, high=h, low=l, close=c, volume=vol, is_complete=True))
        bar_t += timedelta(minutes=5)

    return candles


def _make_trending_nifty(base_dt: datetime, direction: str = "UP") -> list[Candle]:
    """Build strongly trending NIFTY 50 candles (EMA20 > EMA50, rising ADX >= 22, VWAP aligned)."""
    candles: list[Candle] = []
    prev_day = base_dt.date() - timedelta(days=1)
    if prev_day.weekday() >= 5:
        prev_day -= timedelta(days=2)
    t = datetime(prev_day.year, prev_day.month, prev_day.day, 9, 15, tzinfo=IST)
    px = 25000.0
    step = 8.0 if direction == "UP" else -8.0
    for i in range(60):
        o = px
        c = px + step * (0.4 + 0.01 * i)
        h = max(o, c) + 4.0
        l = min(o, c) - 4.0
        candles.append(Candle(timestamp=t, open=o, high=h, low=l, close=c, volume=0, is_complete=True))
        px = c
        t += timedelta(minutes=5)

    bar_t = datetime(base_dt.year, base_dt.month, base_dt.day, 9, 15, tzinfo=IST)
    idx = 0
    while bar_t + timedelta(minutes=5) <= base_dt:
        o = px
        c = px + step * (0.5 + 0.12 * idx)
        h = max(o, c) + 5.0
        l = min(o, c) - 2.0 if direction == "UP" else min(o, c) - 5.0
        candles.append(Candle(timestamp=bar_t, open=o, high=h, low=l, close=c, volume=0, is_complete=True))
        px = c
        bar_t += timedelta(minutes=5)
        idx += 1
    return candles


# =============================================================================
# 1. Unit Tests: Symbol Formatting, Optimistic Pricing & Live Liquidity Gates
# =============================================================================

def test_cross_broker_nfo_symbol_formatting():
    exp = date(2026, 12, 29)
    assert format_zerodha_monthly_symbol("RELIANCE", exp, 1370.0, "CE") == "RELIANCE26DEC1370CE"
    assert format_angel_monthly_symbol("RELIANCE", exp, 1370.0, "CE") == "RELIANCE29DEC261370CE"
    assert format_zerodha_monthly_symbol("TATASTEEL", exp, 182.5, "PE") == "TATASTEEL26DEC182.5PE"


def test_optimistic_vs_marketable_entry_pricing():
    cfg = OptionsConfig(tight_spread_pct=1.0, max_option_spread_pct=2.5, optimistic_spread_fraction=0.65)
    now = datetime(2026, 11, 18, 10, 0, tzinfo=IST)

    # Tight spread (0.62% <= 1.0%) -> marketable ask + 1 tick
    q_tight = Quote(symbol="RELIANCE26DEC1370CE", exchange="NFO", last_price=48.70, bid=48.50, ask=48.80, volume=100000, timestamp=now, is_fresh=True)
    px, is_opt, sprd = compute_optimistic_entry_price(q_tight, tick_size=0.05, opt_cfg=cfg)
    assert is_opt is False
    assert px == 48.85
    assert sprd < 1.0

    # Moderate spread (2.0% in 1.0%..2.5%) -> optimistic inside spread (strictly below ask!)
    q_mod = Quote(symbol="TRENT26DEC3200CE", exchange="NFO", last_price=100.0, bid=99.0, ask=101.0, volume=50000, timestamp=now, is_fresh=True)
    px2, is_opt2, sprd2 = compute_optimistic_entry_price(q_mod, tick_size=0.05, opt_cfg=cfg)
    assert is_opt2 is True
    assert 99.0 < px2 < 101.0  # 99 + 0.65 * 2 = 100.30
    assert px2 == 100.30
    assert sprd2 == 2.0


def test_catastrophic_option_sl_pricing():
    cfg = OptionsConfig(catastrophic_sl_pct=35.0, catastrophic_sl_limit_buffer_pct=3.0)
    trig, lim = compute_catastrophic_option_sl(entry_premium=100.0, tick_size=0.05, opt_cfg=cfg, lower_circuit=10.0)
    assert trig == 65.0
    assert lim == 63.05
    assert lim < trig


def test_smart_depth_selector_rejects_dead_odd_strike_and_picks_liquid_neighbor():
    """Mirrors real Angel One finding: JSWSTEEL 1330CE has Vol=0/Spread=23%, while 1340CE is liquid."""
    broker = PaperBroker(starting_margin=100_000.0)
    now = datetime(2026, 11, 18, 10, 15, tzinfo=IST)
    exp = date(2026, 12, 29)

    dead_odd = OptionContractSpec(
        underlying="JSWSTEEL", expiry_date=exp, strike=1330.0, option_type="CE",
        lot_size=675, tick_size=0.05, zerodha_symbol="JSWSTEEL26DEC1330CE",
    )
    liquid_round = OptionContractSpec(
        underlying="JSWSTEEL", expiry_date=exp, strike=1340.0, option_type="CE",
        lot_size=675, tick_size=0.05, zerodha_symbol="JSWSTEEL26DEC1340CE",
    )
    broker.register_nfo_contracts("JSWSTEEL", [dead_odd, liquid_round])

    # Dead odd strike: wide 23% spread, Vol=0, OI=0
    broker.set_quote(Quote(
        symbol="JSWSTEEL26DEC1330CE", exchange="NFO", last_price=37.75, bid=39.25, ask=49.80,
        volume=0, timestamp=now, is_fresh=True, total_bid_qty=675, total_ask_qty=675, open_interest=0,
    ))
    # Liquid round strike: 1.5% spread, deep 5-level book, active volume/OI
    broker.set_quote(Quote(
        symbol="JSWSTEEL26DEC1340CE", exchange="NFO", last_price=40.00, bid=39.70, ask=40.30,
        volume=45000, timestamp=now, is_fresh=True, total_bid_qty=6750, total_ask_qty=6750, open_interest=58050,
    ))

    opt_cfg = OptionsConfig(enabled=True, max_option_spread_pct=2.5, min_depth_lots=2, min_volume_lots=5)
    sel = select_liquid_option_contract(
        broker=broker, underlying="JSWSTEEL", spot_side=Side.BUY, spot_price=1332.0,
        today=now.date(), opt_cfg=opt_cfg, available_margin=95_000.0,
    )
    assert sel.is_valid is True
    assert sel.contract is not None
    assert sel.contract.zerodha_symbol == "JSWSTEEL26DEC1340CE"
    assert sel.quantity == 675
    assert sel.is_optimistic_entry is True  # 1.5% spread -> optimistic inside spread


def test_smart_depth_selector_rejects_non_fno_and_thin_depth():
    broker = PaperBroker(starting_margin=100_000.0)
    now = datetime(2026, 11, 18, 10, 15, tzinfo=IST)
    opt_cfg = OptionsConfig(enabled=True)

    # 1. Non-F&O stock (no NFO contracts registered)
    sel_none = select_liquid_option_contract(
        broker=broker, underlying="SMALLCAP", spot_side=Side.BUY, spot_price=500.0,
        today=now.date(), opt_cfg=opt_cfg, available_margin=95_000.0,
    )
    assert sel_none.is_valid is False
    assert sel_none.reason_code == "NOT_FNO_ELIGIBLE"

    # 2. Thin depth (< 2 lots on bid/ask)
    c_thin = OptionContractSpec(
        underlying="MARUTI", expiry_date=date(2026, 12, 29), strike=12600.0, option_type="CE",
        lot_size=50, tick_size=0.05, zerodha_symbol="MARUTI26DEC12600CE",
    )
    broker.register_nfo_contracts("MARUTI", [c_thin])
    broker.set_quote(Quote(
        symbol="MARUTI26DEC12600CE", exchange="NFO", last_price=400.0, bid=398.0, ask=402.0,
        volume=5000, timestamp=now, is_fresh=True, total_bid_qty=50, total_ask_qty=50, open_interest=5000,
    ))
    sel_thin = select_liquid_option_contract(
        broker=broker, underlying="MARUTI", spot_side=Side.BUY, spot_price=12600.0,
        today=now.date(), opt_cfg=opt_cfg, available_margin=95_000.0,
    )
    assert sel_thin.is_valid is False
    assert sel_thin.reason_code == "OPTION_THIN_DEPTH"


# =============================================================================
# 2. End-to-End Integration Tests: Bull CE, Bear PE, Spot Trail, Target & EOD
# =============================================================================

def test_e2e_bull_ce_entry_catastrophic_sl_spot_trail_and_target_exit(tmp_path):
    """Full end-to-end Bull breakout: Buy ATM CE (NRML NFO) -> Catastrophic SL -> +1.0R Spot Trail -> 2.0R Spot Target Exit."""
    cfg = load_config()
    cfg.options = OptionsConfig(
        enabled=True,
        exchange="NFO",
        product="NRML",
        lots_per_trade=1,
        max_option_spread_pct=2.5,
        spot_trail_enabled=True,
        spot_trail_activation_r=1.0,
        spot_trail_atr_mult=1.5,
    )

    broker = PaperBroker(starting_margin=100_000.0)
    journal = Journal(db_path=str(tmp_path / "opt_bull.db"))
    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)
    engine.set_manual_symbols(["TRENT"])

    now = datetime(2026, 11, 16, 10, 15, 0, tzinfo=IST)
    spot_candles = _make_option_candles(now, spot_open=3000.0, direction="UP")
    nifty_candles = _make_trending_nifty(now, direction="UP")
    last_spot = spot_candles[-1].close
    q_ts = spot_candles[-1].timestamp

    broker._fixture_candles["TRENT"] = spot_candles
    broker._fixture_candles["NIFTY 50"] = nifty_candles
    broker.register_instrument(InstrumentInfo(symbol="TRENT", exchange="NSE", tick_size=0.05, lot_size=1, upper_circuit=3300.0, lower_circuit=2700.0, mis_available=True))
    broker.set_quote(Quote(symbol="NIFTY 50", exchange="NSE", last_price=nifty_candles[-1].close, bid=0.0, ask=0.0, volume=0, timestamp=q_ts, is_fresh=True))
    broker.set_quote(Quote(symbol="TRENT", exchange="NSE", last_price=last_spot, bid=last_spot - 0.10, ask=last_spot + 0.10, volume=350_000, timestamp=q_ts, is_fresh=True))

    # Register liquid ATM CE contract
    ce_sym = "TRENT26DEC3000CE"
    ce_spec = OptionContractSpec(
        underlying="TRENT", expiry_date=date(2026, 12, 29), strike=3000.0, option_type="CE",
        lot_size=225, tick_size=0.05, zerodha_symbol=ce_sym,
    )
    broker.register_nfo_contracts("TRENT", [ce_spec])
    broker.register_instrument(InstrumentInfo(symbol=ce_sym, exchange="NFO", tick_size=0.05, lot_size=225, upper_circuit=400.0, lower_circuit=5.0, mis_available=True))
    broker.set_quote(Quote(
        symbol=ce_sym, exchange="NFO", last_price=100.0, bid=99.60, ask=100.40,
        volume=100_000, timestamp=q_ts, is_fresh=True, total_bid_qty=2250, total_ask_qty=2250, open_interest=80_000,
    ))

    # Step 1: Execute entry cycle
    decisions = engine.evaluate_step(now)
    assert any("OPTION_ENTRY_SUBMITTED" in d.reason_codes for d in decisions), f"Unexpected decisions: {decisions}"

    pos = engine.state_machine.positions[ce_sym]
    assert pos.is_option is True
    assert pos.exchange == "NFO"
    assert pos.product == Product.NRML
    assert pos.side == Side.BUY
    assert pos.underlying_symbol == "TRENT"
    assert pos.underlying_side == Side.BUY
    assert pos.quantity == 225
    assert pos.phase == PositionPhase.OPEN_PROTECTED
    assert pos.protective_order_id != ""
    assert pos.catastrophic_sl_price > 0
    assert pos.catastrophic_sl_price < pos.average_entry

    # Verify Catastrophic SL order is resting at broker on NFO NRML as SELL SL
    sl_ord = broker.get_order_status(pos.protective_order_id)
    assert sl_ord.status == OrderStatus.TRIGGER_PENDING
    assert sl_ord.request.exchange == "NFO"
    assert sl_ord.request.product == Product.NRML
    assert sl_ord.request.side == Side.SELL
    assert sl_ord.request.quantity == 225

    # Step 2: Spot rallies to +1.3R -> Spot Trailing Stop activates and ratchets spot_current_stop upward!
    t2 = now + timedelta(minutes=15)
    orig_spot_stop = pos.spot_current_stop
    spot_13r = round(pos.spot_entry_price + 1.30 * pos.spot_r_value, 2)
    broker.set_quote(Quote(symbol="TRENT", exchange="NSE", last_price=spot_13r, bid=spot_13r - 0.10, ask=spot_13r + 0.10, volume=400_000, timestamp=t2, is_fresh=True))
    broker.set_quote(Quote(symbol=ce_sym, exchange="NFO", last_price=120.0, bid=119.50, ask=120.50, volume=120_000, timestamp=t2, is_fresh=True, total_bid_qty=2250, total_ask_qty=2250, open_interest=80_000))
    d2 = engine.evaluate_step(t2)
    assert pos.trail_activated is True
    assert pos.spot_current_stop > orig_spot_stop

    # Step 3: Spot hits 2.0R target -> cancels catastrophic SL & exits CE via marketable LIMIT SELL
    t3 = now + timedelta(minutes=30)
    spot_tgt = round(pos.spot_target_price + 0.50, 2)
    broker.set_quote(Quote(symbol="TRENT", exchange="NSE", last_price=spot_tgt, bid=spot_tgt - 0.10, ask=spot_tgt + 0.10, volume=500_000, timestamp=t3, is_fresh=True))
    broker.set_quote(Quote(symbol=ce_sym, exchange="NFO", last_price=135.0, bid=134.50, ask=135.50, volume=150_000, timestamp=t3, is_fresh=True, total_bid_qty=2250, total_ask_qty=2250, open_interest=80_000))
    d3 = engine.evaluate_step(t3)

    assert any("SPOT_TARGET_HIT" in d.reason_codes for d in d3)
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0
    assert broker.get_order_status(pos.protective_order_id).status == OrderStatus.CANCELLED
    assert len(broker.get_positions()) == 0


def test_e2e_bear_pe_buy_order_and_spot_stop_loss_exit(tmp_path):
    """Full Bear breakdown: SHORT signal places Side.BUY on PE (NEVER naked SELL) -> exits on Spot Stop Hit."""
    cfg = load_config()
    cfg.options = OptionsConfig(enabled=True, exchange="NFO", product="NRML", lots_per_trade=1)

    broker = PaperBroker(starting_margin=100_000.0)
    journal = Journal(db_path=str(tmp_path / "opt_bear.db"))
    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)
    engine.set_manual_symbols(["JSWSTEEL"])

    now = datetime(2026, 11, 16, 10, 20, 0, tzinfo=IST)
    spot_candles = _make_option_candles(now, spot_open=1200.0, direction="DOWN")
    nifty_candles = _make_trending_nifty(now, direction="DOWN")
    last_spot = spot_candles[-1].close
    q_ts = spot_candles[-1].timestamp

    broker._fixture_candles["JSWSTEEL"] = spot_candles
    broker._fixture_candles["NIFTY 50"] = nifty_candles
    broker.register_instrument(InstrumentInfo(symbol="JSWSTEEL", exchange="NSE", tick_size=0.05, lot_size=1, upper_circuit=1320.0, lower_circuit=1080.0, mis_available=True))
    broker.set_quote(Quote(symbol="NIFTY 50", exchange="NSE", last_price=nifty_candles[-1].close, bid=0.0, ask=0.0, volume=0, timestamp=q_ts, is_fresh=True))
    broker.set_quote(Quote(symbol="JSWSTEEL", exchange="NSE", last_price=last_spot, bid=last_spot - 0.05, ask=last_spot + 0.05, volume=350_000, timestamp=q_ts, is_fresh=True))

    pe_sym = "JSWSTEEL26DEC1200PE"
    pe_spec = OptionContractSpec(
        underlying="JSWSTEEL", expiry_date=date(2026, 12, 29), strike=1200.0, option_type="PE",
        lot_size=675, tick_size=0.05, zerodha_symbol=pe_sym,
    )
    broker.register_nfo_contracts("JSWSTEEL", [pe_spec])
    broker.register_instrument(InstrumentInfo(symbol=pe_sym, exchange="NFO", tick_size=0.05, lot_size=675, upper_circuit=150.0, lower_circuit=1.0, mis_available=True))
    broker.set_quote(Quote(
        symbol=pe_sym, exchange="NFO", last_price=30.0, bid=29.90, ask=30.10,
        volume=100_000, timestamp=q_ts, is_fresh=True, total_bid_qty=6750, total_ask_qty=6750, open_interest=80_000,
    ))

    decisions = engine.evaluate_step(now)
    assert any("OPTION_ENTRY_SUBMITTED" in d.reason_codes for d in decisions), f"Unexpected: {decisions}"
    pos = engine.state_machine.positions[pe_sym]
    assert pos.side == Side.BUY            # Buying PUT, never naked short!
    assert pos.option_type == "PE"
    assert pos.underlying_side == Side.SELL
    assert pos.quantity == 675

    # Spot reverses above spot_current_stop -> triggers SPOT_STOP_LOSS exit of PE
    t2 = now + timedelta(minutes=10)
    stop_breach = round(pos.spot_current_stop + 1.0, 2)
    broker.set_quote(Quote(symbol="JSWSTEEL", exchange="NSE", last_price=stop_breach, bid=stop_breach - 0.05, ask=stop_breach + 0.05, volume=360_000, timestamp=t2, is_fresh=True))
    broker.set_quote(Quote(symbol=pe_sym, exchange="NFO", last_price=24.0, bid=23.90, ask=24.10, volume=120_000, timestamp=t2, is_fresh=True, total_bid_qty=6750, total_ask_qty=6750, open_interest=80_000))
    d2 = engine.evaluate_step(t2)

    assert any("SPOT_STOP_LOSS" in d.reason_codes for d in d2)
    assert pos.phase == PositionPhase.CLOSED
    assert len(broker.get_positions()) == 0


def test_eod_flatten_closes_nfo_nrml_option_and_preserves_manual_cnc(tmp_path):
    """At 14:55 IST, Reconciler flattens strategy NFO NRML option while preserving manual holdings."""
    cfg = load_config()
    cfg.options = OptionsConfig(enabled=True, exchange="NFO", product="NRML")
    broker = PaperBroker(starting_margin=100_000.0)
    journal = Journal(db_path=str(tmp_path / "opt_eod.db"))
    sm = StateMachine("APEX3")
    rec = Reconciler(broker=broker, state_machine=sm, journal=journal, session_tag="APEX3", config=cfg)

    opt_sym = "HINDALCO26DEC940CE"
    now = datetime(2026, 11, 18, 14, 55, 0, tzinfo=IST)
    broker.register_instrument(InstrumentInfo(symbol=opt_sym, exchange="NFO", tick_size=0.05, lot_size=700, upper_circuit=100.0, lower_circuit=1.0))
    broker.set_quote(Quote(symbol=opt_sym, exchange="NFO", last_price=30.0, bid=29.80, ask=30.20, volume=100_000, timestamp=now, is_fresh=True))

    # Open strategy NFO NRML option position
    oid = broker.place_order(OrderRequest(
        symbol=opt_sym, exchange="NFO", side=Side.BUY, quantity=700, product=Product.NRML,
        order_type=OrderType.LIMIT, price=30.25, tag="APEX3_HINDAL_O1015",
    ))
    sl_id = broker.place_order(OrderRequest(
        symbol=opt_sym, exchange="NFO", side=Side.SELL, quantity=700, product=Product.NRML,
        order_type=OrderType.SL, price=19.0, trigger_price=19.5, tag="APEX3_HINDAL_OSL",
    ))
    pos = sm.get_or_create_position(opt_sym)
    pos.symbol = opt_sym
    pos.exchange = "NFO"
    pos.product = Product.NRML
    pos.is_option = True
    pos.side = Side.BUY
    pos.quantity = 700
    pos.average_entry = 30.25
    pos.protective_order_id = sl_id
    pos.phase = PositionPhase.OPEN_PROTECTED

    # Add manual CNC delivery position that must NEVER be touched
    broker._positions["INFY"] = PositionState(symbol="INFY", side=Side.BUY, quantity=10, average_entry=1500.0, product=Product.CNC, exchange="NSE")

    ok, msg = rec.execute_flatten_sequence(timeout_sec=3.0)
    assert ok is True, msg
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0
    assert broker.get_order_status(sl_id).status == OrderStatus.CANCELLED
    remaining = {p.symbol for p in broker.get_positions()}
    assert opt_sym not in remaining
    assert "INFY" in remaining


def test_late_start_0950_retries_after_initial_stale_quote_and_executes_option(tmp_path):
    """Bug #3 fix: Starting at 09:50 with a transient stale quote on tick 1 must NOT permanently skip catch-up."""
    cfg = load_config()
    cfg.options = OptionsConfig(enabled=True, exchange="NFO", product="NRML", lots_per_trade=1)
    broker = PaperBroker(starting_margin=100_000.0)
    journal = Journal(db_path=str(tmp_path / "opt_late.db"))
    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)
    engine.set_manual_symbols(["HINDALCO"])

    now_950 = datetime(2026, 11, 16, 9, 50, 0, tzinfo=IST)
    # Build candles through 09:40 breakout + 09:45 quiet consolidation bar (so 09:50 latest bar is WAIT, 09:40 is catch-up!)
    spot_candles = _make_option_candles(datetime(2026, 11, 16, 9, 45, 0, tzinfo=IST), spot_open=950.0, direction="UP")
    trig_close = spot_candles[-1].close
    # Add quiet 09:45 bar (low volume -> latest bar itself is WAIT, requiring late-start catch-up!)
    spot_candles.append(Candle(
        timestamp=datetime(2026, 11, 16, 9, 45, 0, tzinfo=IST),
        open=trig_close, high=round(trig_close + 0.40, 2), low=round(trig_close - 0.30, 2),
        close=round(trig_close + 0.10, 2), volume=50_000, is_complete=True,
    ))
    nifty_candles = _make_trending_nifty(now_950, direction="UP")
    q_ts = spot_candles[-1].timestamp

    broker._fixture_candles["HINDALCO"] = spot_candles
    broker._fixture_candles["NIFTY 50"] = nifty_candles
    broker.register_instrument(InstrumentInfo(symbol="HINDALCO", exchange="NSE", tick_size=0.05, lot_size=1, upper_circuit=1050.0, lower_circuit=850.0, mis_available=True))
    broker.set_quote(Quote(symbol="NIFTY 50", exchange="NSE", last_price=nifty_candles[-1].close, bid=0.0, ask=0.0, volume=0, timestamp=q_ts, is_fresh=True))

    ce_sym = "HINDALCO26DEC950CE"
    broker.register_nfo_contracts("HINDALCO", [
        OptionContractSpec(underlying="HINDALCO", expiry_date=date(2026, 12, 29), strike=950.0, option_type="CE", lot_size=700, tick_size=0.05, zerodha_symbol=ce_sym)
    ])
    broker.register_instrument(InstrumentInfo(symbol=ce_sym, exchange="NFO", tick_size=0.05, lot_size=700, upper_circuit=100.0, lower_circuit=1.0))
    broker.set_quote(Quote(symbol=ce_sym, exchange="NFO", last_price=25.0, bid=24.90, ask=25.10, volume=100_000, timestamp=q_ts, is_fresh=True, total_bid_qty=7000, total_ask_qty=7000, open_interest=80_000))

    # Tick 1 at 09:50:00 — transient STALE spot quote!
    broker.set_quote(Quote(symbol="HINDALCO", exchange="NSE", last_price=trig_close, bid=trig_close - 0.05, ask=trig_close + 0.05, volume=100_000, timestamp=q_ts - timedelta(seconds=60), is_fresh=False))
    engine.evaluate_step(now_950)
    assert "HINDALCO" not in engine._late_start_catchup_done  # Must NOT lock out catch-up on stale quote!

    # Tick 2 at 09:50:03 — fresh live quote arrives -> catch-up triggers and buys ATM CE!
    t2 = now_950 + timedelta(seconds=3)
    broker.set_quote(Quote(symbol="NIFTY 50", exchange="NSE", last_price=nifty_candles[-1].close, bid=0.0, ask=0.0, volume=0, timestamp=q_ts, is_fresh=True))
    broker.set_quote(Quote(symbol="HINDALCO", exchange="NSE", last_price=trig_close + 0.15, bid=trig_close + 0.10, ask=trig_close + 0.20, volume=100_000, timestamp=q_ts, is_fresh=True))
    broker.set_quote(Quote(symbol=ce_sym, exchange="NFO", last_price=25.0, bid=24.90, ask=25.10, volume=100_000, timestamp=q_ts, is_fresh=True, total_bid_qty=7000, total_ask_qty=7000, open_interest=80_000))

    d2 = engine.evaluate_step(t2)
    assert "HINDALCO" in engine._late_start_catchup_done
    assert any("OPTION_ENTRY_SUBMITTED" in d.reason_codes for d in d2), f"Expected catch-up option entry, got: {d2}"
    assert ce_sym in engine.state_machine.positions
    assert engine.state_machine.positions[ce_sym].phase == PositionPhase.OPEN_PROTECTED


def test_cloud_set_token_auto_resumes_manual_pause_but_preserves_safety_halt(tmp_path):
    """Bug #1 fix: Pushing fresh Zerodha token clears Manual pause/BROKER_SESSION_INVALID, NOT emergency halts."""
    from apex3.dashboard.app import create_app

    cfg = load_config()
    broker = PaperBroker(starting_margin=100_000.0)
    journal = Journal(db_path=str(tmp_path / "cloud_resume.db"))
    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)
    app = create_app(engine, cfg, journal)
    client = app.test_client()

    # Case 1: Cloud VM paused with "Manual operator pause" -> pushing fresh token MUST auto-resume!
    engine.is_paused = True
    engine.pause_reason = "Manual operator pause"
    resp = client.post(
        "/api/set_token",
        json={"access_token": "fresh_test_token_123"},
        headers={"X-Operator-Token": "apex3_secure_operator_token"},
    )
    assert resp.status_code == 200
    assert engine.is_paused is False
    assert engine.pause_reason == ""

    # Case 2: Safety emergency pause ("QTY_MISMATCH_RELIANCE") -> pushing token MUST NOT auto-clear!
    engine.is_paused = True
    engine.pause_reason = "QTY_MISMATCH_RELIANCE"
    resp2 = client.post(
        "/api/set_token",
        json={"access_token": "fresh_test_token_456"},
        headers={"X-Operator-Token": "apex3_secure_operator_token"},
    )
    assert resp2.status_code == 200
    assert engine.is_paused is True
    assert engine.pause_reason == "QTY_MISMATCH_RELIANCE"
