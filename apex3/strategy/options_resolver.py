"""APEX-3 NSE F&O Contract Resolver & Live Market-Depth Liquidity Selector.

Solves real-market NSE Stock Option execution challenges:
1. F&O Eligibility Verification: Confirms active NFO OPTSTK contracts exist for the stock.
2. Cross-Broker Symbol Translation: Maps identical contracts cleanly between
   Zerodha Kite NFO format (e.g. RELIANCE26DEC1370CE) and Angel One NFO format
   (e.g. RELIANCE29DEC261370CE) by (underlying, expiry_date, strike, option_type).
3. Smart Live Market-Depth Strike Selection: Inspects nearest ATM (+/-1) strikes across
   current and next monthly expiries (during rollover DTE 3..6), fetches live 5-level
   FULL market depth, and rejects dead/illiquid/odd strikes with wide spreads.
4. Optimistic Limit Entry Pricing: Uses marketable ask+1 tick when spread is tight
   (<= 1.0%), or optimistic inside-spread limit (bid + 0.65 * spread) when spread is
   moderate (1.0%..2.5%), preventing slippage burn on wide stock option asks.
5. Catastrophic Option-Premium SL Pricing: Computes a safe broker-resting SL-Limit
   order on the option premium as a crash/network-outage safety net while the engine
   tracks exact underlying spot SL, trailing stop, fee-lock, and 2.0R target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from apex3.config import OptionsConfig
from apex3.models import Quote, Side
from apex3.strategy.trail import ceil_to_tick, floor_to_tick, round_to_tick

logger = logging.getLogger("apex3.options_resolver")


@dataclass(frozen=True)
class OptionContractSpec:
    """Canonical specification of an NSE stock option contract across brokers."""
    underlying: str
    expiry_date: date
    strike: float
    option_type: str              # "CE" or "PE"
    lot_size: int
    tick_size: float = 0.05
    zerodha_symbol: str = ""      # e.g., RELIANCE26DEC1370CE
    angel_symbol: str = ""        # e.g., RELIANCE29DEC261370CE
    angel_token: str = ""
    kite_token: str = ""
    dte: int = 0


@dataclass(frozen=True)
class OptionSelectionResult:
    """Result of live liquidity-aware option contract selection."""
    is_valid: bool
    contract: Optional[OptionContractSpec] = None
    quote: Optional[Quote] = None
    entry_limit_price: float = 0.0
    is_optimistic_entry: bool = False
    spread_pct: float = 999.0
    capital_required: float = 0.0
    quantity: int = 0
    reason_code: str = ""
    explanation: str = ""


def format_zerodha_monthly_symbol(
    underlying: str,
    expiry_date: date,
    strike: float,
    option_type: str,
) -> str:
    """Construct standard Zerodha Kite monthly NFO option tradingsymbol (e.g. RELIANCE26DEC1370CE)."""
    strike_str = str(int(strike)) if abs(strike - round(strike)) < 1e-6 else f"{strike:g}"
    yy = expiry_date.strftime("%y")
    mmm = expiry_date.strftime("%b").upper()
    return f"{underlying.upper()}{yy}{mmm}{strike_str}{option_type.upper()}"


def format_angel_monthly_symbol(
    underlying: str,
    expiry_date: date,
    strike: float,
    option_type: str,
) -> str:
    """Construct standard Angel One NFO option tradingsymbol (e.g. RELIANCE29DEC261370CE)."""
    strike_str = str(int(strike)) if abs(strike - round(strike)) < 1e-6 else f"{strike:g}"
    ddmmmyy = expiry_date.strftime("%d%b%y").upper()
    return f"{underlying.upper()}{ddmmmyy}{strike_str}{option_type.upper()}"


def compute_optimistic_entry_price(
    quote: Quote,
    tick_size: float,
    opt_cfg: OptionsConfig,
    upper_circuit: float = 0.0,
) -> tuple[float, bool, float]:
    """Compute optimistic bounded LIMIT BUY price from live bid-ask depth.

    - Tight spread (<= tight_spread_pct, default 1.0%):
        Marketable limit at `ask + 1 tick` for immediate fill.
    - Moderate spread (tight_spread_pct < spread <= max_option_spread_pct):
        Optimistic inside-spread limit at `bid + optimistic_spread_fraction * (ask - bid)`
        (default 65% across the spread) to capture better fills without chasing wide asks.

    Returns:
        (entry_limit_price, is_optimistic, spread_pct)
    """
    t = tick_size if tick_size > 0 else 0.05
    bid = float(quote.bid or 0.0)
    ask = float(quote.ask or 0.0)
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return 0.0, False, 999.0

    mid = (bid + ask) / 2.0
    spread_pct = ((ask - bid) / mid) * 100.0 if mid > 0 else 999.0

    if spread_pct <= opt_cfg.tight_spread_pct:
        # Tight book: cross the ask by 1 tick for instant fill
        raw_px = ask + t
        is_optimistic = False
    else:
        # Wider book: sit optimistically inside the spread instead of paying full ask
        frac = min(0.95, max(0.50, opt_cfg.optimistic_spread_fraction))
        raw_px = bid + (ask - bid) * frac
        # Ensure we stay strictly below ask if spread is >= 2 ticks, else match ask
        if ask - bid >= 2 * t:
            raw_px = min(raw_px, ask - t)
        is_optimistic = True

    px = ceil_to_tick(raw_px, tick=t)
    if upper_circuit > 0.0:
        px = min(px, floor_to_tick(upper_circuit, tick=t))
    return max(t, px), is_optimistic, round(spread_pct, 3)


def compute_catastrophic_option_sl(
    entry_premium: float,
    tick_size: float,
    opt_cfg: OptionsConfig,
    lower_circuit: float = 0.0,
) -> tuple[float, float]:
    """Compute broker-resting catastrophic SL-Limit (trigger, limit) for an option BUY position.

    Protects the account if the cloud VM or internet drops while spot moves sharply against
    the trade. Placed far enough below entry (`catastrophic_sl_pct`, default 35%) that normal
    option bid-ask wicks do not trigger it prematurely while spot-triggered exits remain primary.
    """
    t = tick_size if tick_size > 0 else 0.05
    sl_pct = min(60.0, max(15.0, float(opt_cfg.catastrophic_sl_pct))) / 100.0
    buf_pct = min(10.0, max(0.5, float(opt_cfg.catastrophic_sl_limit_buffer_pct))) / 100.0

    raw_trigger = max(t * 2.0, entry_premium * (1.0 - sl_pct))
    trigger = floor_to_tick(raw_trigger, tick=t)
    raw_limit = max(t, trigger * (1.0 - buf_pct))
    limit = floor_to_tick(raw_limit, tick=t)

    if lower_circuit > 0.0:
        min_lim = ceil_to_tick(lower_circuit, tick=t)
        limit = max(limit, min_lim)
        if trigger <= limit:
            trigger = ceil_to_tick(limit + t, tick=t)

    if limit >= trigger:
        limit = max(t, floor_to_tick(trigger - t, tick=t))
    if trigger <= limit:
        trigger = ceil_to_tick(limit + t, tick=t)

    return trigger, limit


def evaluate_option_liquidity(
    contract: OptionContractSpec,
    quote: Optional[Quote],
    opt_cfg: OptionsConfig,
    available_margin: float,
) -> tuple[bool, str, str, float, float, bool]:
    """Evaluate a candidate option contract against real-market depth, spread, activity, and margin gates.

    Returns:
        (passed, reason_code, explanation, spread_pct, entry_limit_price, is_optimistic)
    """
    if quote is None or not getattr(quote, "is_fresh", False) or quote.last_price <= 0:
        return False, "OPTION_STALE_QUOTE", f"Option quote missing or stale for {contract.zerodha_symbol}", 999.0, 0.0, False

    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        return (
            False,
            "OPTION_INVALID_DEPTH",
            f"Invalid two-sided depth for {contract.zerodha_symbol} (bid={quote.bid}, ask={quote.ask})",
            999.0,
            0.0,
            False,
        )

    entry_px, is_opt, spread_pct = compute_optimistic_entry_price(
        quote=quote,
        tick_size=contract.tick_size,
        opt_cfg=opt_cfg,
    )
    if spread_pct > opt_cfg.max_option_spread_pct:
        return (
            False,
            "OPTION_ILLIQUID_SPREAD",
            f"{contract.zerodha_symbol} spread {spread_pct:.2f}% > max {opt_cfg.max_option_spread_pct:.2f}% (bid={quote.bid}, ask={quote.ask})",
            spread_pct,
            0.0,
            False,
        )

    # 5-level Market Depth quantity check (ensures entry AND exit fill cleanly)
    lot = max(1, contract.lot_size)
    qty_needed = lot * max(1, opt_cfg.lots_per_trade)
    min_depth_qty = max(qty_needed, lot * max(1, opt_cfg.min_depth_lots))

    total_bid_qty = int(getattr(quote, "total_bid_qty", 0) or getattr(quote, "bid_qty", 0) or 0)
    total_ask_qty = int(getattr(quote, "total_ask_qty", 0) or getattr(quote, "ask_qty", 0) or 0)
    # Only enforce depth quantity if feed reports non-zero depth quantities (e.g., Angel FULL / live Kite depth)
    if total_bid_qty > 0 or total_ask_qty > 0:
        if total_bid_qty < min_depth_qty or total_ask_qty < min_depth_qty:
            return (
                False,
                "OPTION_THIN_DEPTH",
                f"{contract.zerodha_symbol} thin book depth (bid_qty={total_bid_qty}, ask_qty={total_ask_qty}, need>={min_depth_qty})",
                spread_pct,
                0.0,
                False,
            )

    # Activity gate: volume or OI must prove active market making (blocks Vol=0, OI=0 dead strikes)
    vol = int(getattr(quote, "volume", 0) or 0)
    oi = int(getattr(quote, "open_interest", 0) or 0)
    min_vol = lot * max(1, opt_cfg.min_volume_lots)
    min_oi = lot * max(1, opt_cfg.min_oi_lots)
    if vol < min_vol and oi < min_oi:
        return (
            False,
            "OPTION_DEAD_STRIKE",
            f"{contract.zerodha_symbol} inactive strike (vol={vol} < {min_vol} and oi={oi} < {min_oi})",
            spread_pct,
            0.0,
            False,
        )

    # Capital & margin affordability check
    cap_req = round(entry_px * qty_needed, 2)
    if opt_cfg.max_capital_per_trade_inr > 0 and cap_req > opt_cfg.max_capital_per_trade_inr:
        return (
            False,
            "OPTION_CAPITAL_CAP_EXCEEDED",
            f"{contract.zerodha_symbol} 1-lot cost Rs {cap_req:,.0f} > max cap Rs {opt_cfg.max_capital_per_trade_inr:,.0f}",
            spread_pct,
            entry_px,
            is_opt,
        )
    if available_margin > 0 and cap_req > available_margin:
        return (
            False,
            "OPTION_INSUFFICIENT_MARGIN",
            f"{contract.zerodha_symbol} cost Rs {cap_req:,.0f} > usable cash Rs {available_margin:,.0f}",
            spread_pct,
            entry_px,
            is_opt,
        )

    return True, "OPTION_LIQUID_OK", f"{contract.zerodha_symbol} passed all liquidity gates (spread={spread_pct:.2f}%)", spread_pct, entry_px, is_opt


def select_liquid_option_contract(
    broker: Any,
    underlying: str,
    spot_side: Side,
    spot_price: float,
    today: date,
    opt_cfg: OptionsConfig,
    available_margin: float,
) -> OptionSelectionResult:
    """Resolve active NFO contracts for `underlying`, inspect live market depth, and pick the best liquid ATM strike.

    - LONG spot breakout (`Side.BUY`)  -> Buy Call (`CE`)
    - SHORT spot breakout (`Side.SELL`) -> Buy Put (`PE`)
    - Verifies stock has active NSE F&O options (`OPTSTK`).
    - Evaluates nearest `strike_candidate_count` (default 3: ATM and ATM +/- 1) strikes.
    - Dynamically compares current-month vs next-month live depth when DTE is in [min_dte, rollover_compare_dte].
    """
    opt_type = "CE" if spot_side == Side.BUY else "PE"

    # 1. Fetch available NFO option contracts from broker (Hybrid/Kite/Angel/Paper)
    if not hasattr(broker, "get_nfo_option_contracts"):
        return OptionSelectionResult(
            is_valid=False,
            reason_code="FNO_RESOLVER_UNSUPPORTED",
            explanation=f"Broker adapter does not expose NFO option contract lookup for {underlying}",
        )

    try:
        raw_contracts: list[OptionContractSpec] = broker.get_nfo_option_contracts(underlying)
    except Exception as e:
        logger.error("Failed to fetch NFO contracts for %s: %s", underlying, e)
        return OptionSelectionResult(
            is_valid=False,
            reason_code="FNO_LOOKUP_FAILED",
            explanation=f"Could not query NFO contracts for {underlying}: {e}",
        )

    typed_contracts = [c for c in raw_contracts if c.option_type.upper() == opt_type]
    if not typed_contracts:
        return OptionSelectionResult(
            is_valid=False,
            reason_code="NOT_FNO_ELIGIBLE",
            explanation=f"{underlying} does not have active NSE F&O {opt_type} option contracts",
        )

    # 2. Choose eligible expiry date(s):
    #    - Never trade DTE < min_dte_current_month (default 3) to avoid expiry/delivery margin risk
    #    - If nearest valid expiry has DTE <= rollover_compare_dte (default 6), compare BOTH nearest & next expiry!
    expiries = sorted(set(c.expiry_date for c in typed_contracts if (c.expiry_date - today).days >= opt_cfg.min_dte_current_month))
    if not expiries:
        return OptionSelectionResult(
            is_valid=False,
            reason_code="NO_ELIGIBLE_EXPIRY",
            explanation=f"No NFO expiry with DTE >= {opt_cfg.min_dte_current_month} found for {underlying}",
        )

    candidate_expiries = [expiries[0]]
    first_dte = (expiries[0] - today).days
    if first_dte <= opt_cfg.rollover_compare_dte and len(expiries) > 1:
        candidate_expiries.append(expiries[1])

    # 3. Build candidate strike basket: nearest `strike_candidate_count` strikes around spot
    candidates: list[OptionContractSpec] = []
    n_strikes = max(1, int(opt_cfg.strike_candidate_count))
    for exp_d in candidate_expiries:
        exp_contracts = [c for c in typed_contracts if c.expiry_date == exp_d]
        if opt_cfg.strike_selection == "ITM_1":
            # Prefer 1 strike ITM, then ATM, then 2 ITM
            if opt_type == "CE":
                exp_contracts.sort(key=lambda c: (0 if c.strike <= spot_price else 1, abs(c.strike - spot_price)))
            else:
                exp_contracts.sort(key=lambda c: (0 if c.strike >= spot_price else 1, abs(c.strike - spot_price)))
        else:
            # ATM nearest strike first, then ATM +/- 1
            exp_contracts.sort(key=lambda c: abs(c.strike - spot_price))

        for c in exp_contracts[:n_strikes]:
            candidates.append(
                OptionContractSpec(
                    underlying=c.underlying,
                    expiry_date=c.expiry_date,
                    strike=c.strike,
                    option_type=c.option_type,
                    lot_size=c.lot_size,
                    tick_size=c.tick_size or 0.05,
                    zerodha_symbol=c.zerodha_symbol or format_zerodha_monthly_symbol(c.underlying, c.expiry_date, c.strike, c.option_type),
                    angel_symbol=c.angel_symbol or format_angel_monthly_symbol(c.underlying, c.expiry_date, c.strike, c.option_type),
                    angel_token=c.angel_token,
                    kite_token=c.kite_token,
                    dte=(c.expiry_date - today).days,
                )
            )

    if not candidates:
        return OptionSelectionResult(
            is_valid=False,
            reason_code="NO_CANDIDATE_STRIKES",
            explanation=f"No ATM candidate strikes found for {underlying} {opt_type} around spot {spot_price:.2f}",
        )

    # 4. Batch fetch live FULL NFO market depth for all candidates in a single API call
    quote_lookup_symbols = [c.zerodha_symbol for c in candidates]
    quotes_map: dict[str, Quote] = {}
    try:
        if hasattr(broker, "get_option_quotes_batch"):
            quotes_map = broker.get_option_quotes_batch(candidates)
        elif hasattr(broker, "get_quotes_batch"):
            quotes_map = broker.get_quotes_batch(quote_lookup_symbols, exchange=opt_cfg.exchange)
        else:
            for c in candidates:
                quotes_map[c.zerodha_symbol] = broker.get_quote(c.zerodha_symbol, exchange=opt_cfg.exchange)
    except Exception as qe:
        logger.error("Failed to fetch live NFO depth quotes for %s: %s", underlying, qe)
        return OptionSelectionResult(
            is_valid=False,
            reason_code="OPTION_QUOTE_FETCH_FAILED",
            explanation=f"Failed to fetch live NFO market depth for {underlying}: {qe}",
        )

    # 5. Score all passing candidates: prefer tightest spread, proximity to ATM, and highest volume/depth
    passing: list[tuple[tuple, OptionContractSpec, Quote, float, bool, float]] = []
    rejection_notes: list[str] = []
    last_reason = "OPTION_NO_LIQUID_STRIKE"

    for idx, c in enumerate(candidates):
        q = quotes_map.get(c.zerodha_symbol) or quotes_map.get(c.angel_symbol)
        ok, code, msg, sprd, entry_px, is_opt = evaluate_option_liquidity(
            contract=c,
            quote=q,
            opt_cfg=opt_cfg,
            available_margin=available_margin,
        )
        if ok and q is not None:
            moneyness_pct = abs(c.strike - spot_price) / spot_price * 100.0 if spot_price > 0 else 0.0
            # Composite score (lower is better): tight spread first, slight preference for closer ATM, higher liquidity tie-break
            vol = int(getattr(q, "volume", 0) or 0)
            oi = int(getattr(q, "open_interest", 0) or 0)
            score = (round(sprd, 2) + 0.35 * moneyness_pct, -(vol + oi), idx)
            passing.append((score, c, q, entry_px, is_opt, sprd))
        else:
            last_reason = code
            rejection_notes.append(msg)

    if not passing:
        return OptionSelectionResult(
            is_valid=False,
            reason_code=last_reason,
            explanation=f"All {len(candidates)} candidate {opt_type} strikes for {underlying} failed liquidity/margin checks: {'; '.join(rejection_notes[:3])}",
        )

    passing.sort(key=lambda item: item[0])
    _, best_c, best_q, best_px, is_opt, best_sprd = passing[0]
    qty = best_c.lot_size * max(1, opt_cfg.lots_per_trade)
    cap_req = round(best_px * qty, 2)

    return OptionSelectionResult(
        is_valid=True,
        contract=best_c,
        quote=best_q,
        entry_limit_price=best_px,
        is_optimistic_entry=is_opt,
        spread_pct=best_sprd,
        capital_required=cap_req,
        quantity=qty,
        reason_code="OPTION_SELECTED",
        explanation=(
            f"Selected {best_c.zerodha_symbol} (DTE={best_c.dte}, Lot={best_c.lot_size}, "
            f"Bid/Ask={best_q.bid:.2f}/{best_q.ask:.2f}, Spread={best_sprd:.2f}%, "
            f"EntryLimit={best_px:.2f} {'[Optimistic]' if is_opt else '[Marketable]'}, Cost=Rs {cap_req:,.0f})"
        ),
    )
