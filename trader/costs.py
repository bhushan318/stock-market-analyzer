"""Indian equity delivery (CNC) transaction charges."""


def buy_charges(value, c):
    exch = value * c["exchange_txn"]
    sebi = value * c["sebi"]
    brokerage = c["brokerage"]
    gst = (brokerage + exch + sebi) * c["gst"]
    return value * c["stt"] + value * c["stamp_duty_buy"] + exch + sebi + brokerage + gst


def sell_charges(value, c):
    exch = value * c["exchange_txn"]
    sebi = value * c["sebi"]
    brokerage = c["brokerage"]
    gst = (brokerage + exch + sebi) * c["gst"]
    return value * c["stt"] + exch + sebi + brokerage + gst + c["dp_charge_per_sell"]


def sell_rate(c):
    """Proportional part of sell charges (everything except the flat DP charge and brokerage)."""
    return c["stt"] + (c["exchange_txn"] + c["sebi"]) * (1 + c["gst"])


def target_price(shares, buy_price, cost_basis, take_profit, c, net=True):
    """Price at which selling `shares` realises `take_profit` on the position.

    net=True: profit is measured after buy *and* sell charges, i.e. on cost_basis, and the
    sell price covers the sell-side charges too - so a target exit can never be a loss.
    """
    if not net:
        return buy_price * (1 + take_profit)
    fixed = c["dp_charge_per_sell"] + c["brokerage"] * (1 + c["gst"])
    return (cost_basis * (1 + take_profit) + fixed) / (shares * (1 - sell_rate(c)))
