"""CME contract specs and round-trip commissions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    tick_value: float          # dollars per tick per contract
    commission_rt: float       # dollars per contract, round trip

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size


INSTRUMENTS = {
    # Equity index (commission $5 RT per protocol)
    "ES":  Instrument("ES", 0.25, 12.50, 5.0),
    "MES": Instrument("MES", 0.25, 1.25, 5.0),
    "NQ":  Instrument("NQ", 0.25, 5.00, 5.0),
    "MNQ": Instrument("MNQ", 0.25, 0.50, 5.0),
    "NKD": Instrument("NKD", 5.0, 25.00, 5.0),
    # Metals / energy ($15 RT)
    "GC":  Instrument("GC", 0.10, 10.00, 15.0),
    "SI":  Instrument("SI", 0.005, 25.00, 15.0),
    "CL":  Instrument("CL", 0.01, 10.00, 15.0),
}


def get(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError:
        raise KeyError(f"Unknown instrument {symbol!r}; add it to propbot/instruments.py") from None
