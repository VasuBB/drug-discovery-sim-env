"""Oversight penalty — saturating exponential of ignored block warnings.

    penalty = 1 - exp(-ignored)

The aggregator subtracts `weights.oversight_penalty * penalty`, so the first
ignored warning is the costliest and additional ignores saturate.
"""

from __future__ import annotations

import math


class OversightPenalty:
    def score(self, warnings_issued: int, warnings_ignored: int) -> float:
        ignored = max(0, int(warnings_ignored))
        return 1.0 - math.exp(-float(ignored))
