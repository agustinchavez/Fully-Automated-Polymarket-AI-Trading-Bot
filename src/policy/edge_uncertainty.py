"""Edge uncertainty scoring — quantify forecast confidence and penalize edge.

Computes a composite uncertainty score from:
  - Ensemble spread (model disagreement)
  - Evidence quality (information completeness)
  - Base rate distance (how extreme the forecast is)
  - Decomposition disagreement (sub-question inconsistency)

Effective edge is reduced proportionally to uncertainty, preventing
trades on noisy or low-confidence signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.observability.logger import get_logger

log = get_logger(__name__)

# Component weights (from spec section 3.1)
_W_ENSEMBLE_SPREAD = 0.30
_W_EVIDENCE_QUALITY = 0.25
_W_BASE_RATE_DISTANCE = 0.25
_W_DECOMPOSITION = 0.20

# Default decomposition disagreement when decomposition is disabled (no penalty)
_DEFAULT_DECOMPOSITION_DISAGREEMENT = 0.0

# Maximum possible std_dev for probabilities in [0, 1]
_MAX_PROB_STDDEV = 0.25


@dataclass
class UncertaintyInputs:
    """Raw inputs for uncertainty calculation."""
    ensemble_spread: float = 0.0
    evidence_quality: float = 0.5
    base_rate: float = 0.5
    model_probability: float = 0.5
    decomposition_sub_probs: list[float] = field(default_factory=list)


@dataclass
class EdgeUncertaintyResult:
    """Computed uncertainty and its components."""
    uncertainty_score: float
    effective_edge: float
    raw_edge: float
    ensemble_spread_component: float
    evidence_quality_component: float
    base_rate_distance_component: float
    decomposition_disagreement_component: float
    was_adjusted: bool = True


def compute_edge_uncertainty(inputs: UncertaintyInputs) -> float:
    """Compute composite uncertainty score from 0 (certain) to 1 (maximum uncertainty).

    Formula:
      uncertainty = 0.30 × ensemble_spread
                  + 0.25 × (1 - evidence_quality)
                  + 0.25 × base_rate_distance
                  + 0.20 × decomposition_disagreement
    """
    # Clamp inputs to valid ranges
    spread = max(0.0, min(1.0, inputs.ensemble_spread))
    eq = max(0.0, min(1.0, inputs.evidence_quality))
    br_distance = min(1.0, abs(inputs.model_probability - inputs.base_rate))

    # Decomposition disagreement: std_dev of sub-probs, normalized
    if len(inputs.decomposition_sub_probs) >= 2:
        mean_p = sum(inputs.decomposition_sub_probs) / len(inputs.decomposition_sub_probs)
        variance = sum(
            (p - mean_p) ** 2 for p in inputs.decomposition_sub_probs
        ) / len(inputs.decomposition_sub_probs)
        stddev = math.sqrt(variance)
        decomp_disagreement = min(1.0, stddev / _MAX_PROB_STDDEV)
    else:
        decomp_disagreement = _DEFAULT_DECOMPOSITION_DISAGREEMENT

    uncertainty = (
        _W_ENSEMBLE_SPREAD * spread
        + _W_EVIDENCE_QUALITY * (1.0 - eq)
        + _W_BASE_RATE_DISTANCE * br_distance
        + _W_DECOMPOSITION * decomp_disagreement
    )

    return max(0.0, min(1.0, uncertainty))


def adjust_edge_for_uncertainty(
    raw_edge: float,
    uncertainty_score: float,
    penalty_factor: float = 0.5,
) -> float:
    """Reduce effective edge based on uncertainty.

    effective_edge = raw_edge × (1 - uncertainty × penalty_factor)

    Examples with penalty_factor=0.5:
      - 0% uncertainty: effective_edge = raw_edge
      - 80% uncertainty: effective_edge = raw_edge × 0.60
      - 100% uncertainty: effective_edge = raw_edge × 0.50
    """
    return raw_edge * (1.0 - uncertainty_score * penalty_factor)


def compute_and_adjust(
    inputs: UncertaintyInputs,
    raw_edge: float,
    penalty_factor: float = 0.5,
) -> EdgeUncertaintyResult:
    """Compute uncertainty and return full result with breakdown."""
    # Clamp inputs for component reporting
    spread = max(0.0, min(1.0, inputs.ensemble_spread))
    eq = max(0.0, min(1.0, inputs.evidence_quality))
    br_distance = min(1.0, abs(inputs.model_probability - inputs.base_rate))

    if len(inputs.decomposition_sub_probs) >= 2:
        mean_p = sum(inputs.decomposition_sub_probs) / len(inputs.decomposition_sub_probs)
        variance = sum(
            (p - mean_p) ** 2 for p in inputs.decomposition_sub_probs
        ) / len(inputs.decomposition_sub_probs)
        stddev = math.sqrt(variance)
        decomp_disagreement = min(1.0, stddev / _MAX_PROB_STDDEV)
    else:
        decomp_disagreement = _DEFAULT_DECOMPOSITION_DISAGREEMENT

    # Weighted components
    c_spread = _W_ENSEMBLE_SPREAD * spread
    c_evidence = _W_EVIDENCE_QUALITY * (1.0 - eq)
    c_base_rate = _W_BASE_RATE_DISTANCE * br_distance
    c_decomp = _W_DECOMPOSITION * decomp_disagreement

    uncertainty = max(0.0, min(1.0, c_spread + c_evidence + c_base_rate + c_decomp))
    effective = adjust_edge_for_uncertainty(raw_edge, uncertainty, penalty_factor)

    result = EdgeUncertaintyResult(
        uncertainty_score=uncertainty,
        effective_edge=effective,
        raw_edge=raw_edge,
        ensemble_spread_component=round(c_spread, 4),
        evidence_quality_component=round(c_evidence, 4),
        base_rate_distance_component=round(c_base_rate, 4),
        decomposition_disagreement_component=round(c_decomp, 4),
    )

    log.info(
        "edge_uncertainty.computed",
        uncertainty=round(uncertainty, 3),
        raw_edge=round(raw_edge, 4),
        effective_edge=round(effective, 4),
        spread_comp=result.ensemble_spread_component,
        evidence_comp=result.evidence_quality_component,
        base_rate_comp=result.base_rate_distance_component,
        decomp_comp=result.decomposition_disagreement_component,
    )

    return result
