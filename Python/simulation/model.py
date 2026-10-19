"""
Agent-Based Model for Variable Rate Nitrogen Application (VRNA) Technology Adoption
Based on FARMIND framework with Cumulative Prospect Theory (CPT)
Author: Implementation for Zahra Pezeshkian's research

REVISED VERSION - aligned with the manuscript and ODD (Appendix A).
Changes relative to the original model.py are marked "CHANGE 1/2/3":

  CHANGE 1  Annual input-price projection. Nitrogen price and operating costs grow
            each year at fixed rates, so realised gross margins in a given
            environment state differ between years (manuscript Section 3.4).
            Stage 2 evaluates incremental payoffs at the current year's N price.
  CHANGE 2  Stage 1 decision weights follow manuscript Eq. (4): remembered gross
            margins are assumed to be normally distributed, and decision weights are
            differences in weighted decumulative probabilities of that normal
            distribution (FARMIND, Huber et al., 2022, 2023).
  CHANGE 3  An agent is satisfied (repetition) if PV >= 0, as stated in the
            manuscript (original code used PV > 0).

Unchanged on purpose: the profit-maximisation benchmark (state-specific comparison)
and the area-based payment (payment x farm_size_ha).
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# CHANGE 1: input-price parameters (base year = year in which the input-file
# gross margins were computed). FILL IN the values from manuscript Table 5.
# ---------------------------------------------------------------------------
DEFAULT_PRICE_PARAMS = {
    "n_uniform_kg_ha": 109.0,   # uniform N rate under baseline management (kg N/ha)
    "n_price_base": None,       # base-year nitrogen price (CAD per kg N)      <-- FILL IN
    "n_price_growth": None,     # annual growth rate of N price (e.g., 0.02)   <-- FILL IN
    "opcost_base": None,        # base-year operating cost (CAD per ha)        <-- FILL IN
    "opcost_growth": None,      # annual growth rate of operating costs        <-- FILL IN
}


@dataclass
class FarmerAgent:
    """Represents a farmer agent with CPT parameters and farm characteristics."""
    farmer_id: int
    farm_size_ha: float
    alpha: float  # Risk sensitivity for gains (α+)
    beta_val: float  # Risk sensitivity for losses (α-)
    lambda_param: float  # Loss aversion (λ)
    gamma_gain: float  # Probability weighting for gains
    gamma_loss: float  # Probability weighting for losses
    vref: float  # Reference gross margin
    gm_vrna: np.ndarray  # Gross margins under VRNA for 60 states (base-year prices)
    gm_baseline: np.ndarray  # Gross margins under baseline for 60 states (base-year prices)
    benefit: np.ndarray  # Pre-computed Δπ = GM_VRNA - GM_Baseline (base-year prices)
    nred: np.ndarray  # Nitrogen reduction for 60 states (kg)

    # Dynamic state variables
    adopted: bool = False  # Current adoption status
    memory_gm: List[float] = field(default_factory=list)  # Past realized GMs
    adoption_history: List[int] = field(default_factory=list)  # Adoption status per year

    def __post_init__(self):
        self.memory_gm = []
        self.adoption_history = []


class CPTCalculator:
    """Implements Cumulative Prospect Theory calculations."""

    @staticmethod
    def value_function(x: float, alpha: float, beta: float, lambda_param: float) -> float:
        """v(x) = x^α if x >= 0;  v(x) = -λ|x|^β if x < 0"""
        if x >= 0:
            return np.power(x, alpha)
        else:
            return -lambda_param * np.power(np.abs(x), beta)

    @staticmethod
    def probability_weight(p: float, gamma: float) -> float:
        """w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)   (Wu and Gonzalez, 1996)"""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        if gamma <= 0:
            gamma = 0.01  # Avoid division issues
        p_gamma = np.power(p, gamma)
        one_minus_p_gamma = np.power(1 - p, gamma)
        denominator = np.power(p_gamma + one_minus_p_gamma, 1 / gamma)
        if denominator == 0:
            return 0.0
        return p_gamma / denominator

    @staticmethod
    def compute_decision_weights(outcomes: np.ndarray, gamma_gain: float, gamma_loss: float) -> np.ndarray:
        """
        Rank-dependent CPT decision weights with equal objective probabilities
        (used in Stage 2, over the 60 environment states). Unchanged.
        """
        n = len(outcomes)
        weights = np.zeros(n)
        sorted_indices = np.argsort(outcomes)
        sorted_outcomes = outcomes[sorted_indices]
        for i, idx in enumerate(sorted_indices):
            outcome = sorted_outcomes[i]
            cumulative_prob = (i + 1) / n
            prev_cumulative_prob = i / n
            if outcome >= 0:
                w_current = CPTCalculator.probability_weight(1 - prev_cumulative_prob, gamma_gain)
                w_next = CPTCalculator.probability_weight(1 - cumulative_prob, gamma_gain)
                weights[idx] = w_current - w_next
            else:
                w_current = CPTCalculator.probability_weight(cumulative_prob, gamma_loss)
                w_prev = CPTCalculator.probability_weight(prev_cumulative_prob, gamma_loss)
                weights[idx] = w_current - w_prev
        return weights

    # ------------------------------------------------------------------
    # CHANGE 2: Stage 1 decision weights based on a normal distribution
    # ------------------------------------------------------------------
    @staticmethod
    def _normal_cdf(x: float, mu: float, sd: float) -> float:
        if x == math.inf:
            return 1.0
        if x == -math.inf:
            return 0.0
        return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))

    @staticmethod
    def compute_decision_weights_normal(outcomes: np.ndarray, gamma: float) -> np.ndarray:
        """
        Manuscript Eq. (4): Φ(x) = w(1 - F(x)) - w(1 - F(x + Δ)), where F is the
        cumulative distribution function of a normal distribution fitted to the
        remembered gross margins.

        Implementation: outcomes are ranked; each outcome represents the interval
        between the midpoints to its neighbouring outcomes (the lowest outcome extends
        to -inf, the highest to +inf), so that Δ is the width of that interval and the
        weights sum to one.
        """
        x = np.asarray(outcomes, dtype=float)
        m = len(x)
        if m == 1:
            return np.ones(1)
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        # Identical outcomes (no variation): all weight is equally shared
        if not np.isfinite(sd) or sd <= 1e-9 * max(1.0, abs(mu)):
            return np.full(m, 1.0 / m)

        order = np.argsort(x)
        xs = x[order]
        mids = (xs[:-1] + xs[1:]) / 2.0
        lower = np.concatenate(([-math.inf], mids))
        upper = np.concatenate((mids, [math.inf]))

        weights = np.zeros(m)
        for k in range(m):
            p_above_lower = 1.0 - CPTCalculator._normal_cdf(lower[k], mu, sd)
            p_above_upper = 1.0 - CPTCalculator._normal_cdf(upper[k], mu, sd)
            weights[order[k]] = (CPTCalculator.probability_weight(p_above_lower, gamma)
                                 - CPTCalculator.probability_weight(p_above_upper, gamma))
        return weights

    @staticmethod
    def compute_prospect_value(outcomes: np.ndarray, reference: float,
                               alpha: float, beta: float, lambda_param: float,
                               gamma_gain: float, gamma_loss: float,
                               weights: Optional[np.ndarray] = None) -> float:
        """PV = Σ Φ_x * v(x - reference). Weights can be supplied (Stage 1)."""
        deviations = outcomes - reference
        if weights is None:
            weights = CPTCalculator.compute_decision_weights(deviations, gamma_gain, gamma_loss)
        pv = 0.0
        for i, dev in enumerate(deviations):
            pv += weights[i] * CPTCalculator.value_function(dev, alpha, beta, lambda_param)
        return pv

    @staticmethod
    def compute_certainty_equivalent(pv: float, alpha: float, beta: float, lambda_param: float) -> float:
        """CE = PV^(1/α) if PV >= 0;  CE = -(|PV|/λ)^(1/β) if PV < 0"""
        if pv >= 0:
            return np.power(pv, 1 / alpha)
        else:
            return -np.power(np.abs(pv) / lambda_param, 1 / beta)


class VRNAAdoptionModel:
    """
    Agent-Based Model for VRNA Technology Adoption.
    Implements the two-stage decision process based on FARMIND framework.
    """

    def __init__(self, data_path: str, memory_horizon: int = 5, skip_stage1: bool = False,
                 profit_max_stage2: bool = False, price_params: Optional[Dict] = None):
        self.memory_horizon = memory_horizon
        self.skip_stage1 = skip_stage1
        self.profit_max_stage2 = profit_max_stage2
        self.agents: List[FarmerAgent] = []
        self.cpt = CPTCalculator()

        # CHANGE 1: price parameters
        self.price = dict(DEFAULT_PRICE_PARAMS)
        if price_params:
            self.price.update(price_params)
        missing = [k for k, v in self.price.items() if v is None]
        if missing:
            raise ValueError(f"Fill in the price parameters before running: {missing}")

        self._stage2_cache: Dict = {}
        self.load_data(data_path)

    def load_data(self, data_path: str):
        """Load farmer data from CSV and create agents."""
        df = pd.read_csv(data_path)
        df = df[df['Vref'].notna() & (df['Vref'] != 'NA')]
        df['Vref'] = df['Vref'].astype(float)
        print(f"Loaded {len(df)} farmers (excluded those with NA Vref)")

        for _, row in df.iterrows():
            gm_vrna = np.array([row[f'GM_VRNA_{i}'] for i in range(1, 61)], dtype=float)
            gm_baseline = np.array([row[f'GM_Baseline_{i}'] for i in range(1, 61)], dtype=float)
            benefit = np.array([row[f'Benefit_{i}'] for i in range(1, 61)], dtype=float)
            nred = np.array([row[f'Nred_{i}'] for i in range(1, 61)], dtype=float)
            agent = FarmerAgent(
                farmer_id=int(row['Farmer_ID']),
                farm_size_ha=float(row['Farm_Size_ha']),
                alpha=float(row['alpha']),
                beta_val=float(row['beta_val']),
                lambda_param=float(row['lambda']),
                gamma_gain=float(row['gamma_gain']),
                gamma_loss=float(row['gamma_loss']),
                vref=float(row['Vref']),
                gm_vrna=gm_vrna,
                gm_baseline=gm_baseline,
                benefit=benefit,
                nred=nred
            )
            self.agents.append(agent)

    def reset_agents(self):
        """Reset all agents to initial non-adopter state."""
        for agent in self.agents:
            agent.adopted = False
            agent.memory_gm = []
            agent.adoption_history = []

    # ------------------------------------------------------------------
    # CHANGE 1: price changes relative to the base year
    # ------------------------------------------------------------------
    def price_changes(self, year: int) -> Tuple[float, float]:
        """
        Change in N price (CAD/kg) and operating cost (CAD/ha) in `year`
        relative to the base year (year 1 = base year, no change).
        """
        k = year - 1
        d_n_price = self.price["n_price_base"] * ((1 + self.price["n_price_growth"]) ** k - 1)
        d_opcost = self.price["opcost_base"] * ((1 + self.price["opcost_growth"]) ** k - 1)
        return d_n_price, d_opcost

    def realized_gm(self, agent: FarmerAgent, state_idx: int, adopted: bool, year: int) -> float:
        """Realised gross margin in `year` (without policy payment)."""
        d_n_price, d_opcost = self.price_changes(year)
        n_baseline = self.price["n_uniform_kg_ha"] * agent.farm_size_ha
        if adopted:
            gm = agent.gm_vrna[state_idx]
            n_applied = n_baseline - agent.nred[state_idx]
        else:
            gm = agent.gm_baseline[state_idx]
            n_applied = n_baseline
        return gm - n_applied * d_n_price - agent.farm_size_ha * d_opcost

    def compute_policy_payment(self, agent: FarmerAgent, state_idx: int,
                               policy_type: str, payment_level: float) -> float:
        """Compute policy payment for a given agent and state. Unchanged."""
        if policy_type == 'results_based':
            return payment_level * agent.nred[state_idx]
        elif policy_type == 'area_based':
            return payment_level * agent.farm_size_ha
        elif policy_type == 'flat':
            return payment_level
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    def stage1_strategy_selection(self, agent: FarmerAgent) -> str:
        """
        Stage 1: Repetition vs Optimization, based on satisfaction with past
        performance (manuscript Eqs. 1-5).
        """
        if len(agent.memory_gm) == 0:
            # Year 1: baseline GMs across the 60 states serve as the history
            outcomes = agent.gm_baseline.copy()
        else:
            outcomes = np.array(agent.memory_gm[-self.memory_horizon:])

        # CHANGE 2: normal-distribution-based decision weights (Eq. 4)
        weights = self.cpt.compute_decision_weights_normal(outcomes, agent.gamma_gain)

        pv = self.cpt.compute_prospect_value(
            outcomes=outcomes,
            reference=agent.vref,
            alpha=agent.alpha,
            beta=agent.beta_val,
            lambda_param=agent.lambda_param,
            gamma_gain=agent.gamma_gain,
            gamma_loss=agent.gamma_loss,
            weights=weights
        )
        # CHANGE 3: satisfied if PV >= 0
        return 'repetition' if pv >= 0 else 'optimization'

    def stage2_profit_max_decision(self, agent: FarmerAgent, state_idx: int,
                                   policy_type: str, payment_level: float,
                                   year: int = 1) -> bool:
        """
        Stage 2 (profit-maximisation benchmark): state-specific comparison.
        UNCHANGED decision rule; N price of the current year is applied for
        consistency with CHANGE 1.
        """
        d_n_price, _ = self.price_changes(year)
        base_benefit = agent.benefit[state_idx] + agent.nred[state_idx] * d_n_price
        policy_payment = self.compute_policy_payment(agent, state_idx, policy_type, payment_level)
        return base_benefit + policy_payment > 0

    def stage2_adoption_decision(self, agent: FarmerAgent, policy_type: str,
                                 payment_level: float, year: int = 1) -> bool:
        """
        Stage 2: adopt if the CPT certainty equivalent of the incremental payoff
        distribution across the 60 states (incl. payment) is positive.
        CHANGE 1: incremental payoffs reflect the current year's N price
        (VRNA saves N, so a higher N price raises its benefit).
        The decision does not depend on the current state, so it is cached.
        """
        key = (agent.farmer_id, policy_type, payment_level, year)
        if key in self._stage2_cache:
            return self._stage2_cache[key]

        d_n_price, _ = self.price_changes(year)
        delta_pi = np.zeros(60)
        for s in range(60):
            base_benefit = agent.benefit[s] + agent.nred[s] * d_n_price
            policy_payment = self.compute_policy_payment(agent, s, policy_type, payment_level)
            delta_pi[s] = base_benefit + policy_payment

        pv = self.cpt.compute_prospect_value(
            outcomes=delta_pi, reference=0.0,
            alpha=agent.alpha, beta=agent.beta_val, lambda_param=agent.lambda_param,
            gamma_gain=agent.gamma_gain, gamma_loss=agent.gamma_loss
        )
        ce = self.cpt.compute_certainty_equivalent(
            pv=pv, alpha=agent.alpha, beta=agent.beta_val, lambda_param=agent.lambda_param
        )
        decision = ce > 0
        self._stage2_cache[key] = decision
        return decision

    def simulate_year(self, state_idx: int, policy_type: str,
                      payment_level: float, year: int) -> Dict:
        """Simulate one year for all agents."""
        adoptions = []
        n_reductions = []

        for agent in self.agents:
            if self.skip_stage1:
                strategy = 'optimization'
            else:
                strategy = self.stage1_strategy_selection(agent)

            if strategy == 'repetition':
                new_adoption = agent.adopted
            else:
                if self.profit_max_stage2:
                    new_adoption = self.stage2_profit_max_decision(
                        agent, state_idx, policy_type, payment_level, year)
                else:
                    new_adoption = self.stage2_adoption_decision(
                        agent, policy_type, payment_level, year)

            agent.adopted = new_adoption
            agent.adoption_history.append(1 if new_adoption else 0)

            # CHANGE 1: realised GM of the current year (without policy payment)
            gm = self.realized_gm(agent, state_idx, new_adoption, year)
            agent.memory_gm.append(gm)

            adoptions.append(1 if new_adoption else 0)
            n_reductions.append(agent.nred[state_idx] if new_adoption else 0.0)

        return {
            'year': year,
            'adoption_rate': np.mean(adoptions),
            'total_n_reduction': np.sum(n_reductions),
            'adoptions': adoptions
        }

    def simulate_state(self, state_idx: int, policy_type: str,
                       payment_level: float, n_years: int = 10) -> Dict:
        """Simulate 10 years for a single environment state."""
        self.reset_agents()
        yearly_results = []
        for year in range(1, n_years + 1):
            yearly_results.append(self.simulate_year(state_idx, policy_type, payment_level, year))

        final_adoptions = [agent.adoption_history[-1] for agent in self.agents]
        final_n_reduction = sum(agent.nred[state_idx] if agent.adopted else 0
                                for agent in self.agents)
        final_ha_vrna = sum(agent.farm_size_ha if agent.adopted else 0
                            for agent in self.agents)
        return {
            'state_idx': state_idx,
            'final_adoption_rate': np.mean(final_adoptions),
            'final_n_reduction': final_n_reduction,
            'final_ha_vrna': final_ha_vrna,
            'final_adopters': int(np.sum(final_adoptions)),
            'yearly_results': yearly_results,
            'adoption_histories': [agent.adoption_history.copy() for agent in self.agents]
        }

    def simulate_payment_level(self, policy_type: str, payment_level: float,
                               n_states: int = 60, n_years: int = 10) -> Dict:
        """Run full simulation for one payment level across all 60 states."""
        all_state_results = [self.simulate_state(s, policy_type, payment_level, n_years)
                             for s in range(n_states)]

        farmer_adoption_matrices = []
        for agent_idx in range(len(self.agents)):
            agent_matrix = np.zeros((n_states, n_years))
            for state_idx, result in enumerate(all_state_results):
                agent_matrix[state_idx, :] = result['adoption_histories'][agent_idx]
            farmer_adoption_matrices.append(agent_matrix)

        return {
            'policy_type': policy_type,
            'payment_level': payment_level,
            'avg_adoption_rate': np.mean([r['final_adoption_rate'] for r in all_state_results]),
            'expected_n_reduction': np.mean([r['final_n_reduction'] for r in all_state_results]),
            'expected_ha_vrna': np.mean([r['final_ha_vrna'] for r in all_state_results]),
            'expected_adopters': np.mean([r['final_adopters'] for r in all_state_results]),
            'state_results': all_state_results,
            'farmer_adoption_matrices': farmer_adoption_matrices
        }
