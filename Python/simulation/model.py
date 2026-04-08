"""
Agent-Based Model for Variable Rate Nitrogen Application (VRNA) Technology Adoption
Based on FARMIND framework with Cumulative Prospect Theory (CPT)
Author: Implementation for Zahra Pezeshkian's research
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


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
    gm_vrna: np.ndarray  # Gross margins under VRNA for 60 states
    gm_baseline: np.ndarray  # Gross margins under baseline for 60 states
    benefit: np.ndarray  # Pre-computed Δπ = GM_VRNA - GM_Baseline
    nred: np.ndarray  # Nitrogen reduction for 60 states
    
    # Dynamic state variables
    adopted: bool = False  # Current adoption status
    memory_gm: List[float] = field(default_factory=list)  # Past 5 years of realized GMs
    adoption_history: List[int] = field(default_factory=list)  # Adoption status per year
    
    def __post_init__(self):
        self.memory_gm = []
        self.adoption_history = []


class CPTCalculator:
    """Implements Cumulative Prospect Theory calculations."""
    
    @staticmethod
    def value_function(x: float, alpha: float, beta: float, lambda_param: float) -> float:
        """
        CPT value function.
        v(x) = x^α if x >= 0
        v(x) = -λ|x|^β if x < 0
        """
        if x >= 0:
            return np.power(x, alpha)
        else:
            return -lambda_param * np.power(np.abs(x), beta)
    
    @staticmethod
    def probability_weight(p: float, gamma: float) -> float:
        """
        CPT probability weighting function.
        w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
        """
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
        Compute CPT decision weights for ranked outcomes.
        Uses cumulative probability weighting procedure.
        """
        n = len(outcomes)
        weights = np.zeros(n)
        
        # Sort outcomes and get indices
        sorted_indices = np.argsort(outcomes)
        sorted_outcomes = outcomes[sorted_indices]
        
        # Equal objective probabilities
        p = 1.0 / n
        
        for i, idx in enumerate(sorted_indices):
            outcome = sorted_outcomes[i]
            
            # Cumulative probability up to this outcome
            cumulative_prob = (i + 1) / n
            prev_cumulative_prob = i / n
            
            if outcome >= 0:
                # For gains: w(P(X >= x)) - w(P(X > x))
                # which is w(1 - F(x-)) - w(1 - F(x))
                w_current = CPTCalculator.probability_weight(1 - prev_cumulative_prob, gamma_gain)
                w_next = CPTCalculator.probability_weight(1 - cumulative_prob, gamma_gain)
                weights[idx] = w_current - w_next
            else:
                # For losses: w(F(x)) - w(F(x-))
                w_current = CPTCalculator.probability_weight(cumulative_prob, gamma_loss)
                w_prev = CPTCalculator.probability_weight(prev_cumulative_prob, gamma_loss)
                weights[idx] = w_current - w_prev
        
        return weights
    
    @staticmethod
    def compute_prospect_value(outcomes: np.ndarray, reference: float,
                                alpha: float, beta: float, lambda_param: float,
                                gamma_gain: float, gamma_loss: float) -> float:
        """
        Compute prospect value for a distribution of outcomes.
        PV = Σ π_x * v(x - reference)
        """
        deviations = outcomes - reference
        weights = CPTCalculator.compute_decision_weights(deviations, gamma_gain, gamma_loss)
        
        pv = 0.0
        for i, dev in enumerate(deviations):
            v = CPTCalculator.value_function(dev, alpha, beta, lambda_param)
            pv += weights[i] * v
        
        return pv
    
    @staticmethod
    def compute_certainty_equivalent(pv: float, alpha: float, beta: float, lambda_param: float) -> float:
        """
        Compute certainty equivalent from prospect value.
        CE = PV^(1/α) if PV >= 0
        CE = -(|PV|/λ)^(1/β) if PV < 0
        """
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
                 profit_max_stage2: bool = False):
        """
        Initialize the model with farmer data.
        
        Args:
            data_path: Path to CSV file with farmer data
            memory_horizon: Number of years for memory (default 5)
            skip_stage1: If True, all agents are optimizers (skip satisfaction check)
            profit_max_stage2: If True, use simple profit comparison in Stage 2 instead of CPT
        """
        self.memory_horizon = memory_horizon
        self.skip_stage1 = skip_stage1
        self.profit_max_stage2 = profit_max_stage2
        self.agents: List[FarmerAgent] = []
        self.cpt = CPTCalculator()
        self.load_data(data_path)
        
    def load_data(self, data_path: str):
        """Load farmer data from CSV and create agents."""
        df = pd.read_csv(data_path)
        
        # Remove farmers with NA Vref
        df = df[df['Vref'].notna() & (df['Vref'] != 'NA')]
        df['Vref'] = df['Vref'].astype(float)
        
        print(f"Loaded {len(df)} farmers (excluded those with NA Vref)")
        
        for _, row in df.iterrows():
            # Extract GM_VRNA columns (1-60)
            gm_vrna = np.array([row[f'GM_VRNA_{i}'] for i in range(1, 61)])
            gm_baseline = np.array([row[f'GM_Baseline_{i}'] for i in range(1, 61)])
            benefit = np.array([row[f'Benefit_{i}'] for i in range(1, 61)])
            nred = np.array([row[f'Nred_{i}'] for i in range(1, 61)])
            
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
    
    def compute_policy_payment(self, agent: FarmerAgent, state_idx: int,
                                policy_type: str, payment_level: float) -> float:
        """
        Compute policy payment for a given agent and state.
        
        Args:
            agent: The farmer agent
            state_idx: Index of the environment state (0-59)
            policy_type: 'results_based', 'area_based', or 'flat'
            payment_level: The payment rate
            
        Returns:
            Policy payment amount
        """
        if policy_type == 'results_based':
            # CAD per kg of N saved
            return payment_level * agent.nred[state_idx]
        elif policy_type == 'area_based':
            # CAD per hectare
            return payment_level * agent.farm_size_ha
        elif policy_type == 'flat':
            # CAD per farm
            return payment_level
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def stage1_strategy_selection(self, agent: FarmerAgent) -> str:
        """
        Stage 1: Determine decision strategy (Repetition vs Optimization).
        Based on satisfaction from past performance.
        
        Returns:
            'repetition' if satisfied (PV > 0), 'optimization' if dissatisfied
        """
        if len(agent.memory_gm) == 0:
            # Use baseline GMs as historical distribution for year 1
            outcomes = agent.gm_baseline.copy()
        else:
            # Use actual memory of past realized GMs
            outcomes = np.array(agent.memory_gm[-self.memory_horizon:])
        
        # Compute prospect value relative to reference income
        pv = self.cpt.compute_prospect_value(
            outcomes=outcomes,
            reference=agent.vref,
            alpha=agent.alpha,
            beta=agent.beta_val,
            lambda_param=agent.lambda_param,
            gamma_gain=agent.gamma_gain,
            gamma_loss=agent.gamma_loss
        )
        
        return 'repetition' if pv > 0 else 'optimization'
    
    def stage2_profit_max_decision(self, agent: FarmerAgent, state_idx: int,
                                    policy_type: str, payment_level: float) -> bool:
        """
        Stage 2 (Profit Maximizer): State-specific profit comparison (Huber et al. style).
        
        Adopt if: GM_VRNA(state) + payment > GM_baseline(state)
        Equivalently: Δπ(state) + payment > 0
        
        This evaluates adoption for THIS specific scenario only,
        allowing different farmers to adopt in different scenarios.
        
        Returns:
            True if agent should adopt VRNA, False otherwise
        """
        # Compute Δπ for THIS state only (not all 60)
        base_benefit = agent.benefit[state_idx]  # GM_VRNA - GM_Baseline for this state
        policy_payment = self.compute_policy_payment(agent, state_idx, policy_type, payment_level)
        delta_pi = base_benefit + policy_payment
        
        # Adopt if profit from VRNA > profit from baseline in THIS scenario
        return delta_pi > 0
    
    def stage2_adoption_decision(self, agent: FarmerAgent, 
                                  policy_type: str, payment_level: float) -> bool:
        """
        Stage 2: Evaluate VRNA adoption using CPT certainty equivalent.
        
        Returns:
            True if agent should adopt VRNA, False otherwise
        """
        # Compute Δπ for all 60 states including policy payment
        delta_pi = np.zeros(60)
        for s in range(60):
            base_benefit = agent.benefit[s]  # GM_VRNA - GM_Baseline
            policy_payment = self.compute_policy_payment(agent, s, policy_type, payment_level)
            delta_pi[s] = base_benefit + policy_payment
        
        # Compute prospect value with reference = 0 (no improvement)
        pv = self.cpt.compute_prospect_value(
            outcomes=delta_pi,
            reference=0.0,
            alpha=agent.alpha,
            beta=agent.beta_val,
            lambda_param=agent.lambda_param,
            gamma_gain=agent.gamma_gain,
            gamma_loss=agent.gamma_loss
        )
        
        # Compute certainty equivalent
        ce = self.cpt.compute_certainty_equivalent(
            pv=pv,
            alpha=agent.alpha,
            beta=agent.beta_val,
            lambda_param=agent.lambda_param
        )
        
        # Adopt if CE > 0
        return ce > 0
    
    def simulate_year(self, state_idx: int, policy_type: str, 
                      payment_level: float, year: int) -> Dict:
        """
        Simulate one year for all agents.
        
        Args:
            state_idx: Environment state index (0-59)
            policy_type: Type of policy instrument
            payment_level: Payment level for the policy
            year: Current simulation year (1-10)
            
        Returns:
            Dictionary with year results
        """
        adoptions = []
        n_reductions = []
        
        for agent in self.agents:
            if self.skip_stage1:
                # Skip Stage 1 - all agents are optimizers
                strategy = 'optimization'
            else:
                # Stage 1: Determine strategy based on satisfaction
                strategy = self.stage1_strategy_selection(agent)
            
            if strategy == 'repetition':
                # Keep current adoption status (no change)
                new_adoption = agent.adopted
            else:
                # Stage 2: Optimization - Evaluate VRNA adoption
                if self.profit_max_stage2:
                    # State-specific profit comparison (Huber et al. style)
                    new_adoption = self.stage2_profit_max_decision(
                        agent, state_idx, policy_type, payment_level
                    )
                else:
                    # CPT-based CE decision (uses all 60 states)
                    new_adoption = self.stage2_adoption_decision(
                        agent, policy_type, payment_level
                    )
            
            # Update agent state
            agent.adopted = new_adoption
            agent.adoption_history.append(1 if new_adoption else 0)
            
            # Determine realized GM for this year (raw, without policy payment for memory)
            if new_adoption:
                realized_gm = agent.gm_vrna[state_idx]
                n_reduction = agent.nred[state_idx]
            else:
                realized_gm = agent.gm_baseline[state_idx]
                n_reduction = 0.0
            
            # Update memory (raw GM without policy payment)
            agent.memory_gm.append(realized_gm)
            
            adoptions.append(1 if new_adoption else 0)
            n_reductions.append(n_reduction)
        
        return {
            'year': year,
            'adoption_rate': np.mean(adoptions),
            'total_n_reduction': np.sum(n_reductions),
            'adoptions': adoptions
        }
    
    def simulate_state(self, state_idx: int, policy_type: str,
                       payment_level: float, n_years: int = 10) -> Dict:
        """
        Simulate 10 years for a single environment state.
        
        Returns:
            Dictionary with simulation results
        """
        self.reset_agents()
        
        yearly_results = []
        for year in range(1, n_years + 1):
            result = self.simulate_year(state_idx, policy_type, payment_level, year)
            yearly_results.append(result)
        
        # Get final year adoption status and N reduction
        final_adoptions = [agent.adoption_history[-1] for agent in self.agents]
        final_n_reduction = sum(
            agent.nred[state_idx] if agent.adopted else 0 
            for agent in self.agents
        )
        
        return {
            'state_idx': state_idx,
            'final_adoption_rate': np.mean(final_adoptions),
            'final_n_reduction': final_n_reduction,
            'yearly_results': yearly_results,
            'adoption_histories': [agent.adoption_history.copy() for agent in self.agents]
        }
    
    def simulate_payment_level(self, policy_type: str, payment_level: float,
                                n_states: int = 60, n_years: int = 10) -> Dict:
        """
        Run full simulation for one payment level across all 60 states.
        
        Returns:
            Dictionary with aggregated results
        """
        all_state_results = []
        
        for state_idx in range(n_states):
            result = self.simulate_state(state_idx, policy_type, payment_level, n_years)
            all_state_results.append(result)
        
        # Aggregate results across all states
        # Average adoption rate across 60 states (final year)
        avg_adoption_rate = np.mean([r['final_adoption_rate'] for r in all_state_results])
        
        # Expected N reduction (average across 60 states)
        expected_n_reduction = np.mean([r['final_n_reduction'] for r in all_state_results])
        
        # Collect adoption histories per farmer (60 states × 10 years)
        farmer_adoption_matrices = []
        for agent_idx in range(len(self.agents)):
            # Matrix: 60 states × 10 years
            agent_matrix = np.zeros((n_states, n_years))
            for state_idx, result in enumerate(all_state_results):
                agent_matrix[state_idx, :] = result['adoption_histories'][agent_idx]
            farmer_adoption_matrices.append(agent_matrix)
        
        return {
            'policy_type': policy_type,
            'payment_level': payment_level,
            'avg_adoption_rate': avg_adoption_rate,
            'expected_n_reduction': expected_n_reduction,
            'state_results': all_state_results,
            'farmer_adoption_matrices': farmer_adoption_matrices
        }
    
    def run_policy_simulation(self, policy_type: str, 
                               payment_increment: float,
                               max_payment: float = None,
                               convergence_threshold: float = 0.01,
                               min_iterations: int = 5) -> List[Dict]:
        """
        Run simulation for a policy type, incrementing payment until convergence.
        
        Args:
            policy_type: 'results_based', 'area_based', or 'flat'
            payment_increment: Increment step for payment level
            max_payment: Maximum payment level (optional safety limit)
            convergence_threshold: Stop when marginal N reduction < this fraction
            min_iterations: Minimum number of payment levels before checking convergence
            
        Returns:
            List of results for each payment level
        """
        results = []
        payment_level = 0.0
        prev_n_reduction = 0.0
        
        # Set reasonable max limits if not specified
        if max_payment is None:
            if policy_type == 'results_based':
                max_payment = 50.0  # CAD/kg
            elif policy_type == 'area_based':
                max_payment = 500.0  # CAD/ha
            else:  # flat
                max_payment = 50000.0  # CAD/farm
        
        iteration = 0
        while payment_level <= max_payment:
            print(f"  Payment level: {payment_level:.2f}")
            
            result = self.simulate_payment_level(policy_type, payment_level)
            results.append(result)
            
            current_n_reduction = result['expected_n_reduction']
            
            # Check convergence (after minimum iterations and at least one non-zero result)
            if iteration >= min_iterations and prev_n_reduction > 0:
                marginal_increase = (current_n_reduction - prev_n_reduction) / prev_n_reduction
                if marginal_increase < convergence_threshold:
                    print(f"  Converged at payment level {payment_level:.2f} "
                          f"(marginal increase: {marginal_increase:.4f})")
                    break
            
            prev_n_reduction = current_n_reduction if current_n_reduction > 0 else prev_n_reduction
            payment_level += payment_increment
            iteration += 1
        
        return results


def run_full_simulation(data_path: str, output_path: str):
    """
    Run the complete simulation for all three policy types.
    
    Args:
        data_path: Path to input CSV file
        output_path: Path for output CSV file
    """
    print("Initializing model...")
    model = VRNAAdoptionModel(data_path)
    
    all_results = []
    
    # Policy configurations
    policies = [
        ('results_based', 0.5),   # CAD/kg, increment by 0.5
        ('area_based', 5.0),      # CAD/ha, increment by 5
        ('flat', 500.0)           # CAD/farm, increment by 500
    ]
    
    for policy_type, increment in policies:
        print(f"\nRunning simulation for {policy_type} policy...")
        policy_results = model.run_policy_simulation(policy_type, increment)
        
        for result in policy_results:
            all_results.append({
                'policy_type': result['policy_type'],
                'payment_level': result['payment_level'],
                'avg_adoption_rate': result['avg_adoption_rate'],
                'expected_n_reduction': result['expected_n_reduction']
            })
    
    # Save summary results
    summary_df = pd.DataFrame(all_results)
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary results saved to: {summary_path}")
    
    # Create detailed output with farmer-level adoption
    print("\nGenerating detailed farmer-level output...")
    generate_detailed_output(model, policies, output_path)
    
    return model, all_results


def generate_detailed_output(model: VRNAAdoptionModel, policies: List[Tuple], output_path: str):
    """
    Generate detailed CSV output with adoption status for each farmer.
    
    Output format:
    Farmer_ID | Policy | Payment_Level | Year_1 | Year_2 | ... | Year_10 | Avg_Nred
    """
    rows = []
    
    for policy_type, increment in policies:
        print(f"  Processing {policy_type}...")
        payment_level = 0.0
        prev_n_reduction = 0.0
        
        # Determine max payment
        if policy_type == 'results_based':
            max_payment = 50.0
        elif policy_type == 'area_based':
            max_payment = 500.0
        else:
            max_payment = 50000.0
        
        while payment_level <= max_payment:
            result = model.simulate_payment_level(policy_type, payment_level)
            
            # For each farmer, get their adoption history across all states
            for agent_idx, agent in enumerate(model.agents):
                # Average adoption across 60 states for each year
                adoption_matrix = result['farmer_adoption_matrices'][agent_idx]
                avg_adoptions_per_year = adoption_matrix.mean(axis=0)  # Average across states
                
                # Average N reduction if adopted
                avg_nred = 0.0
                for state_idx in range(60):
                    if adoption_matrix[state_idx, -1] == 1:  # Adopted in final year
                        avg_nred += agent.nred[state_idx]
                avg_nred /= 60  # Average across states
                
                row = {
                    'Farmer_ID': agent.farmer_id,
                    'Policy': policy_type,
                    'Payment_Level': payment_level
                }
                for year in range(10):
                    row[f'Year_{year + 1}'] = avg_adoptions_per_year[year]
                row['Avg_Nred'] = avg_nred
                
                rows.append(row)
            
            # Check convergence
            current_n_reduction = result['expected_n_reduction']
            if prev_n_reduction > 0:
                marginal_increase = (current_n_reduction - prev_n_reduction) / prev_n_reduction
                if marginal_increase < 0.01:
                    break
            
            prev_n_reduction = current_n_reduction if current_n_reduction > 0 else prev_n_reduction
            payment_level += increment
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    # This will be called from run.py
    pass
