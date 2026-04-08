"""
Run two-stage CPT model with state-by-state N reduction output
For generating 95% confidence intervals like Huber et al. (2023)

Outputs N reduction for each of the 60 environmental states separately,
allowing calculation of mean and CI in R.
"""

import numpy as np
import pandas as pd
from model import VRNAAdoptionModel

def run_simulation_with_state_output(data_path: str, output_prefix: str):
    """
    Run simulation and output N reduction per environmental state.
    
    Output columns:
    - policy_type
    - payment_level
    - state_idx (0-59)
    - n_reduction (total N reduction for that state)
    - adoption_rate (adoption rate for that state)
    """
    
    print("Initializing two-stage CPT model...")
    model = VRNAAdoptionModel(data_path, skip_stage1=False, profit_max_stage2=False)
    
    # Policy configurations
    policies = [
        ('results_based', 0.5, 10.0),   # (type, increment, max)
        ('area_based', 5.0, 100.0),     
        ('flat', 1000.0, 20000.0)       
    ]
    
    # Store results
    state_level_rows = []  # For CI calculation
    summary_rows = []       # Overall summary
    
    for policy_type, increment, max_payment in policies:
        print(f"\n{'='*50}")
        print(f"Running {policy_type} policy")
        print(f"{'='*50}")
        
        payment_level = 0.0
        prev_n_reduction = 0.0
        iteration = 0
        min_iterations = 5
        
        while payment_level <= max_payment:
            print(f"  Payment level: {payment_level:.1f}")
            
            # Run simulation for this payment level (all 60 states)
            result = model.simulate_payment_level(policy_type, payment_level)
            
            # Extract state-level results
            state_n_reductions = []
            state_adoption_rates = []
            
            for state_idx, state_result in enumerate(result['state_results']):
                n_red = state_result['final_n_reduction']
                adopt_rate = state_result['final_adoption_rate']
                
                state_n_reductions.append(n_red)
                state_adoption_rates.append(adopt_rate)
                
                # Store state-level data for CI calculation
                state_level_rows.append({
                    'policy_type': policy_type,
                    'payment_level': payment_level,
                    'state_idx': state_idx,
                    'n_reduction': n_red,
                    'adoption_rate': adopt_rate
                })
            
            # Calculate summary statistics
            mean_n_red = np.mean(state_n_reductions)
            std_n_red = np.std(state_n_reductions, ddof=1)
            se_n_red = std_n_red / np.sqrt(60)
            ci_lower = mean_n_red - 1.96 * se_n_red
            ci_upper = mean_n_red + 1.96 * se_n_red
            
            mean_adopt = np.mean(state_adoption_rates)
            std_adopt = np.std(state_adoption_rates, ddof=1)
            
            summary_rows.append({
                'policy_type': policy_type,
                'payment_level': payment_level,
                'mean_n_reduction': mean_n_red,
                'std_n_reduction': std_n_red,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean_adoption_rate': mean_adopt,
                'std_adoption_rate': std_adopt
            })
            
            print(f"    Mean N reduction: {mean_n_red/1000:.1f} t (±{1.96*se_n_red/1000:.1f})")
            print(f"    Adoption rate: {mean_adopt*100:.1f}%")
            
            # Check convergence
            if iteration >= min_iterations and prev_n_reduction > 0:
                marginal_increase = (mean_n_red - prev_n_reduction) / prev_n_reduction
                if marginal_increase < 0.01:
                    print(f"  Converged at payment level {payment_level:.1f}")
                    break
            
            prev_n_reduction = mean_n_red if mean_n_red > 0 else prev_n_reduction
            payment_level += increment
            iteration += 1
    
    # Save state-level results (for R CI calculation)
    state_df = pd.DataFrame(state_level_rows)
    state_path = f"{output_prefix}_by_state.csv"
    state_df.to_csv(state_path, index=False)
    print(f"\nState-level results saved to: {state_path}")
    
    # Save summary with CI
    summary_df = pd.DataFrame(summary_rows)
    summary_path = f"{output_prefix}_summary_with_CI.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary with CI saved to: {summary_path}")
    
    return state_df, summary_df


if __name__ == "__main__":
    data_path = "repast_agents_input_with_GM60_Benefit60_Nred60.csv"
    output_prefix = "Final_csv/two_stages_original"
    
    state_df, summary_df = run_simulation_with_state_output(data_path, output_prefix)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print(f"  1. {output_prefix}_by_state.csv - State-level data for CI calculation")
    print(f"  2. {output_prefix}_summary_with_CI.csv - Summary with 95% CI")
    print("\nUse the _by_state.csv file in R to create Huber et al. style plots")
