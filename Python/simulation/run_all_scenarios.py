"""
Run script for the VRNA adoption ABM - all model specifications, one fixed payment grid.

Produces every result file used in the manuscript from a single, consistent run:
  1. results_summary.csv        one row per specification x instrument x payment level
  2. results_by_state.csv       same, for each of the 60 environment states
  3. results_farmer_level.csv   farmer-level adoption (full behavioural model only)

Model specifications (manuscript Section 4.3):
  Full behavioural model     : Stage 1 + CPT certainty-equivalent rule
  Without Stage 1            : all agents optimise every year + CE rule
  Stage 1 + profit max       : Stage 1 + state-specific profit comparison
  Pure profit maximisation   : no Stage 1 + state-specific profit comparison

Payment grid (fixed, no stopping rule; manuscript Section 3.4 / ODD A.9):
  results-based  0-10 CAD/kg N   in steps of 0.5
  area-based     0-100 CAD/ha    in steps of 5
  flat           0-20,000 CAD/farm in steps of 1,000
"""

import os
import time
import numpy as np
import pandas as pd
from model import VRNAAdoptionModel

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "repast_agents_input_with_GM60_Benefit60_Nred60.csv")
OUT_SUMMARY = os.path.join(DATA_DIR, "results_summary.csv")
OUT_BY_STATE = os.path.join(DATA_DIR, "results_by_state.csv")
OUT_FARMER = os.path.join(DATA_DIR, "results_farmer_level.csv")

# ---------------------------------------------------------------------------
# FILL IN from manuscript Table 5 (same values as used in the bioeconomic model)
# ---------------------------------------------------------------------------
PRICE_PARAMS = {
    "n_uniform_kg_ha": 109.0,  # kg N/ha, baseline uniform rate
    "n_price_base": 1.35,      # CAD per kg N in the base year (2025)
    "n_price_growth": 0.05,    # annual growth rate (5%)
    "opcost_base": 499.0,      # CAD per ha in the base year
    "opcost_growth": 0.04,     # annual growth rate (4%)
}

REFERENCE_N_KG = 109.0 * 507588  # reference N use (uniform rate on all canola area), kg

SCENARIOS = {
    "Full behavioural model": dict(skip_stage1=False, profit_max_stage2=False),
    "Without Stage 1": dict(skip_stage1=True, profit_max_stage2=False),
    "Stage 1 + profit max": dict(skip_stage1=False, profit_max_stage2=True),
    "Pure profit maximisation": dict(skip_stage1=True, profit_max_stage2=True),
}

PAYMENT_GRID = {
    "results_based": np.arange(0, 10.0 + 1e-9, 0.5),
    "area_based": np.arange(0, 100.0 + 1e-9, 5.0),
    "flat": np.arange(0, 20000.0 + 1e-9, 1000.0),
}


def policy_cost(policy, payment, n_red_kg, ha_vrna, adopters):
    """Annual public cost (CAD) of an instrument, paid to ALL adopters."""
    if policy == "results_based":
        return payment * n_red_kg
    if policy == "area_based":
        return payment * ha_vrna
    return payment * adopters


def main():
    summary_rows, state_rows, farmer_rows = [], [], []
    t0 = time.time()

    for scen_name, flags in SCENARIOS.items():
        print(f"\n=== {scen_name} ===")
        model = VRNAAdoptionModel(INPUT_FILE, price_params=PRICE_PARAMS, **flags)
        zero = None  # zero-payment results of this specification

        for policy, grid in PAYMENT_GRID.items():
            for payment in grid:
                res = model.simulate_payment_level(policy, float(payment))
                n_red = res["expected_n_reduction"]
                ha = res["expected_ha_vrna"]
                adopters = res["expected_adopters"]
                cost = policy_cost(policy, payment, n_red, ha, adopters)
                if payment == 0:
                    zero = dict(n_red=n_red, ha=ha, adopters=adopters)
                add_kg = n_red - zero["n_red"]

                # Share of expenditure going to zero-payment adopters
                if payment == 0 or cost == 0:
                    infra = np.nan
                elif policy == "results_based":
                    infra = zero["n_red"] / n_red
                elif policy == "area_based":
                    infra = zero["ha"] / ha
                else:
                    infra = zero["adopters"] / adopters

                summary_rows.append({
                    "scenario": scen_name, "policy_type": policy, "payment_level": payment,
                    "adoption_rate": res["avg_adoption_rate"],
                    "adopters": adopters,
                    "n_reduction_t": n_red / 1000,
                    "additional_n_reduction_t": add_kg / 1000,
                    "pct_of_reference_N": 100 * n_red / REFERENCE_N_KG,
                    "ha_vrna": ha,
                    "annual_cost_M_CAD": cost / 1e6,
                    "average_cost_CAD_per_kg": cost / n_red if n_red > 0 else np.nan,
                    "cost_per_additional_kg": cost / add_kg if add_kg > 0 else np.nan,
                    "share_to_zero_payment_adopters": infra,
                })

                for r in res["state_results"]:
                    state_rows.append({
                        "scenario": scen_name, "policy_type": policy, "payment_level": payment,
                        "state_idx": r["state_idx"],
                        "adoption_rate": r["final_adoption_rate"],
                        "n_reduction_kg": r["final_n_reduction"],
                        "ha_vrna": r["final_ha_vrna"],
                        "adopters": r["final_adopters"],
                    })

                if scen_name == "Full behavioural model":
                    for agent_idx, agent in enumerate(model.agents):
                        m = res["farmer_adoption_matrices"][agent_idx]  # 60 states x 10 years
                        row = {"Farmer_ID": agent.farmer_id, "farm_size_ha": agent.farm_size_ha,
                               "policy_type": policy, "payment_level": payment}
                        for y in range(m.shape[1]):
                            row[f"Year_{y + 1}"] = m[:, y].mean()
                        row["Avg_Nred_kg"] = float(np.mean(np.where(m[:, -1] == 1, agent.nred, 0.0)))
                        farmer_rows.append(row)

                print(f"  {policy:14s} {payment:>8.1f}  adoption {res['avg_adoption_rate']:.3f}  "
                      f"N red {n_red / 1000:,.0f} t  ({time.time() - t0:,.0f} s)")

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(state_rows).to_csv(OUT_BY_STATE, index=False)
    pd.DataFrame(farmer_rows).to_csv(OUT_FARMER, index=False)
    print(f"\nSaved:\n  {OUT_SUMMARY}\n  {OUT_BY_STATE}\n  {OUT_FARMER}")


if __name__ == "__main__":
    main()
