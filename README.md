# Agent-Based Model for VRNA Adoption in Canadian Canola Farms

An agent-based model (ABM) simulating farmer decision-making regarding the adoption of **Variable Rate Nitrogen Application (VRNA)** technology on Canadian Prairie canola farms. The model integrates the **FARMIND** decision-making framework with **Cumulative Prospect Theory (CPT)** to evaluate how alternative policy instruments affect adoption, nitrogen reduction, and policy cost-efficiency.

## Overview

The model couples a **bioeconomic pre-processing module** (R) with a **behavioural simulation engine** (Python) to evaluate three policy instruments:

- **Results-based payment** — CAD per kg of nitrogen reduced  
- **Area-based subsidy** — CAD per hectare under VRNA  
- **Flat payment** — fixed CAD per farm per year  

Each instrument is evaluated across **60 environment states** (12 landscape templates × 5 canola price draws), generating distributions of adoption and nitrogen-reduction outcomes under heterogeneous farm conditions.

### Key Features

- **396 empirically parameterised farmer agents** initialised from a 2023 survey of Prairie canola growers  
- **Two-stage CPT decision process**: Stage 1 (satisfaction/reluctance filter) → Stage 2 (prospect-theoretic adoption evaluation)  
- **Four behavioural scenarios**: Full behavioural model, without reluctance factor, without certainty equivalent, and pure profit maximisation  
- **Global sensitivity analysis**: Partial Rank Correlation Coefficients (PRCC) and Standardised Regression Coefficients (SRC) via Latin Hypercube Sampling  

## Repository Structure

```
├── Python/
│   ├── simulation/                    # Core ABM simulation
│   │   ├── model.py                   # FarmerAgent, CPTCalculator, VRNAAdoptionModel
│   │   ├── run_two_stages_with_CI.py  # Main simulation runner (full behavioural model)
│   │   ├── run_comparison_fig5.py     # Run all 4 behavioural scenarios
│   │   ├── run_no_stage1_profit_max.py
│   │   └── run_two_stages_profit_max.py
│   ├── sensitivity_analysis/          # Sensitivity analysis scripts
│   │   ├── sensitivity_model_interface.py  # SA ↔ ABM interface wrapper
│   │   ├── PRCC_Sensitivity_Analysis_Python.py  # PRCC analysis (LHS, 1000 samples)
│   │   └── PRCC_SRC_ByTemplate.py     # Per-template PRCC/SRC analysis
│   └── analysis/                      # Post-processing and analysis
│       ├── run_farm_size_analysis.py   # Adoption by farm-size category
│       ├── calc_nred_by_size.py        # N reduction by farm-size category
│       └── cost_small_farms_v2.py      # Cost-efficiency calculations
├── R/
│   ├── bioeconomic/
│   │   └── Bioeconomic.R              # Bioeconomic pre-processing model
│   └── analysis/
│       └── Final_analysis.R           # Results visualisation and tables
├── ODD_Protocol.md                    # ODD+D protocol documentation
└── README.md
```

## Model Components

### 1. Bioeconomic Model (`R/bioeconomic/Bioeconomic.R`)

Pre-processing module that generates agent-level economic outcomes:

- **Yield response**: Quadratic function $Y = b_0 + b_1 N - b_2 N^2$ for three productivity zones (Low, Medium, High)  
- **Nitrogen optimisation**: Zone-specific optimal N rates under VRNA vs. uniform 108 kg/ha baseline  
- **Gross margin calculation**: Incorporating canola price uncertainty (5 draws from U[550, 685] CAD/t), N cost (1.35 CAD/kg), operating cost (499 CAD/ha), and VRNA technology cost  
- **12 landscape templates**: 3 productivity compositions × 4 spatial configurations  
- **Output**: 60-state arrays of gross margins, benefits, and nitrogen reduction per farmer  

### 2. ABM Simulation Engine (`Python/simulation/model.py`)

Core simulation with three main classes:

- **`FarmerAgent`** — Dataclass holding farm characteristics, CPT parameters, and 60-state outcome arrays  
- **`CPTCalculator`** — Static methods for value function, probability weighting, decision weights, prospect value, and certainty equivalent  
- **`VRNAAdoptionModel`** — Orchestrates policy simulations with iterative payment-level sweeps and convergence checking  

**Decision process:**
1. **Stage 1 (Reluctance filter):** Farmer checks if baseline expected gross margin exceeds reference value; if satisfied, remains non-adopter  
2. **Stage 2 (CPT evaluation):** Computes prospect value of VRNA vs. baseline across 60 states using CPT value function and probability weighting; adopts if CE(VRNA) > CE(baseline)  

### 3. Sensitivity Analysis (`Python/sensitivity_analysis/`)

Global sensitivity analysis using Latin Hypercube Sampling with 1,000 samples:

- **8 parameters varied**: reference value ($V_{ref}$), risk sensitivity ($\alpha$, $\beta$), loss aversion ($\lambda$), probability weighting ($\gamma_{gain}$, $\gamma_{loss}$), farm size, and canola price  
- **PRCC**: Partial Rank Correlation Coefficients for nonlinear monotonic relationships  
- **SRC**: Standardised Regression Coefficients for linear importance  
- **Two outputs**: Adoption rate and total nitrogen reduction  

## Requirements

### Python
- Python ≥ 3.10
- NumPy, Pandas, SciPy, Matplotlib, Seaborn

### R
- R ≥ 4.0
- Packages: `dplyr`, `tidyr`, `ggplot2`, `scales`, `gridExtra`, `lpSolve`, `readr`

## Usage

### Step 1: Generate bioeconomic input data (R)

```r
source("R/bioeconomic/Bioeconomic.R")
```

This produces the agent input CSV with 60-state gross margin, benefit, and nitrogen reduction arrays.

### Step 2: Run ABM simulation (Python)

```bash
cd Python/simulation
python run_two_stages_with_CI.py
```

For all four behavioural scenarios:

```bash
python run_comparison_fig5.py
```

### Step 3: Sensitivity analysis (Python)

```bash
cd Python/sensitivity_analysis
python PRCC_Sensitivity_Analysis_Python.py
```

### Step 4: Visualise results (R)

```r
source("R/analysis/Final_analysis.R")
```

## Input Data

The model requires a CSV file with farmer-level data containing:

| Column(s) | Description | Source |
|-----------|-------------|--------|
| `Farmer_ID` | Unique identifier | Census |
| `Farm_Size_ha` | Farm size (hectares) | Census |
| `alpha`, `beta_val`, `lambda` | CPT risk/loss parameters | Survey (2023) |
| `gamma_gain`, `gamma_loss` | Probability weighting parameters | Survey (2023) |
| `Vref` | Reference gross margin | Bioeconomic model |
| `GM_VRNA_1`–`GM_VRNA_60` | Gross margins under VRNA | Bioeconomic model |
| `GM_Baseline_1`–`GM_Baseline_60` | Gross margins under baseline | Bioeconomic model |
| `Benefit_1`–`Benefit_60` | Incremental payoff (VRNA − baseline) | Bioeconomic model |
| `Nred_1`–`Nred_60` | Nitrogen reduction (kg N) | Bioeconomic model |

> **Note:** The raw survey/census data are not included in this repository due to confidentiality. The bioeconomic script (`Bioeconomic.R`) documents the data generation process.

## Citation

If you use this model in your research, please cite:

> Pezeshkian, Z. (2026). Agent-based model for evaluating policy instruments to promote variable rate nitrogen application adoption in Canadian canola farms. *[Journal TBD]*.

## License

This project is provided for academic and research purposes.
