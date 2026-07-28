# Home Advantage in African Football (2010–2025)

**Quantifying Home Advantage in African Football Competitions Using Statistical Data Analysis: Evidence from the CAF Champions League and AFCON (2010–2025)**

> Ezeobi Henry Chinedu  
> Department of Computer Engineering, University of Uyo, Nigeria  
> ezeobihenry333@gmail.com

---

## Overview

This repository contains the dataset and analysis code accompanying the above paper submitted to an IEEE conference. The study quantifies home advantage (HA) in 812 non-neutral fixtures from the CAF Champions League (CAF CL, 766 matches) and AFCON (46 matches) spanning 2010–2025, using standard statistical methods including Poisson rate ratio estimation, binary logistic regression, OLS trend analysis, and chi-square tests.

**Key findings:**
- HA% = 68.1% (CAF CL) and 68.6% (AFCON), substantially above the ~60% European benchmark
- North Africa records the highest regional HA (72.8%); West Africa the lowest (61.3%)
- Home teams score 85.5% more goals per match than away teams in the CAF CL (γ̂ = 0.618)
- Temporal trend is non-significant (β = −0.213%/yr, p = 0.398), indicating HA stability over the study period

---

## Repository Contents

```
home-advantage-african-football/
├── Home_Advantage_Enriched.csv   # Full dataset (812 non-neutral fixtures)
├── analysis.py                   # Python script reproducing all tables and Fig. 2
├── README.md                     # This file
```

---

## Dataset Description

**File:** `Home_Advantage_Enriched.csv`  
**Rows:** 812 (non-neutral fixtures only)  
**Sources:** FBRef (AFCON dataset); cross-referenced with FotMob, SofaScore, and Wikipedia (CAF CL dataset)

| Column | Description |
|---|---|
| `Date` | Match date (YYYY-MM-DD) |
| `Season` | Season label (e.g., 2018, 2018/19) |
| `Competition` | CAF Champions League or African Cup of Nations |
| `Stage` | Competition stage (Group Stage, Quarter-final, etc.) |
| `Home Team` | Home team name |
| `Away Team` | Away team name |
| `Home Goals` | Goals scored by home team |
| `Away Goals` | Goals scored by away team |
| `Neutral` | Whether fixture was played at a neutral venue (all False — neutral fixtures excluded) |
| `Match Result` | Home Win / Draw / Away Win |
| `Home Team Country` | Home team's country |
| `Away Team Country` | Away team's country |
| `Home Team Region` | CAF region of the home team |
| `Away Team Region` | CAF region of the away team |
| `Cross Region Match` | Whether the teams come from different CAF regions |
| `Venue Altitude (m)` | Elevation of the match venue (metres) |
| `Away Team Altitude (m)` | Elevation of the away team's home city (metres) |
| `Altitude Difference (m)` | Venue altitude minus away team home altitude |
| `Travel Distance (km)` | Great-circle distance from away team city to match venue |
| `Climate Zone` | Köppen climate classification of the match venue |

---

## Reproducing the Results

### Requirements

```bash
pip install pandas numpy scipy statsmodels matplotlib
```

Tested on Python 3.9+.

### Running the Script

```bash
python analysis.py
```

This prints all table values to stdout and saves `Figure_2_Temporal_Trend.png` in the working directory.

### What the Script Reproduces

| Output | Paper Location |
|---|---|
| Home Advantage % by competition | Table I |
| Home Advantage % by region | Table II |
| Logistic regression coefficients | Table III |
| Poisson rate ratio estimates | Table IV |
| OLS temporal trend chart | Figure 2 |
| Chi-square test statistic | Section IV.A |

### Notes on Methodology

- **Table II regional n-values** are computed over all 812 fixtures (CAF CL + AFCON combined), matching the paper. The logistic regression (Table III) uses only CAF CL fixtures (n = 766) because travel distance cannot be defined under AFCON's single-host structure.
- **OLS slope** (β = −0.213%/yr, p = 0.398) is obtained by regressing five 3-year period HA% values on their period mid-years using all 812 fixtures.
- **Poisson rate ratio** is a descriptive estimator (log of sample mean ratio), not a regression-based parameter. It quantifies the multiplicative goal-scoring advantage of home teams.
- **McFadden R² = 0.019** in the logistic model reflects the inherent difficulty of predicting binary match outcomes from environmental covariates alone. The primary inferential interest lies in the regional coefficient estimates, whose significance is robust.

---

## Citation

If you use this dataset or code, please cite:

> H. C. Ezeobi, "Quantifying Home Advantage in African Football Competitions Using Statistical Data Analysis: Evidence from the CAF Champions League and AFCON (2010–2025)," *IEEE Conference Proceedings*, 2025.

---

## License

This repository is made available for academic reproducibility purposes. The dataset was compiled from publicly available sources (FBRef, FotMob, SofaScore, Wikipedia). Please cite appropriately if used in derivative work.
