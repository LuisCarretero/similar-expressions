# Pareto Volume Statistical Power Analysis for WandB Sweeps

## Executive Summary

Analysis of existing neural vs vanilla SR benchmarking data to determine optimal sweep configuration for hyperparameter optimization. Key finding: **current 10 runs per equation is already quite good**, with 15-20 runs providing optimal balance.

## Key Findings

### 1. Variance Characteristics
- **Average CV across equations: 0.172** (acceptable stability)
- **Median CV: 0.082** (very stable for most equations)
- **20% of equations have CV > 0.2** (some problematic variance)
- **Mean estimator is 23% more precise than median** (contrary to initial hypothesis)

### 2. Current Performance Assessment
**With existing 10 runs per equation:**
- **Margin of error: ±4.9%** for typical equations (median CV)
- **Margin of error: ±15.0%** averaged across all equations
- **✅ Acceptable uncertainty for hyperparameter optimization**

### 3. Equation Selection Strategy
**Investigated bias in stability-based selection:**
- **No complexity bias**: Stable equations span full complexity range (r = -0.041)
- **⚠️ Significant noise bias**: Stable equations tend to be lower noise (p < 0.0001)
- **⚠️ Performance bias**: Stable equations have higher pareto volumes (r = -0.319)
- **Risk**: Selecting only stable equations would overfit to "easier" problems

## Recommendations

### Optimal Sweep Configuration

**Dataset Selection:**
- **Use first 50 equations** from pysr-univariate dataset
- Provides good diversity without excessive computation
- Avoids selection bias from cherry-picking stable equations

**Sample Size:**
- **15-20 runs per equation** (sweet spot)
- **Total: 750-1,000 runs per hyperparameter configuration**
- **Expected margin of error: ±8-10%** (excellent for hyperparameter optimization)

**Aggregation Method:**
- **Simple mean across all equations** (standard ML practice)
- Avoid weighted aggregation (not standard in hyperparameter sweeps)
- Report confidence intervals for sweep convergence assessment

### Statistical Power Analysis Results

| Metric | Current (10 runs) | Recommended (15-20 runs) |
|--------|-------------------|--------------------------|
| Margin of Error | ±4.9% (median CV) | ±4-6% (median CV) |
| Total Runs per Config | 500 | 750-1,000 |
| Computational Cost | Baseline | +50-100% |
| Statistical Reliability | Good | Excellent |

### Sample Size Requirements by Confidence Level

**For first 50 equations (95% confidence, 10% margin of error):**
- **Conservative estimate**: 132 runs per equation (overkill)
- **Typical estimate**: 3 runs per equation (too optimistic)
- **Current practice**: 10 runs per equation (good)
- **Recommended**: 15-20 runs per equation (optimal)

## Implementation Notes

### WandB Sweep Metric
```python
# Standard approach - simple mean across equations
sweep_metric = np.mean([eq_pareto_volume_mean for eq in first_50_equations])
wandb.log({"sweep_pareto_volume": sweep_metric})
```

### Equation Selection
```python
# Use first n equations (no selection bias)
selected_equations = list(range(50))  # equations 0-49
```

### Statistical Validation
- Include confidence intervals in sweep results
- Monitor sweep convergence using CI width
- Report both individual equation performance and aggregated metrics

## Files Generated

1. `pareto_volume_power_analysis.py` - Main statistical analysis script
2. `equation_variance_stats.csv` - Per-equation variance metrics
3. `pooling_strategy_comparison.csv` - Mean vs median comparison
4. `statistical_power_analysis.csv` - Power analysis results
5. `equation_stability_ranking.csv` - Equation stability rankings
6. `sample_size_recommendations.csv` - Sample size requirements
7. `investigate_equation_properties.py` - Bias analysis script
8. `first_n_equations_analysis.py` - First-n equations analysis
9. `pareto_volume_power_analysis.png` - Summary visualizations

## Conclusion

The analysis reveals that your current approach is already quite effective. The recommended enhancement to 15-20 runs per equation with first 50 equations will provide robust, unbiased hyperparameter optimization while maintaining computational feasibility. This follows standard ML practices and avoids overfitting to specific equation properties.