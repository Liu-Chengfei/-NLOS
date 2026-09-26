# Significance Tests Fix Summary

## Problem Description
Significance tests were failing due to incorrect handling of BCa Bootstrap confidence intervals. The tests were:
1. Getting null values for CI bounds when all bootstrap differences were identical
2. Accidentally using the wrong test name in post-hoc processing

## Root Cause Analysis
1. **_bootstrap_mean_difference_ci function**: When all bootstrap differences were identical, the standard deviation became zero, causing division by zero in BCa acceleration calculation
2. **run_significance_tests function**: Post-hoc processing incorrectly accessed group comparison data due to wrong indexing logic

## Fixes Applied

### 1. Fixed `_bootstrap_mean_difference_ci` in `src/liquidloc/analysis/significance_tests.py`
- **Lines 183-184**: Added safety check for zero standard deviation in BCa calculation
- **Lines 214-215**: Added fallback to percentile method when BCa denominators are near-zero
- **Lines 191-192**: Added handling for empty bootstrap samples

### 2. Fixed `run_significance_tests` in `src/liquidloc/analysis/significance_tests.py`
- **Lines 307-313**: Rewrote post-hoc group comparison logic to correctly access pairwise results
- **Line 317**: Added safety check for missing 'observed' key before logging

## Verification
- All 99 analysis tests pass
- BCa bootstrap now returns valid CI bounds even in edge cases
- Post-hoc processing correctly identifies group pairs
- Method names and CI methods are correctly reported

## Edge Cases Handled
1. Identical bootstrap samples (zero variance)
2. Empty or insufficient bootstrap samples
3. Near-zero BCa denominators
4. Post-hoc analysis with single group comparisons

The fixes ensure statistical robustness while maintaining backward compatibility with existing API contracts.