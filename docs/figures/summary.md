# WiLoc results

Reference points: 240, radio map RMSE: GP 3.29 dB vs path-loss model 6.46 dB

## Single-scan estimators
| Method | mean [m] | median [m] | rmse [m] | p75 [m] | p90 [m] |
|---|---:|---:|---:|---:|---:|
| NN | 1.22 | 0.77 | 1.75 | 1.70 | 2.95 |
| KNN (k=4) | 1.29 | 1.12 | 1.56 | 1.70 | 2.37 |
| WKNN (k=4) | 1.21 | 1.05 | 1.48 | 1.59 | 2.21 |
| Probabilistic | 1.11 | 0.78 | 1.51 | 1.46 | 2.43 |
| Trilateration | 2.48 | 2.08 | 2.97 | 3.53 | 4.81 |

## Tracking (odometry + Wi-Fi)
| Method | mean [m] | median [m] | rmse [m] | p75 [m] | p90 [m] |
|---|---:|---:|---:|---:|---:|
| Odometry only | 2.65 | 2.01 | 3.38 | 4.11 | 6.35 |
| WKNN (k=4) (per scan) | 1.14 | 0.82 | 1.69 | 1.34 | 2.16 |
| EKF (odom + WKNN) | 0.34 | 0.34 | 0.37 | 0.44 | 0.52 |
| Particle filter | 0.24 | 0.22 | 0.27 | 0.32 | 0.43 |

## Particle-filter sensor model
| Method | mean [m] | median [m] | rmse [m] | p75 [m] | p90 [m] |
|---|---:|---:|---:|---:|---:|
| log-distance fit | 0.45 | 0.39 | 0.52 | 0.67 | 0.83 |
| GP radio map | 0.22 | 0.22 | 0.25 | 0.30 | 0.37 |

## Global localisation
```
{
  "median_settling_time_s": 0.5,
  "settling_threshold_m": 1.5,
  "converged_runs": 10,
  "runs": 10,
  "final_median_error_m": 0.21705955897616064
}
```
