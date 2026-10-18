# WiLoc theory: how each module maps to Perception & Localization

This note walks through the models in WiLoc in the order a localization course
usually builds them up: sensor (perception) model, map, motion model, then
Bayesian state estimation. Each section names the file that implements it.

---

## 1. Perception: the Wi-Fi RSSI sensor model (`wiloc/radio.py`)

A Wi-Fi scan returns, for every access point (AP) *a* the radio can hear, its
received signal strength indicator (RSSI) in dBm. WiLoc simulates it as

$$
\text{RSSI}_a(\mathbf{x}, t) = P_{tx,a} - \underbrace{\Big[PL(d_0) + 10\,n\log_{10}\tfrac{d}{d_0}\Big]}_{\text{log-distance}} - \underbrace{\sum_{k\in\text{walls}} L_k}_{\text{multi-wall}} + \underbrace{S_a(\mathbf{x})}_{\text{shadowing}} + \underbrace{\varepsilon_a(t)}_{\text{fast fading}}
$$

| Term | Model | Default | Reference |
|---|---|---|---|
| $PL(d_0)$ | Friis free-space loss at $d_0 = 1$ m, 2.437 GHz | 40.2 dB | Rappaport 2002 |
| $n$ | path-loss exponent (close to free space because walls are explicit) | 2.0 | Seidel & Rappaport 1992 |
| $L_k$ | per-wall penetration loss by material | concrete 12, brick 7, drywall 3.5, glass 2 dB | Motley & Keenan 1988; COST 231 |
| $S_a(\mathbf{x})$ | zero-mean Gaussian random field, $\mathrm{Cov} = \sigma_s^2 e^{-\lVert\mathbf{x}-\mathbf{x}'\rVert/d_c}$ | $\sigma_s = 5$ dB, $d_c = 3$ m | Gudmundson 1991 |
| $\varepsilon_a(t)$ | i.i.d. Gaussian per scan | $\sigma_t = 3$ dB | |
| quantisation and sensitivity | rounded to 1 dBm; below −95 dBm is *not heard* (NaN) | | |

**Why the shadowing field matters.** $S_a$ is drawn **once** per environment,
by circulant-embedding FFT sampling, and then stays fixed. It is the part of
the signal that geometry cannot predict but that repeats at the same place, so
it is exactly what a site survey can learn. The radio-map figure shows this:
the path-loss model alone has about 6.5 dB RMSE against the truth, while the
GP radio map learned from the survey has about 3.3 dB. If the shadowing were
redrawn for every scan, fingerprinting would collapse into trilateration.

## 2. Mapping: the offline site survey and radio map (`wiloc/fingerprint.py`)

* **Survey.** Reference points (RPs) on a 1 m grid in free space, 20 scans
  each, give a per-RP mean $\mu_{r,a}$, a standard deviation $\sigma_{r,a}$
  and a detection rate $p_{r,a}$. This is the fingerprint database, and is
  the same idea as the RADAR and Horus offline phases.
* **Gaussian-process radio map** (Ferris, Hähnel & Fox, RSS 2006). For each
  AP, the RSSI is modelled as

  $$ \text{RSSI}_a(\mathbf{x}) \sim \mathcal{GP}\big(m_a(\mathbf{x}),\; k(\mathbf{x},\mathbf{x}')\big), \quad m_a(\mathbf{x}) = A_a - 10 n_a\log_{10}\lVert\mathbf{x}-\mathbf{x}_a\rVert $$

  * Mean $m_a$: a log-distance model fitted to the survey by least squares.
  * Kernel $k$: RBF plus white noise, with hyper-parameters chosen by
    maximising the log marginal likelihood (ML-II, L-BFGS-B).
  * Prediction: the GP gives a continuous mean and a predictive variance
    anywhere on the floor. The variance is added to the temporal noise, so the
    particle filter knows where the map is uncertain.

## 3. Single-scan localization (`wiloc/estimators.py`)

| Estimator | Idea | Reference |
|---|---|---|
| NN / KNN / WKNN | nearest RPs in signal space ($L_2$ distance on dBm); WKNN weights them by $1/d$ | Bahl & Padmanabhan 2000 |
| Probabilistic | $\log p(\mathbf{z}\mid r)=\sum_a \log\mathcal{N}(z_a;\mu_{r,a},\sigma_{r,a}^2)$, with a missing AP scored by $1-p_{r,a}$; posterior mean over the top-$k$ RPs | Youssef & Agrawala 2005 |
| Trilateration | invert the fitted log-distance model to ranges, then weighted non-linear least squares | model-based baseline |

Each estimator also returns a covariance, the weighted spread of its
candidates, so its fix can be used as a measurement in a Kalman filter.

## 4. Motion: differential-drive kinematics and odometry (`wiloc/motion.py`)

* **Kinematics.** With wheel speeds $\omega_l, \omega_r$, wheel radius $r$ and
  track width $b$:

  $$ v = \frac{r(\omega_r + \omega_l)}{2}, \qquad \omega = \frac{r(\omega_r-\omega_l)}{b} $$

  Each step is integrated exactly along an arc.
* **Wheel odometry.** Encoder increments get random slip (5 % per step) and
  calibration errors (+1.5 % radius, +4 % track width), so dead reckoning drifts.
* **Odometry motion model** (Thrun, Burgard & Fox, *Probabilistic Robotics*,
  §5.4). Each increment is decomposed into $(\delta_{rot1}, \delta_{trans},
  \delta_{rot2})$ with noise

  $$ \hat\delta_{rot1}=\delta_{rot1}+\mathcal{N}\big(0,\ \alpha_1\delta_{rot1}^2+\alpha_2\delta_{trans}^2\big) $$

  and the same form for the other two terms. The model is sampled in the
  particle filter and linearised (Jacobians $G$, $V$) for the EKF.

## 5. Bayesian state estimation (`wiloc/filters.py`)

### Particle filter (Monte Carlo localization)

1. **Predict.** Sample every particle through the odometry motion model.
   * *Map constraint:* a particle whose move crosses a wall has its weight
     multiplied by $10^{-3}$.
   * Particles that leave the building get weight 0.
2. **Correct.** Multiply weights by $p(\mathbf{z}\mid\mathbf{x})^{\tau}$, using
   the GP map's $\mu_a(\mathbf{x})$ and $\sigma_a(\mathbf{x})$:
   * a heard AP contributes a robust mixture $(1-\epsilon)\,\mathcal{N}(z_a;\mu_a,\sigma_a^2) + \epsilon\,\mathcal{U}$;
   * an AP that was **not heard** is a censored observation, with
     probability $\Phi\big((s - \mu_a)/\sigma_a\big)$ where $s$ is the
     receiver sensitivity. The fact that an AP is missing is itself information.
   * $\tau = 0.25$ tempers the likelihood. AP readings are not independent:
     the GP-map error is spatially correlated, and so is shadowing. Without
     tempering the filter becomes overconfident. In a parameter sweep over two
     patrols, tracking from a known start was insensitive to $\tau$ (RMSE
     0.24–0.31 m). Global localization was not: with $\tau \ge 0.5$ the filter
     often locked onto a wrong mode early (RMSE 1.5–2.7 m, peaks of 4–8 m),
     while with $\tau \le 0.3$ it reached 0.26 m.
   * In the ROS node the update only runs after the robot has moved at least
     `update_min_d` = 0.1 m or `update_min_a` = 0.2 rad, as in AMCL. Scans from
     a standing robot share the same map error; applying all of them collapses
     the cloud onto one possibly wrong mode.
3. **Resample.** Low-variance (systematic) resampling, run when
   $N_\text{eff} < N/2$.
4. **Augmented MCL.** Short- and long-term averages of the likelihood
   ($w_\text{fast}$, $w_\text{slow}$) trigger random-particle injection, which
   handles global localization and the kidnapped-robot problem (Thrun et al.,
   Table 8.3).
5. **Estimate.** Mean-shift on the weighted particles gives the dominant
   mode. This is AMCL's "best cluster" in a simpler form.

### EKF (odometry + Wi-Fi position fixes)

* **Prediction.** $\mu' = g(\mu,u)$ and $\Sigma' = G\Sigma G^\top + VMV^\top$
  from the odometry model.
* **Update.** A WKNN fix $\mathbf{z}=(x,y)$ with $H=[I_2\;0]$, and $R$ taken
  from the estimator's covariance.
* **Outlier gating.** A fix whose Mahalanobis distance exceeds
  $\chi^2_{2,0.999}\approx 13.8$ is rejected. After 6 rejections in a row the
  covariance is re-opened, so a filter that has drifted can recover.

---

## 6. Experiment checklist for a course report

The experiments below are regenerated by `python -m wiloc all`.

1. **Sensor model.** `radio_maps.png`: true field vs. path-loss model vs. GP map.
2. **Fingerprinting accuracy.** `static_cdf.png`: error CDF of NN, KNN,
   WKNN, probabilistic and trilateration.
3. **Design trade-offs.** `sweeps.png`: number of APs, survey density
   (survey cost vs. accuracy), $k$, and temporal noise.
4. **Sensor fusion.** `tracking_*.png`: odometry only vs. Wi-Fi only vs.
   EKF vs. PF along a patrol.
5. **Global localization.** `global_localization.png`: PF from a uniform prior.
6. **Ablation.** PF using the GP map vs. using the parametric path-loss model.
