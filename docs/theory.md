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
5. **Estimate.** The weighted mean of the heaviest particle cluster (section
   7.3), AMCL's "best cluster". Mean-shift is still available
   (`estimator:=meanshift`).

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
7. **Venues.** `venues.png` / `venues.md` (`python -m wiloc venues`): six
   venues × their router layouts × PF / EKF / WKNN, with people and vehicles
   blocking Wi-Fi and one kidnap per run.
8. **Router outage.** `outage.json` (`python -m wiloc outage`): each method
   with and without the outage detector.
9. **Particle-filter options.** `python -m wiloc filters`: KLD sampling,
   cluster estimate, re-seeding and sensor resetting under repeated kidnaps.

---

## 7. Techniques behind the live demo

### 7.1 Path planning (`wiloc/planning.py`)

The patrol and click-to-go use A* on an occupancy grid (0.1 m cells) built
from the walls and furniture. The robot's radius (0.27 m) inflates the
obstacles. A clearance cost from a Euclidean distance transform keeps paths in
the middle of doors and corridors. Line-of-sight smoothing then leaves a few
vertices for the path follower. A click on an obstacle snaps to the nearest
free cell. The driver replans when the pose jumps (a kidnap) and backs up when
it is stuck.

### 7.2 KLD sampling and sensor resetting (`wiloc/filters.py`)

* **KLD sampling** (Fox, 2003) sizes the particle set while resampling. It
  draws particles until their number exceeds the bound for the occupied
  histogram bins (0.5 m × 0.5 m × 20°):
  $n = \frac{k-1}{2\varepsilon}\big(1-\frac{2}{9(k-1)}+\sqrt{\frac{2}{9(k-1)}}\,z_{1-\delta}\big)^3$,
  where $k$ is the number of occupied bins, between 500 and 5000.
  * A confident filter needs a few hundred particles.
  * After a kidnap, injected particles occupy many bins and the set grows
    towards 5000.
* **Sensor resetting** (Lenser & Veloso, 2000) places half of the injected
  particles around the current WKNN fix, with any heading, instead of
  uniformly. Uniform injection rarely lands near the true position in a large
  building; the fix points there at once.

Over 216 simulated kidnaps, all options together improved:

| Metric | Before | After |
|---|---|---|
| p90 recovery time | 23 s | 8 s |
| Tracking error with kidnaps | 0.64 m | 0.37 m |
| Compute per update | 8.6 ms | 5.2 ms |

### 7.3 Mode clustering (`ParticleFilter.modes`)

Particles are binned on a 0.5 m grid. Touching occupied bins (8-connected,
`scipy.ndimage.label`) form clusters, and each cluster's share of the weight
is its probability.
* The estimate is the heaviest cluster's mean. A global mean of two clusters
  can land inside a wall between them.
* The top clusters drive the Studio's "Not sure: Room 3 (55 %) or Room 11
  (35 %)" caption.
* Clusters within 1.5 m of a heavier one are merged first
  (`wiloc/story.py: merge_modes`), so two sides of a doorway count as one
  place.

### 7.4 Moving bodies that block Wi-Fi (`wiloc/movers.py`)

People, forklifts, AGVs, gantries and carts follow scripted routes; Gazebo
plays the same routes as actors. A mover at distance $d$ from the robot–router
line weakens that router's signal by

$$L(d) = L_\text{type}\,e^{-(d/r_\text{type})^2},$$

where $L$ is about 4 dB for a person, 8 dB for an AGV and 12 dB for a
forklift. This loss is continuous and symmetric in robot and router.
*Noise zones* (welding bays, motors) add temporal noise in quadrature. The
localiser is **not** told about the movers: they test robustness. In the
office, the particle filter's median error went from 0.27 m with bodies off to
0.30 m with them on.

### 7.5 Router outage (`wiloc/outage.py`)

A router that falls silent where the map predicts it should be loud, with
$\mu_a - 3\sigma_a$ above the sensitivity at the estimate for 4 scans in a
row, is masked: the filters and WKNN ignore it until it is heard again. Masks
are added only while the estimate is confident (spread ≤ 1.5 m), so a lost
robot does not mask routers wrongly. One router switched off over 24 runs:

| Method | Error without the mask | Error with the mask |
|---|---|---|
| WKNN | 4.45 m | 0.83 m |
| EKF | 4.68 m | 0.32 m |
| Particle filter | 0.27 m | 0.27 m |

The outage was detected in 24 of 24 runs, after a median of 1.6 s. The
particle filter's censored "not heard" likelihood is already robust; WKNN and
the EKF need the mask.

### 7.6 Venues and router layouts (`wiloc/data/venues/*.yaml`)

Each venue is a YAML file holding:
* walls with materials (concrete, brick, glass, wood, metal racks)
* furniture, rooms and movers
* a patrol and an overview camera pose
* several router layouts

A layout switches live without restarting Gazebo. The localiser swaps in that
layout's cached radio map and keeps tracking from its current estimate. The
venue table (section 6, item 7) shows where each method works and why. It
covers 18 venue/layout pairs, 3 runs each, with people and vehicles blocking
Wi-Fi.

* **Wi-Fi alone (WKNN) depends on the routers.** Its median error is 0.9 m
  under a dense ceiling grid (library, 9 routers) and rises to 2–6 m with 1–2
  routers. With one router at the library's issue desk it reaches 9.3 m.
* **The particle filter barely depends on them.** Its median error stays
  between 0.20 and 0.50 m in 17 of 18 cases, including one router in a flat
  (0.26 m) and in the library (0.38 m). The wheel odometry and the floor plan
  carry it between the few informative scans.
* **The few-router layouts expose the limit: finding itself again.**
  * With the routers as installed, the particle filter re-finds a kidnapped
    robot in a median of 2.5–10 s.
  * With 1–2 routers it takes 14–43 s or fails: in the library with one
    router it never re-finds the robot. One signal strength cannot tell
    places at the same distance apart.
* **The warehouse is the exception, and the Kalman filter wins there**: 0.29
  against 0.39 m as installed, and 0.31 against 0.70 m with routers at the
  aisle ends. Long identical aisles between metal racks produce look-alike
  signal patterns. The particle filter's multi-modal belief sometimes follows
  the wrong aisle for a while. The Kalman filter's single Gaussian, fed by the
  averaged WKNN fix, smooths over the aliasing. The particle filter is still
  better with only two routers (0.44 against 1.49 m).

### 7.7 The live race and the story layer (`wiloc/race.py`, `wiloc/story.py`)

* The race ranks the methods on a **30 s rolling median**, leaving out 20 s
  after each kidnap.
* Recovery time is measured separately: the time until the error stays below
  1 m for 3 s.
* The Studio's status pill uses hysteresis:
  * **Located:** spread < 1 m and one place holds ≥ 75 % of the weight.
  * **Searching:** the Augmented-MCL alarm $w_\text{fast}/w_\text{slow}$ is
    up.
  * **Not sure:** anything else.
  * The pill changes only after the new state has held for 2 s, except
    Searching, which shows after 0.5 s.
