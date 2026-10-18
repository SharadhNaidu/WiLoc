# References and related repositories

Bibliographic details were checked against Crossref, publisher pages, or
library catalogues. Remaining uncertainties are noted where they apply.

## Papers and books

### Fingerprinting systems

1. **RADAR.** P. Bahl, V. N. Padmanabhan, "RADAR: An in-building RF-based user location and tracking system," *Proc. IEEE INFOCOM 2000*, vol. 2, pp. 775–784, 2000. [doi:10.1109/INFCOM.2000.832252](https://doi.org/10.1109/INFCOM.2000.832252)
   - Used in WiLoc for: the NN / KNN fingerprinting idea and the offline/online phases.
2. **Horus.** M. Youssef, A. Agrawala, "The Horus WLAN location determination system," *Proc. MobiSys '05*, pp. 205–218, 2005. [doi:10.1145/1067170.1067193](https://doi.org/10.1145/1067170.1067193)
   - Used in WiLoc for: the probabilistic estimator.

### Radio propagation

3. S. Y. Seidel, T. S. Rappaport, "914 MHz path loss prediction models for indoor wireless communications in multifloored buildings," *IEEE Trans. Antennas Propag.*, 40(2):207–217, 1992. [doi:10.1109/8.127405](https://doi.org/10.1109/8.127405)
   - Used in WiLoc for: the log-distance model with partition losses.
4. A. J. Motley, J. M. P. Keenan, "Personal communication radio coverage in buildings at 900 MHz and 1700 MHz," *Electronics Letters*, 24(12):763–764, 1988. [doi:10.1049/el:19880515](https://doi.org/10.1049/el:19880515)
   - Used in WiLoc for: the multi-wall model.
5. COST Action 231, *Digital Mobile Radio Towards Future Generation Systems, Final Report*, European Commission, 1999, ISBN 92-828-5416-7.
   - Contents: the indoor multi-wall model and typical wall losses, in the indoor chapter, pp. 415–468.
   - Usually cited as E. Damosso (Ed.). The editor list was not fully verified.
6. M. Gudmundson, "Correlation model for shadow fading in mobile radio systems," *Electronics Letters*, 27(23):2145–2146, 1991. [doi:10.1049/el:19911328](https://doi.org/10.1049/el:19911328)
   - Used in WiLoc for: the exponential correlation of shadowing.
7. T. S. Rappaport, *Wireless Communications: Principles and Practice*, 2nd ed., Prentice Hall, 2002, ISBN 0-13-042232-0.
   - Used in WiLoc for: the log-distance path loss (ch. 4).

### Gaussian-process signal maps

8. B. Ferris, D. Hähnel, D. Fox, "Gaussian processes for signal strength-based location estimation," *Robotics: Science and Systems II*, 2006. [doi:10.15607/RSS.2006.II.039](https://doi.org/10.15607/RSS.2006.II.039)
   - Used in WiLoc for: the GP radio map with a path-loss mean.
9. B. Ferris, D. Fox, N. Lawrence, "WiFi-SLAM using Gaussian process latent variable models," *Proc. IJCAI 2007*, pp. 2480–2485. [ijcai.org/Abstract/07/399](https://www.ijcai.org/Abstract/07/399)
   - Relevance: a natural extension, mapping without a labelled survey.

### Robot localization

10. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005, ISBN 978-0-262-20162-9.
    - Used in WiLoc for: the odometry motion model (§5.4), EKF (ch. 3, 7), MCL and augmented MCL (ch. 8), low-variance resampling (Table 4.4).
11. J. Biswas, M. Veloso, "WiFi localization and navigation for autonomous indoor mobile robots," *Proc. IEEE ICRA 2010*, pp. 4379–4384. [doi:10.1109/ROBOT.2010.5509842](https://doi.org/10.1109/ROBOT.2010.5509842)
    - Relevance: Wi-Fi-only localization on a real robot, the closest prior work to this project.

### Datasets and surveys

12. J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," *Proc. IPIN 2014*, pp. 261–270. [doi:10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492)
    - Relevance: a real dataset to validate the estimators against (see *Extending WiLoc* in the README).
13. S. He, S.-H. G. Chan, "Wi-Fi fingerprint-based indoor positioning: Recent advances and comparisons," *IEEE Commun. Surveys & Tutorials*, 18(1):466–490, 2016. [doi:10.1109/COMST.2015.2464084](https://doi.org/10.1109/COMST.2015.2464084)
14. F. Zafari, A. Gkelias, K. K. Leung, "A survey of indoor localization systems and technologies," *IEEE Commun. Surveys & Tutorials*, 21(3):2568–2599, 2019. [doi:10.1109/COMST.2019.2911558](https://doi.org/10.1109/COMST.2019.2911558)

## GitHub repositories consulted

| Repository | What it is | Relation to WiLoc |
|---|---|---|
| [gazebosim/gazebo-classic](https://github.com/gazebosim/gazebo-classic) | Gazebo 11; includes `wireless_transmitter` / `wireless_receiver` sensors | Their RSSI is a free-space model without walls or shadowing, which is why WiLoc generates RSSI in its own node |
| [Lakshadeep/gazebo_wifi_plugin](https://github.com/Lakshadeep/gazebo_wifi_plugin) | ROS 1 Gazebo Wi-Fi router and receiver plugin | Earlier simulation approach built on the same Gazebo sensors |
| [sangwoomoon/gazebo_ros_wireless_comm_lib](https://github.com/sangwoomoon/gazebo_ros_wireless_comm_lib) | Wireless communication library for Gazebo/ROS | Alternative channel modelling |
| [RMiyagusuku/ros-wifi-localization](https://github.com/RMiyagusuku/ros-wifi-localization) | ROS 1 Wi-Fi localization with GP signal models | Closest ROS prior art; WiLoc ports the GP-map idea to ROS 2 |
| [herolab-uga/indoor-rssi-mobile-robot](https://github.com/herolab-uga/indoor-rssi-mobile-robot) | Indoor Wi-Fi RSSI dataset collected by mobile robots | Real data for validating the sensor model |
| [arthuraleksandrovich/UJIIndoorLoc_KNN](https://github.com/arthuraleksandrovich/UJIIndoorLoc_KNN) | KNN / WKNN on UJIIndoorLoc | Reference implementation of the deterministic estimators |
| [debbynirwan/mcl](https://github.com/debbynirwan/mcl) | ROS 2 Monte Carlo localization in Python | Structure of a ROS 2 particle-filter node |
| [ros-navigation/navigation2 (nav2_amcl)](https://github.com/ros-navigation/navigation2) | The reference ROS 2 AMCL | WiLoc follows its interface: `map→odom` TF, `/initialpose`, particle cloud, and a global-localization service |
