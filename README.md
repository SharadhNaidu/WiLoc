# WiLoc: Indoor Localization of Mobile Robots using Wi-Fi RSSI Fingerprinting

WiLoc is an indoor robot localization framework and ROS 2 package that tracks a differential-drive robot using only Wi-Fi Received Signal Strength Indication (RSSI) and wheel odometry, without cameras or LIDAR.

WiLoc includes both a ROS-agnostic core library (`wiloc`) with radio propagation and Bayes filter implementations, and a full ROS 2 Humble + Gazebo Classic simulation environment (`wiloc_ros`) featuring **WiLoc Studio**, an interactive 3D and real-time visualization UI.

---

## Key Features

- **Multi-Method Comparison**: Real-time evaluation of:
  - **Particle Filter (PF / MCL)** with KLD adaptive sampling, sensor reset, and kidnapping recovery.
  - **Extended Kalman Filter (EKF)** fusing differential-drive kinematics with Wi-Fi range/gradient updates.
  - **Single-Scan Fingerprinting**: k-Nearest Neighbors (KNN/WKNN), Probabilistic / Bayesian (Horus-style), and Trilateration baselines.
- **6 Diverse Venues**:
  - Office, Home, Hospital ward, Warehouse, Library, and Factory.
  - Multiple configurable router layout presets per venue.
- **Realistic Radio Propagation**:
  - Log-distance path loss, multi-wall penetration losses by material (concrete, brick, drywall, glass).
  - Correlated spatial shadowing fields via Gaussian processes and fast temporal fading.
  - Dynamic attenuators (moving pedestrians and warehouse forklifts that attenuate RF signals).
  - Router failure and outage detection.
- **Interactive WiLoc Studio UI**:
  - Synchronized live 3D Gazebo camera feed and 2D belief heatmap.
  - Interactive controls: kidnap robot, disconnect/reconnect access points, toggle dynamic obstacles, waypoint navigation.
  - Live benchmark race leaderboard tracking continuous localization errors.

---

## Quickstart

For detailed, step-by-step setup and walkthrough instructions, see [**INSTRUCTION.md**](INSTRUCTION.md).

### Prerequisites
- **Ubuntu 22.04** (native or via **WSL 2** on Windows 10/11)
- **ROS 2 Humble** (Gazebo 11 / `gazebo_ros_pkgs`)

### Installation & Run

1. **One-time Setup**:
   ```bash
   ./wiloc.sh setup
   ```
2. **Start WiLoc Studio**:
   ```bash
   ./wiloc.sh run
   ```
   *(For virtual machines or environments without hardware OpenGL acceleration, use `./wiloc.sh run --software-gl`)*

3. **Run Pre-flight Check**:
   ```bash
   ./wiloc.sh check
   ```

4. **Run Unit Tests**:
   ```bash
   ./wiloc.sh test
   ```

---

## Documentation

- [Step-by-Step Instructions & Demonstration Guide](INSTRUCTION.md)
- [Mathematical Theory & System Architecture](docs/theory.md)
- [Academic References & Prior Art](docs/references.md)

---

## License

This project is licensed under the [MIT License](LICENSE).
