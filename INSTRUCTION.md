# WiLoc: how to run it

WiLoc simulates a humanoid robot on wheels driving around an office. The robot
works out where it is from Wi-Fi signal strength combined with its wheel
odometry. You drive the robot in **Gazebo** and watch the localisation live in
the **WiLoc window**.

---

## 1. What you need

* Ubuntu **22.04** (a VM is fine)
* **ROS 2 Humble**, installed in `/opt/ros/humble`
* Internet access the first time, so the setup step can install a few packages

Other ROS 2 versions (Iron, Jazzy) will not work: WiLoc uses Gazebo
Classic (Gazebo 11), which only ships with Humble.

To check your setup:

```bash
ls /opt/ros/humble/setup.bash     # this file must exist
lsb_release -rs                   # should print 22.04
```

---

## 2. Get the code

Put the `WiLoc` folder anywhere in your home directory, either with
`git clone <repo-url>` or by copying the folder. Then open a terminal in it:

```bash
cd ~/WiLoc        # or wherever you put it
```

---

## 3. One-time setup (about 5 minutes)

```bash
./wiloc.sh setup
```

This does two things:

* asks for your password and installs the packages WiLoc needs: Gazebo ROS,
  xacro, PyQt5, matplotlib, scipy and a few others;
* builds everything into `~/wiloc_ws`, so nothing is built inside the
  `WiLoc` folder.

At the end it prints `setup done`.

> If you get `Permission denied`, run `chmod +x wiloc.sh` once, then repeat
> the command.

---

## 4. Run it

```bash
./wiloc.sh run
```

Two windows open after about 20 to 40 seconds:

| Window | What it is |
|---|---|
| **Gazebo** (left) | The 3D office and the robot. You drive the robot here. |
| **WiLoc** (right) | Everything the localisation is doing, updated live. |

To stop the simulation, press **Ctrl+C** in the terminal. If anything is
left running, use `./wiloc.sh stop`.

### Driving the robot

Click once inside Gazebo's 3D view, then hold the keys:

| Key | Action |
|---|---|
| **Up / Down arrow** | drive forward / backward |
| **Left / Right arrow** | turn left / right |
| hold two arrows | drive in a curve |
| **Space** | stop |
| **M** | switch between automatic patrol and manual driving |
| **K** | *kidnap*: teleport the robot somewhere random. Watch the filter find it again. |
| **G** | make the filter forget where the robot is (global localisation) |
| **H** | change the heat map: strongest AP, then each AP in turn |

The same keys also work when the WiLoc window has focus. There you can use
**W/A/S/D** as well, and hold **Shift** to drive twice as fast. In Gazebo,
prefer the arrow keys: Gazebo uses S, R and T as its own tool shortcuts.

### Reading the WiLoc window

The **Overview** tab shows everything at once:

* **Map (left).** The coloured background is a heat map of Wi-Fi signal
  strength. The **white circle** is where the robot really is; the **blue
  dot** with its shaded ellipse is where WiLoc thinks it is. The small
  **orange dots** are the particle filter's guesses, and the **coloured
  lines** run from the robot to each Wi-Fi access point, labelled with the
  signal strength and the number of walls in between.
* **Latest Wi-Fi scan.** For each access point, the measured signal next to
  what the learned radio map predicts.
* **Position error.** The localisation error over time. The blue line
  (WiLoc) should stay well below the Wi-Fi-only and odometry-only lines.
* **Filter internals.** How diverse and how spread out the particles are.

The other tabs show the same information in more detail: the full RSSI
breakdown table, larger plots, and a live graph of the ROS nodes and their
message rates.

---

## 5. Useful options

Anything after `run` is passed on to the launch file:

```bash
./wiloc.sh run drive:=true          # robot patrols the office on its own (press M to take over)
./wiloc.sh run method:=ekf          # use the EKF instead of the particle filter
./wiloc.sh run method:=wknn         # raw Wi-Fi fingerprinting, no filter
./wiloc.sh run global_init:=true    # robot starts with no idea where it is
./wiloc.sh run rviz:=true           # also open RViz
./wiloc.sh run ui_font_dpi:=120     # make the text in the windows bigger or smaller
```

Other commands:

```bash
./wiloc.sh test          # run the unit tests (about 1 minute)
./wiloc.sh experiments   # offline experiments, figures go to ./results (a few minutes)
./wiloc.sh build         # rebuild after changing code
./wiloc.sh               # show all commands
```

---

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| Gazebo window is black, flickers or crashes straight away (common in VMs) | `./wiloc.sh run --software-gl`. Also enable 3D acceleration in your VM settings if you can. |
| Gazebo is very slow | Give the VM more CPU cores and memory. You can also skip the Gazebo window and drive from the WiLoc window: `./wiloc.sh run gui:=false` |
| `Spawn service failed` or the robot never appears | Something from a previous run is still alive. Run `./wiloc.sh stop`, then `./wiloc.sh run` again. |
| Keys do nothing | Click inside Gazebo's 3D view, or on the WiLoc window, first. Only the focused window receives keys. |
| Text in the windows is tiny or huge | `./wiloc.sh run ui_font_dpi:=144` (bigger) or `ui_font_dpi:=96` (normal) |
| The robot moves on its own or two simulations interfere | Someone else on the same network is running ROS. Pick your own number: `export ROS_DOMAIN_ID=42`, then run again. |
| `ROS 2 Humble not found` | ROS is installed somewhere else, or it is a different version. WiLoc needs Humble in `/opt/ros/humble`. |
| Build errors after pulling new code | `rm -rf ~/wiloc_ws`, then `./wiloc.sh build` |
