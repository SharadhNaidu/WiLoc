# WiLoc: step-by-step instructions

Follow these steps **in order** and do not skip any. Copy each command exactly
and paste it into the terminal. You do not need to understand the commands.

**What you will get:** a 3D office in which a small humanoid robot on wheels
drives around. You drive the robot with the arrow keys. A second window shows
where the robot really is and where Wi-Fi tells us it is.

---

## Step 0: Check that your computer can run it

You need:

* **Ubuntu 22.04**. A virtual machine (VirtualBox or VMware) is fine.
* **ROS 2 Humble**. Your VM probably has it already.

**Open a terminal:** press **Ctrl + Alt + T** at the same time. A window with
a black or purple background opens. This is the terminal; you type or paste
commands into it.

> **How to paste into the terminal:** use **Ctrl + Shift + V**, not just
> Ctrl + V. After pasting, press **Enter** to run the command.

Paste this and press Enter:

```bash
ls /opt/ros/humble/setup.bash && lsb_release -rs
```

You should see:

```
/opt/ros/humble/setup.bash
22.04
```

* If you see that, continue with Step 1.
* If you see `No such file or directory`, ROS 2 Humble is not installed.
  Ask the team before going further, because WiLoc only works with ROS 2
  Humble.

---

## Step 1: Install git (skip this if you already have it)

Paste this and press Enter:

```bash
sudo apt update && sudo apt install -y git
```

It asks for your **password**: the one you use to log in to Ubuntu.
**Nothing appears on screen while you type the password. That is normal.**
Type it and press Enter.

Wait until the command finishes and you can type again.

---

## Step 2: Download WiLoc

Paste this and press Enter:

```bash
cd ~ && git clone https://github.com/SharadhNaidu/WiLoc.git && cd WiLoc
```

This downloads the project into a folder called `WiLoc` in your home
folder, and moves the terminal into it.

**Check:** the last part of the line where you type should now show
`~/WiLoc`.

---

## Step 3: One-time setup (about 5-10 minutes)

Paste this and press Enter:

```bash
./wiloc.sh setup
```

* It may ask for your password again. Type it (nothing shows) and press Enter.
* A lot of text will scroll past. That is normal. Wait.
* It is finished when you see:

  ```
  [wiloc] setup done. Start the simulation with:  ./wiloc.sh run
  ```

> **If you see `Permission denied`:** paste `chmod +x wiloc.sh`, press
> Enter, then run `./wiloc.sh setup` again.

**You only do Steps 1 to 3 once.**

---

## Step 4: Start the simulation

Paste this and press Enter:

```bash
./wiloc.sh run
```

Wait **20 to 60 seconds**. Two windows will open:

* **Gazebo** (left): the 3D office with the robot in it. The robot is the
  white humanoid with blue eyes standing in the corridor.
* **WiLoc** (right): a map with coloured areas, plus some graphs.

> Leave the terminal open. Closing it stops the simulation.

---

## Step 5: Drive the robot

1. **Click once** with the mouse anywhere inside the 3D view of the
   **Gazebo** window (the picture of the office).
2. Press and **hold** the arrow keys:

| Key | What the robot does |
|---|---|
| **Up arrow** | drives forward |
| **Down arrow** | drives backward |
| **Left arrow** | turns left |
| **Right arrow** | turns right |
| **Up + Left together** | drives in a curve |
| **Space** | stops |

Let go of the key and the robot stops.

You can also click on the **WiLoc** window and drive from there. In that
window **W, A, S, D** work as well, and holding **Shift** makes the robot go
twice as fast.

### Fun extra keys (work in both windows)

| Key | What happens |
|---|---|
| **K** | **Kidnap**: the robot is teleported to a random place. Watch how WiLoc finds it again. |
| **G** | WiLoc "forgets" where the robot is and has to find it from scratch. |
| **H** | Changes the coloured Wi-Fi map: strongest signal, then each Wi-Fi router in turn. |
| **M** | The robot drives around the office by itself. Press **M** again to take back control. |

---

## Step 6: Understand what you see in the WiLoc window

Look at the **Overview** tab, the first one at the top.

**The big map on the left:**

* The **coloured background** shows the Wi-Fi signal strength. Red or orange
  means strong, light blue means weak.
* The **coloured dots** with names (AP1, AP2, ...) are the **Wi-Fi routers**.
* The **white circle** is where the robot **really** is.
* The **blue dot** with the blue shaded area is where **WiLoc thinks** the
  robot is. The shaded area shows how unsure it is.
* The **small orange dots** are the guesses the algorithm is considering.
* The **coloured lines** run from the robot to each router. Each label shows
  the signal strength (for example `-47 dBm`) and how many walls the signal
  passes through (`3w` means 3 walls).
* The **lines on the floor** show the paths: black is the real path, blue is
  what WiLoc estimated, and grey dashed is where the robot would think it is
  using only its wheels.

**The graphs on the right:**

* **Latest Wi-Fi scan:** the signal from each router, measured versus
  predicted.
* **Position error:** how wrong each method is, in metres. The **blue line
  (WiLoc) should be the lowest**.
* **Filter internals:** technical details of the algorithm.

**The status bar at the top** shows the current error, for example
`error now 0.30 m`.

The other tabs (**Wi-Fi details**, **Localization details**, **ROS graph**)
show more detail if you are curious.

---

## Step 7: Stop the simulation

Click on the **terminal** window and press **Ctrl + C**.

If any windows stay open, paste this and press Enter:

```bash
./wiloc.sh stop
```

---

## Running it again another day

Open a terminal (**Ctrl + Alt + T**) and paste:

```bash
cd ~/WiLoc && ./wiloc.sh run
```

That's all. Steps 1 to 3 are **not** needed again.

To get the newest version of the project first:

```bash
cd ~/WiLoc && git pull && ./wiloc.sh build && ./wiloc.sh run
```

---

## Something went wrong?

| What you see | What to do |
|---|---|
| `Permission denied` when running `./wiloc.sh` | Run `chmod +x wiloc.sh`, then try again. |
| `ROS 2 Humble not found` | ROS 2 Humble is not installed. Ask the team. |
| The Gazebo window is **black**, flickers, or closes right away (common in virtual machines) | Stop with **Ctrl + C**, then start with `./wiloc.sh run --software-gl` |
| Everything is **very slow** | In your VM settings, give it more CPUs (4 or more) and memory (8 GB or more). You can also run without the 3D window and drive from the WiLoc window: `./wiloc.sh run gui:=false` |
| `Spawn service failed`, or **no robot** appears | Run `./wiloc.sh stop`, wait 5 seconds, then `./wiloc.sh run` again. |
| Pressing keys does **nothing** | **Click** inside the Gazebo 3D view, or on the WiLoc window, first. Keys only go to the window you clicked. |
| Text in the windows is **too small or too big** | Bigger: `./wiloc.sh run ui_font_dpi:=144`. Normal: `./wiloc.sh run ui_font_dpi:=96` |
| The robot moves **by itself** in a strange way | Someone else on the same network runs ROS. Run `export ROS_DOMAIN_ID=42`, then `./wiloc.sh run` in the same terminal. |
| Errors after `git pull` | Run `rm -rf ~/wiloc_ws && ./wiloc.sh build`, then `./wiloc.sh run`. |

Still stuck? Send the team a photo of the terminal showing the error.

---

## For the curious: more options

```bash
./wiloc.sh run drive:=true          # robot starts by driving around on its own
./wiloc.sh run method:=ekf          # use a different algorithm (Kalman filter)
./wiloc.sh run method:=wknn         # Wi-Fi only, no filter (to see how much worse it is)
./wiloc.sh run global_init:=true    # robot starts with no idea where it is
./wiloc.sh run rviz:=true           # also open RViz
./wiloc.sh test                     # run the automatic tests (about 1 minute)
./wiloc.sh experiments              # make all the result graphs (into the results folder)
./wiloc.sh                          # list every command
```
