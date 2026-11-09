# WiLoc: step-by-step instructions

Follow these steps **in order** and do not skip any. Copy each command exactly
and paste it into the terminal. You do not need to understand the commands.

**What you will get:** one window, **WiLoc Studio**. On the left is a live 3D
building (an office, a home, a hospital ward, a warehouse, a library or a
factory) with a small robot driving around. On the right is a map of where the
robot thinks it is, worked out from nothing but Wi-Fi signal strength and its
wheel turns.

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

## Step 4: Start WiLoc

Paste this and press Enter:

```bash
./wiloc.sh run
```

One window opens, **WiLoc Studio**, full screen. It shows "Starting WiLoc…"
for about 10 to 30 seconds while the simulation starts behind it.

* **F11** switches between full screen and a normal window.
* **Ctrl + Shift + Q** quits.
* Leave the terminal open. Closing it stops WiLoc.

---

## Step 5: What you see

| Part of the window | What it shows |
|---|---|
| **Top right** | The **big number** is how far, in metres, the robot's own guess is from where it really is. The pill next to it says **Located**, **Not sure** or **Searching**, with one line of explanation underneath. |
| **Venue and Routers chips** | Pick a building: Office, Home, Hospital ward, Warehouse, Library, Factory. Then pick how the Wi-Fi routers are placed in it. A new building takes about 15 seconds; a new router layout switches at once. |
| **The real world** (left) | The live 3-D simulation. The coloured lines run from the robot to each Wi-Fi router: thick means a strong signal, dashed means it goes through a wall, and faded means a person or a forklift is in the way. The rings on the floor are the guesses of the three methods. |
| **What the robot believes** (right) | The map. The orange background is the Wi-Fi signal strength. The coloured dots are the routers. The black triangle is the real robot, and the blue dot with small blue dots around it is the particle filter's guess with its particles. People and vehicles move on the map too. |
| **The race** (bottom left) | How wrong each method is over the last 2 minutes: the **particle filter** (Wi-Fi + wheels + map), the **Kalman filter** (Wi-Fi + wheels), **Wi-Fi only**, and **wheels only**. The chips show each method's typical error over the last 30 seconds. |
| **Try it** (bottom right) | The buttons for visitors. |

---

## Step 6: Try it

| Do this | What happens |
|---|---|
| **Kidnap the robot** | The robot is teleported somewhere else, so its wheels do not notice. The pill turns to **Searching**, and a few seconds later a message says **"Re-found in 6.2 s"**. |
| **Click on the map** | The robot plans a path there, through the doors, and drives to it. |
| **Click a router** on the map | Switches that router off (unplugged), or back on. |
| **Unplug a router** | Switches off the router the robot hears loudest. Within about 2 seconds the map shows it as **"off · noticed"**: the robot has worked out that it is gone and stops relying on it. Press again to plug everything back in. |
| **People block Wi-Fi: on / off** | Whether people and vehicles weaken the Wi-Fi signals passing through them. |
| **Patrol** | The robot goes back to driving its round. |
| **View: overview / follow** | The 3-D camera watches the whole building, or follows the robot. |
| **Arrow keys** or **W A S D** | Drive the robot by hand. **Patrol** gives control back. |

Keys for the person running the stand (visitors will not press these by accident):

| Keys | What happens |
|---|---|
| **Ctrl + Shift + R** | Soft reset: robot back to the start, all routers on, people on, patrol on. |
| **Ctrl + Shift + F** | The robot "forgets" where it is and has to find itself from scratch. |
| **Ctrl + Shift + A** | Auto tour on or off (see Step 8). |
| **Ctrl + Shift + Q** | Quit. |

---

## Step 7: Stop

Press **Ctrl + Shift + Q** in the window, or **Ctrl + C** in the terminal.

If something stays open, paste this and press Enter:

```bash
./wiloc.sh stop
```

---

## Step 8: At the poster stand

**The day before**
1. `./wiloc.sh check`: a 2-minute test run without a window. Every line must
   say **PASS**. For example: *particle filter tracks the robot: median error
   0.3 m*; *re-found after a kidnap: 7 s*.
2. `./wiloc.sh record`: records a 6-minute backup video to
   `results/wiloc_demo.mp4`. It takes about 7 minutes. Copy the video to your
   phone as well.

**At the venue, 15 minutes before visitors arrive**
* **Power:** laptop on the charger. Windows power mode **Best performance**.
* **Never sleep:** Settings → System → Power: screen and sleep **Never**.
  Closing the lid: **Do nothing**.
* **No pop-ups:** turn on **Do not disturb**. Pause Windows Update for a week.
* **Projector or monitor:** press **Win + P** → **Duplicate**.
* **Start:** in the terminal run `cd ~/WiLoc && ./wiloc.sh demo`.

`demo` is `run` plus the **auto tour**. When nobody has touched the laptop for
90 seconds, it patrols, kidnaps the robot every couple of minutes and moves on
to the next building every 6 minutes. Any key or click hands control back to
the visitor at once.

**A 2-minute talk for visitors**
1. *"This robot has no GPS and no camera. It finds itself using only the Wi-Fi
   signal strength from the routers, plus its wheel turns."* Point at the big
   number: it is usually 0.2 to 0.5 m.
2. *"Watch: I'll kidnap it."* Press **Kidnap the robot**. *"Its wheels didn't
   feel that, so it thinks it is still over there. Wi-Fi tells it otherwise.
   Searching… Re-found."*
3. *"Three methods are racing."* Point at the race chart: *"Wi-Fi alone jumps
   around by metres. Adding the wheels and the floor plan, which is the
   particle filter, keeps it steady."*
4. For people who stay: unplug a router, switch to the **Hospital ward**
   (identical rooms: "Not sure: Room 3 or Room 11"), or the **Warehouse**
   (metal racks block Wi-Fi). Let them click on the map to send the robot.

**If something goes wrong in front of visitors:** press **Ctrl + Shift + R**.
If the window shows "Restarting the simulation…", wait 15 seconds: it restarts
itself. As a last resort, play the backup video.

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
| The 3-D view stays **grey or black**, or the window closes (common in virtual machines) | Quit, then start with `./wiloc.sh run --software-gl` |
| Everything is **very slow** | Close other programs. In a VM, give it 4+ CPUs and 8 GB+ memory. |
| "Starting WiLoc…" for more than a minute | Quit, run `./wiloc.sh stop`, wait 5 seconds, then `./wiloc.sh run` again. |
| Your own `ros2 topic list` in another terminal shows nothing | WiLoc keeps ROS traffic on this computer. Run `export ROS_LOCALHOST_ONLY=1` in that terminal first. |
| Errors after `git pull` | Run `rm -rf ~/wiloc_ws && ./wiloc.sh build`, then `./wiloc.sh run`. |

Still stuck? Send the team a photo of the terminal showing the error.

---

## For the curious: more options

```bash
./wiloc.sh run --venue hospital --layout staggered   # start in another building / router layout
./wiloc.sh run --windowed                            # a normal window instead of full screen
./wiloc.sh run --launch-arg global_init:=true        # robot starts with no idea where it is
./wiloc.sh run --launch-arg method:=ekf              # the Kalman filter drives the robot's map frame
./wiloc.sh launch gui:=true rviz:=true               # the bare simulation with Gazebo's own window and RViz
./wiloc.sh check                                     # pre-flight check (about 2 minutes)
./wiloc.sh record my_video.mp4                       # record the backup video
./wiloc.sh test                                      # run the automatic tests (about 1 minute)
./wiloc.sh experiments                               # make all the result graphs (into the results folder)
PYTHONPATH=. python3 -m wiloc venues                 # the six-venue comparison (results/venues.png)
./wiloc.sh                                           # list every command
```

How it works, with the maths: [docs/theory.md](docs/theory.md).
