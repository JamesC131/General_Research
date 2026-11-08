from controller import Robot
from soar_bridge import SoarBridge

import math
import random
import hashlib

# ----------------------------
# Robot init
# ----------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

name = robot.getName()
print("Starting robot:", name)

emitter = robot.getDevice("emitter")

heading = 0.0

# Soar
# ----------------------------
bridge = SoarBridge(name)

# ----------------------------
# Devices
# ----------------------------
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

MAX_SPEED = 6.28

# ----------------------------
# GPS (IMPORTANT ADDITION)
# ----------------------------
gps = None
try:
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    print("GPS enabled")
except:
    print("WARNING: GPS not found (using fallback 0,0)")

# ----------------------------
# Sensors
# ----------------------------
proximity = []

for i in range(8):
    ps = robot.getDevice(f"ps{i}")
    ps.enable(timestep)
    proximity.append(ps)

# ----------------------------
# Helpers
# ----------------------------
def read_proximity():
    return [ps.getValue() / 4096.0 for ps in proximity]

def get_position():
    """REAL POSITION FEEDBACK"""
    if gps is not None:
        x, y, z = gps.getValues()
        return x, z   # Webots uses XZ plane for ground robots
    return 0.0, 0.0

def build_packet(name, x, y, heading, battery):
    return {
        "id": name,
        "pose": {
            "x": float(x),
            "y": float(y),
            "heading": float(heading)
        },
        "battery": float(battery),
        "sensors": {
            "proximity": read_proximity()
        }
    }

# ----------------------------
# ACTION EXECUTION
# ----------------------------
def execute_action(action, pos):
    x, y = pos

    if action is None:
        return



    # clamp
    #left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
    #right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

# ----------------------------
# Fake world state (temporary)
# ----------------------------
battery = 100.0
carrying_resource = False

def get_vision_state():
    return False, 10.0, False

# ----------------------------
# Main loop
# ----------------------------
while robot.step(timestep) != -1:

    #print(f"{name}: loop")

    ps_vals = read_proximity()

    # battery decay
    battery -= 0.01
    if battery < 0:
        battery = 0

    # ----------------------------
    # POSITION FEEDBACK (CRITICAL)
    # ----------------------------
    x, y = get_position()

    resource_visible, resource_distance, nest_visible = get_vision_state()

    bridge.update_state(
        resource_visible,
        carrying_resource,
        nest_visible,
        battery,
        resource_distance
    )

    x, y = get_position()

    heading = 0.0  # placeholder (can improve later)

    packet = build_packet(name, x, y, heading, battery)

    import json
    emitter.send(json.dumps(packet))

    action = bridge.step()
    #print(f"{name}: action = {action}")

    # ----------------------------
    # SAFETY (optional)
    # ----------------------------
    ps_vals = read_proximity()

    if max(ps_vals) > 0.8:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        continue

    # ----------------------------
    # EXECUTE CONTROL
    # ----------------------------
    execute_action(action, (x, y))