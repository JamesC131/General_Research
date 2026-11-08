from controller import Supervisor
import json
from SwarmSnapshot import SwarmSnapshot

supervisor = Supervisor()
TIME_STEP = 32

snapshot = SwarmSnapshot(timeout=1.5)

receiver = supervisor.getDevice("receiver")

if receiver is None:
    print("ERROR: Receiver device not found. Make sure the supervisor controller has a receiver.")
    exit(1)

receiver.enable(TIME_STEP)

print("Supervisor controller started. Listening for robot packets...")
emitter = supervisor.getDevice("emitter")

root = supervisor.getRoot()
children = root.getField("children")



def save_swarm_graph(snapshot):
    graph = {
        "robots": {},
        "timestamp": supervisor.getTime()
    }

    #print("Saving swarm graph with", len(snapshot.robots), "robots.")
    #print ("Snapshot robots:", snapshot.robots)
    if snapshot.robots.items() == []:
        print("No robots in snapshot, skipping save.")
        return
    
    for robot_id, robot_data in snapshot.robots.items():
        print("Entered save_swarm_graph")
        print(f"Robot {robot_id}: {robot_data}")

        graph["robots"][robot_id] = {
            "position": [
                robot_data.get("x", 0.0),
                robot_data.get("y", 0.0)
            ],
            "heading": robot_data.get("heading", 0.0),
            "battery": robot_data.get("battery", 100),
            "action": robot_data.get("action", "idle"),
            "target": robot_data.get("target", None)
        }

    with open("swarm_graph.json", "w") as f:
        json.dump(graph, f, indent=2)


while supervisor.step(TIME_STEP) != -1:

    # Process incoming robot packets
    #print("Checking for incoming packets...")
    #print("Receiver queue length:", receiver.getQueueLength())
    while receiver.getQueueLength() > 0:

        import json

        message = receiver.getString()
        print("Received message:", message)

        try:
            packet = json.loads(message)
        except json.JSONDecodeError:
            print("Bad packet:", message)
            receiver.nextPacket()
            continue

        snapshot.update(packet)

        receiver.nextPacket()

    save_swarm_graph(snapshot)