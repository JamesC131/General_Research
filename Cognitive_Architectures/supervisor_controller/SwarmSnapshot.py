import time

class SwarmSnapshot:
    def __init__(self, timeout=2.0):
        """
        timeout: how long (seconds) before a robot is considered 'stale'
        """
        self.robots = {}
        self.timeout = timeout

    def update(self, packet):
        """
        packet example:
        {
            "id": "robot_1",
            "pose": {"x": 1.2, "y": 3.4, "heading": 0.5},
            "battery": 0.9,
            "sensors": {...}
        }
        """

        if "id" not in packet:
            return  # ignore bad packets

        rid = packet["id"]

        # attach timestamp automatically
        packet["last_seen"] = time.time()

        self.robots[rid] = packet

    def remove_stale_robots(self):
        now = time.time()
        to_remove = []

        for rid, data in self.robots.items():
            if now - data.get("last_seen", now) > self.timeout:
                to_remove.append(rid)

        for rid in to_remove:
            del self.robots[rid]

    def get_positions(self):
        return {
            rid: (r["pose"]["x"], r["pose"]["y"])
            for rid, r in self.robots.items()
            if "pose" in r
        }
    
    def get_closest_robot(self, x, y):
        best_id = None
        best_dist = float("inf")

        for rid, r in self.robots.items():
            if "pose" not in r:
                continue

            rx, ry = r["pose"]["x"], r["pose"]["y"]
            dist = (rx - x)**2 + (ry - y)**2

            if dist < best_dist:
                best_dist = dist
                best_id = rid

        return best_id
    
    def get_centroid(self):
        if not self.robots:
            return None

        sx = sy = 0
        count = 0

        for r in self.robots.values():
            if "pose" not in r:
                continue

            sx += r["pose"]["x"]
            sy += r["pose"]["y"]
            count += 1

        return (sx / count, sy / count)
    
    def export(self):
        return {
            "robots": self.robots,
            "count": len(self.robots),
            "timestamp": time.time()
        }