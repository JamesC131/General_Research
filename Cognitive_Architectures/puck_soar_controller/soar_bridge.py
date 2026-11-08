from soar_kernel_singleton import get_kernel
import os
import ctypes
import sys

# Load Soar native library FIRST
ctypes.CDLL("/home/james/Downloads/Soar-releases-9.6.4/out/libSoar.so")

sys.path.append("/home/james/Downloads/Soar-releases-9.6.4/out")
import Python_sml_ClientInterface as sml


class SoarBridge:

    def __init__(self, robot_id):

        # IMPORTANT: one kernel per robot (your setup is multi-instance)
        self.kernel = get_kernel()

        if self.kernel.HadError():
            raise RuntimeError(self.kernel.GetLastErrorDescription())

        self.agent = self.kernel.CreateAgent(f"robot_{robot_id}")

        if self.kernel.HadError() or self.agent is None:
            raise RuntimeError(self.kernel.GetLastErrorDescription())

        # Load Soar rules
        soar_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "resource_collector.soar"
        )

        if not self.agent.LoadProductions(soar_file):
            raise RuntimeError(self.agent.GetLastErrorDescription())

        #print(f"[{robot_id}] Soar loaded OK")

        # -------------------------
        # INPUT LINK (ALL STRINGS)
        # -------------------------
        self.il = self.agent.GetInputLink()

        self.resource_visible = self.agent.CreateStringWME(self.il, "resource-visible", "false")
        self.carrying_resource = self.agent.CreateStringWME(self.il, "carrying-resource", "false")
        self.nest_visible = self.agent.CreateStringWME(self.il, "nest-visible", "false")

        self.battery = self.agent.CreateStringWME(self.il, "battery", "ok")

        self.resource_distance = self.agent.CreateFloatWME(self.il, "resource_distance", 0.0)

        self.agent.Commit()

        # output cache
        self.last_action = None

    # -------------------------
    # STATE UPDATE
    # -------------------------
    def update_state(self, resource_visible, carrying_resource, nest_visible, battery, resource_distance):

        self.agent.Update(self.resource_visible, "true" if resource_visible else "false")
        self.agent.Update(self.carrying_resource, "true" if carrying_resource else "false")
        self.agent.Update(self.nest_visible, "true" if nest_visible else "false")

        battery_state = "low" if battery < 50 else "ok"
        self.agent.Update(self.battery, battery_state)

        self.agent.Update(self.resource_distance, float(resource_distance))

    # -------------------------
    # STEP
    # -------------------------
    def step(self):

        # 1. push inputs
        self.agent.Commit()

        # 2. run ONE decision cycle
        self.agent.RunSelf(1)


        #print("Phase:", self.agent.GetCurrentPhase())
        #print("Operators:", self.agent.GetNumberCommands())
        # 3. read output safely (COMMAND MODEL)
        self.last_action = self._read_output()

        # 4. clear output link changes (IMPORTANT for next cycle)

        return self.last_action

    # -------------------------
    # OUTPUT READER (SAFE)
    # -------------------------
    def _read_output(self):
        ol = self.agent.GetOutputLink()

        if ol is None:
            print("No output link")
            return None

        #print("Output link exists")

        i = 0
        child = ol.GetChild(i)

        if child is None:
            print("Output link empty")
            return None

        action = None

        while child is not None:
            attr = child.GetAttribute()
            value = child.GetValueAsString()

           #print("WME:", attr, "=", value)

            if attr == "action":
                action = value

            i += 1
            child = ol.GetChild(i)

        return action