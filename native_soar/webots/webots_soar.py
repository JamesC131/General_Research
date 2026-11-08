from controller import Robot

robot = Robot()

print("\n========== PIONEER DEVICES ==========")

number_of_devices = robot.getNumberOfDevices()

for i in range(number_of_devices):
    device = robot.getDeviceByIndex(i)

    print(
        f"Device {i}: "
        f"{device.getName()} "
        f"({device.getNodeType()})"
    )

print("=====================================\n")