import pygame
import numpy as np
import random

# ============================================================
# SETTINGS
# ============================================================

WORLD_MIN = -10
WORLD_MAX = 10

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

CAR_RADIUS = 0.25

DT = 0.1

# ============================================================
# PYGAME SETUP
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption("RC Car Autopilot Simulator")

clock = pygame.time.Clock()

font = pygame.font.Font(None, 28)


# ============================================================
# WORLD <-> SCREEN COORDINATES
# ============================================================

def world_to_screen(x, y):

    screen_x = int(
        (x - WORLD_MIN)
        / (WORLD_MAX - WORLD_MIN)
        * SCREEN_WIDTH
    )

    # Flip Y because screen coordinates increase downward
    screen_y = int(
        SCREEN_HEIGHT
        - (y - WORLD_MIN)
        / (WORLD_MAX - WORLD_MIN)
        * SCREEN_HEIGHT
    )

    return screen_x, screen_y


def world_length_to_pixels(length):

    return int(
        length
        / (WORLD_MAX - WORLD_MIN)
        * SCREEN_WIDTH
    )


# ============================================================
# RAYCAST
# ============================================================

def raycast(
    x,
    y,
    angle,
    obstacles,
    body_radius=0.0,
    world_min=WORLD_MIN,
    world_max=WORLD_MAX
):

    dx = np.cos(angle)
    dy = np.sin(angle)

    distances = []

    # -------------------------
    # WORLD BOUNDARIES
    # -------------------------

    if dx > 0:
        distances.append(
            (world_max - x) / dx
        )

    elif dx < 0:
        distances.append(
            (world_min - x) / dx
        )

    if dy > 0:
        distances.append(
            (world_max - y) / dy
        )

    elif dy < 0:
        distances.append(
            (world_min - y) / dy
        )

    # -------------------------
    # OBSTACLES
    # -------------------------

    for obstacle in obstacles:

        xmin = (
            obstacle["x"]
            - obstacle["width"] / 2
            - body_radius
        )

        xmax = (
            obstacle["x"]
            + obstacle["width"] / 2
            + body_radius
        )

        ymin = (
            obstacle["y"]
            - obstacle["height"] / 2
            - body_radius
        )

        ymax = (
            obstacle["y"]
            + obstacle["height"] / 2
            + body_radius
        )

        if abs(dx) < 1e-8:

            if x < xmin or x > xmax:
                continue

            tx_min = -np.inf
            tx_max = np.inf

        else:

            tx1 = (xmin - x) / dx
            tx2 = (xmax - x) / dx

            tx_min = min(tx1, tx2)
            tx_max = max(tx1, tx2)

        if abs(dy) < 1e-8:

            if y < ymin or y > ymax:
                continue

            ty_min = -np.inf
            ty_max = np.inf

        else:

            ty1 = (ymin - y) / dy
            ty2 = (ymax - y) / dy

            ty_min = min(ty1, ty2)
            ty_max = max(ty1, ty2)

        t_min = max(tx_min, ty_min)
        t_max = min(tx_max, ty_max)

        if t_max >= max(t_min, 0):

            if t_min >= 0:
                distances.append(t_min)

            else:
                distances.append(t_max)

    return min(d for d in distances if d > 0)


# ============================================================
# EXPERT CONTROLLER
# ============================================================



def work_through_example(
    end_goal_x,
    end_goal_y,
    sensors,
    current_x,
    current_y,
    current_angle
):


    distance_to_goal = np.sqrt(
        (end_goal_x - current_x)**2 +
        (end_goal_y - current_y)**2
    )

    # -------------------------
    # GOAL
    # -------------------------

    if distance_to_goal < 0.1:
        return 0.0, 0.0

    # -------------------------
    # OBSTACLE
    # -------------------------

    # -------------------------
    # OBSTACLE AVOIDANCE
    # -------------------------

    OBSTACLE_DISTANCE = 1.5
    SIDE_DISTANCE = 1.0

    front = sensors["front"]
    left = sensors["left"]
    right = sensors["right"]

    obstacle_detected = (
        front < OBSTACLE_DISTANCE
        or left < SIDE_DISTANCE
        or right < SIDE_DISTANCE
    )

    if obstacle_detected and distance_to_goal > 1.0:

        # --------------------------------
        # FRONT OBSTACLE
        # --------------------------------
        if front < OBSTACLE_DISTANCE:

            # Determine which side has more free space
            if left > right:
                steering = 0.8
            else:
                steering = -0.8

            # Slow down as obstacle gets closer
            speed = 0.5 * (
                front / OBSTACLE_DISTANCE
            )

            speed = np.clip(
                speed,
                0.15,
                0.5
            )

            return steering, speed
        # --------------------------------
        # LEFT OBSTACLE
        # --------------------------------

        if left < SIDE_DISTANCE:

            urgency = (
                SIDE_DISTANCE - left
            ) / SIDE_DISTANCE

            steering = -0.5 * urgency

            return steering, 0.5

        # --------------------------------
        # RIGHT OBSTACLE
        # --------------------------------

        if right < SIDE_DISTANCE:

            urgency = (
                SIDE_DISTANCE - right
            ) / SIDE_DISTANCE

            steering = 0.5 * urgency

            return steering, 0.5

    # -------------------------
    # NORMAL GOAL SEEKING
    # -------------------------

    angle_to_goal = np.arctan2(
        end_goal_y - current_y,
        end_goal_x - current_x
    )

    angle_diff = angle_to_goal - current_angle

    angle_diff = (
        angle_diff + np.pi
    ) % (2 * np.pi) - np.pi


    STEERING_GAIN = 0.5
    STEERING_DEADBAND = 0.03

    steering = np.clip(
        angle_diff * STEERING_GAIN,
        -1.0,
        1.0
    )

    speed = 1.0

    # Kill tiny steering corrections
    if abs(steering) < STEERING_DEADBAND:
        steering = 0.0

    if distance_to_goal < 1.0:
        speed = 0.3

    return steering, speed

# ============================================================
# CAR PHYSICS
# ============================================================

MAX_STEERING = 1.0
STEERING_RATE = 3.0
MAX_STEERING_CHANGE = 0.15

def update_car(
    current_x,
    current_y,
    current_angle,
    steering,
    speed,
    dt=DT
):

    # --------------------------------------------------------
    # LIMIT REQUESTED STEERING
    # --------------------------------------------------------

    steering = np.clip(
        steering,
        -MAX_STEERING,
        MAX_STEERING
    )

    # --------------------------------------------------------
    # LIMIT HOW FAST STEERING CAN CHANGE
    # --------------------------------------------------------

    steering_change = steering - update_car.current_steering

    steering_change = np.clip(
        steering_change,
        -MAX_STEERING_CHANGE,
        MAX_STEERING_CHANGE
    )

    update_car.current_steering += steering_change

    # --------------------------------------------------------
    # APPLY STEERING TO CAR HEADING
    # --------------------------------------------------------

    current_angle += (
        update_car.current_steering
        * STEERING_RATE
        * dt
    )

    # --------------------------------------------------------
    # MOVE CAR
    # --------------------------------------------------------

    current_x += (
        speed
        *
        np.cos(current_angle)
        *
        dt
    )

    current_y += (
        speed
        *
        np.sin(current_angle)
        *
        dt
    )

    return (
        current_x,
        current_y,
        current_angle
    )


# Starting steering position
update_car.current_steering = 0.0

def collision_with_obstacles(
    x,
    y,
    obstacles
):

    for obstacle in obstacles:

        xmin = (
            obstacle["x"]
            -
            obstacle["width"] / 2
            -
            CAR_RADIUS
        )

        xmax = (
            obstacle["x"]
            +
            obstacle["width"] / 2
            +
            CAR_RADIUS
        )

        ymin = (
            obstacle["y"]
            -
            obstacle["height"] / 2
            -
            CAR_RADIUS
        )

        ymax = (
            obstacle["y"]
            +
            obstacle["height"] / 2
            +
            CAR_RADIUS
        )

        if (
            xmin <= x <= xmax
            and
            ymin <= y <= ymax
        ):
            return True

    return False


def generate_world():

    while True:

        goal_x = random.uniform(
            WORLD_MIN + 1,
            WORLD_MAX - 1
        )

        goal_y = random.uniform(
            WORLD_MIN + 1,
            WORLD_MAX - 1
        )

        car_x = random.uniform(
            WORLD_MIN + 1,
            WORLD_MAX - 1
        )

        car_y = random.uniform(
            WORLD_MIN + 1,
            WORLD_MAX - 1
        )

        # Don't start too close to goal

        distance = np.sqrt(
            (goal_x - car_x) ** 2
            +
            (goal_y - car_y) ** 2
        )

        if distance > 5:
            break

    car_angle = random.uniform(
        -np.pi,
        np.pi
    )

    obstacles = []

    for _ in range(5):

        for attempt in range(100):

            obstacle_x = random.uniform(
                WORLD_MIN + 1.5,
                WORLD_MAX - 1.5
            )

            obstacle_y = random.uniform(
                WORLD_MIN + 1.5,
                WORLD_MAX - 1.5
            )

            width = random.uniform(
                1.0,
                3.0
            )

            height = random.uniform(
                1.0,
                3.0
            )

            # Don't put obstacle near car

            car_distance = np.sqrt(
                (obstacle_x - car_x) ** 2
                +
                (obstacle_y - car_y) ** 2
            )

            # Don't put obstacle near goal

            goal_distance = np.sqrt(
                (obstacle_x - goal_x) ** 2
                +
                (obstacle_y - goal_y) ** 2
            )

            if car_distance < 2:
                continue

            if goal_distance < 2:
                continue

            obstacles.append({
                "x": obstacle_x,
                "y": obstacle_y,
                "width": width,
                "height": height
            })

            break

    return (
        car_x,
        car_y,
        car_angle,
        goal_x,
        goal_y,
        obstacles
    )


# ============================================================
# DRAW CAR
# ============================================================

def draw_car(
    x,
    y,
    angle
):

    center = np.array(
        world_to_screen(x, y)
    )

    size = world_length_to_pixels(
        0.7
    )

    # Triangle pointing forward

    forward = np.array([
        np.cos(angle),
        -np.sin(angle)
    ])

    right = np.array([
        np.cos(angle - np.pi / 2),
        -np.sin(angle - np.pi / 2)
    ])

    p1 = center + forward * size
    p2 = center - forward * size * 0.7 + right * size * 0.6
    p3 = center - forward * size * 0.7 - right * size * 0.6

    pygame.draw.polygon(
        screen,
        (50, 150, 255),
        [
            tuple(p1.astype(int)),
            tuple(p2.astype(int)),
            tuple(p3.astype(int))
        ]
    )


# ============================================================
# DRAW WORLD
# ============================================================

def draw_world(
    car_x,
    car_y,
    car_angle,
    goal_x,
    goal_y,
    obstacles,
    front_distance
):

    screen.fill((30, 30, 30))

    # -------------------------
    # WORLD BORDER
    # -------------------------

    pygame.draw.rect(
        screen,
        (200, 200, 200),
        (
            0,
            0,
            SCREEN_WIDTH - 1,
            SCREEN_HEIGHT - 1
        ),
        3
    )

    # -------------------------
    # OBSTACLES
    # -------------------------

    for obstacle in obstacles:

        x, y = world_to_screen(
            obstacle["x"],
            obstacle["y"]
        )

        width = world_length_to_pixels(
            obstacle["width"]
        )

        height = world_length_to_pixels(
            obstacle["height"]
        )

        rect = pygame.Rect(
            x - width // 2,
            y - height // 2,
            width,
            height
        )

        pygame.draw.rect(
            screen,
            (180, 180, 180),
            rect
        )

    # -------------------------
    # GOAL
    # -------------------------

    goal_screen = world_to_screen(
        goal_x,
        goal_y
    )

    pygame.draw.circle(
        screen,
        (50, 220, 50),
        goal_screen,
        10
    )

    # -------------------------
    # SENSOR RAY
    # -------------------------

    ray_end_x = (
        car_x
        +
        front_distance
        *
        np.cos(car_angle)
    )

    ray_end_y = (
        car_y
        +
        front_distance
        *
        np.sin(car_angle)
    )

    ray_start = world_to_screen(
        car_x,
        car_y
    )

    ray_end = world_to_screen(
        ray_end_x,
        ray_end_y
    )

    pygame.draw.line(
        screen,
        (255, 255, 0),
        ray_start,
        ray_end,
        2
    )

    # -------------------------
    # CAR
    # -------------------------

    draw_car(
        car_x,
        car_y,
        car_angle
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        car_x,
        car_y,
        car_angle,
        goal_x,
        goal_y,
        obstacles
    ) = generate_world()

    running = True

    while running:

        # -------------------------
        # EVENTS
        # -------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # Press R for new world

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_r
            ):

                (
                    car_x,
                    car_y,
                    car_angle,
                    goal_x,
                    goal_y,
                    obstacles
                ) = generate_world()

        # -------------------------
        # SENSOR
        # -------------------------

        sensors = {
            "front": raycast(
                car_x,
                car_y,
                car_angle,
                obstacles,
                CAR_RADIUS
            ),

            "left": raycast(
                car_x,
                car_y,
                car_angle + np.pi / 4,
                obstacles,
                CAR_RADIUS
            ),

            "right": raycast(
                car_x,
                car_y,
                car_angle - np.pi / 4,
                obstacles,
                CAR_RADIUS
            )
        }


        # -------------------------
        # CONTROLLER
        # -------------------------

        steering, speed = work_through_example(
            goal_x,
            goal_y,
            sensors,
            car_x,
            car_y,
            car_angle
        )

        # -------------------------
        # MOVE CAR
        # -------------------------

        car_x, car_y, car_angle = update_car(
            car_x,
            car_y,
            car_angle,
            steering,
            speed
        )

        # -------------------------
        # CHECK COLLISION
        # -------------------------

        if collision_with_obstacles(
            car_x,
            car_y,
            obstacles
        ):

            print("COLLISION!")

            (
                car_x,
                car_y,
                car_angle,
                goal_x,
                goal_y,
                obstacles
            ) = generate_world()

        # -------------------------
        # CHECK GOAL
        # -------------------------

        distance_to_goal = np.sqrt(
            (goal_x - car_x) ** 2
            +
            (goal_y - car_y) ** 2
        )

        if distance_to_goal < 0.1:

            print("GOAL REACHED!")

            (
                car_x,
                car_y,
                car_angle,
                goal_x,
                goal_y,
                obstacles
            ) = generate_world()

        # -------------------------
        # DRAW
        # -------------------------

        draw_world(
            car_x,
            car_y,
            car_angle,
            goal_x,
            goal_y,
            obstacles,
            sensors["front"]
        )

        # -------------------------
        # DEBUG TEXT
        # -------------------------

        text = font.render(
            f"Front: {sensors['front']:.2f}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            text,
            (15, 15)
        )

        text = font.render(
            f"Goal distance: {distance_to_goal:.2f}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            text,
            (15, 45)
        )

        text = font.render(
            f"Steering: {steering:.2f}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            text,
            (15, 75)
        )

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()