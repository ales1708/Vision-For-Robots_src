import numpy as np
import matplotlib.pyplot as plt
import cv2
from geometry_msgs.msg import Twist

A = 9  # length
B = 6  # width
C = 00.5  # line width
D = 0.1  # penalty mark size
E = 0.6  # goal area length
F = 2.2  # goal area width
G = 1.65  # penalty area length
H = 4  # penalty area width
I = 1.3  # penalty mark distance
J = 1.5  # center circle diameter
K = 0.7  # border strip width


def draw_field(ax):
    # outer field
    ax.add_patch(
        plt.Rectangle((0, 0), A, B, fill=False, edgecolor="white", linewidth=2)
    )

    # halfway line
    ax.plot([A / 2, A / 2], [0, B], "w", linewidth=2)

    # center circle
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(
        A / 2 + (J / 2) * np.cos(th), B / 2 + (J / 2) * np.sin(th), "w", linewidth=2
    )

    # center mark
    ax.plot(A / 2, B / 2, "wo", markersize=4)

    # left penalty area
    ax.add_patch(
        plt.Rectangle(
            (0, (B - H) / 2), G, H, fill=False, edgecolor="white", linewidth=2
        )
    )

    # right penalty area
    ax.add_patch(
        plt.Rectangle(
            (A - G, (B - H) / 2), G, H, fill=False, edgecolor="white", linewidth=2
        )
    )

    # left goal area
    ax.add_patch(
        plt.Rectangle(
            (0, (B - F) / 2), E, F, fill=False, edgecolor="white", linewidth=2
        )
    )

    # right goal area
    ax.add_patch(
        plt.Rectangle(
            (A - E, (B - F) / 2), E, F, fill=False, edgecolor="white", linewidth=2
        )
    )

    # penalty marks
    ax.plot(I, B / 2, "wo", markersize=4)
    ax.plot(A - I, B / 2, "wo", markersize=4)


def triangulation_3p(
    a=[9.0, 3.0], b=[7.350, 0.0], c=[4.5, 0.0], a_d=4.0, b_d=3.0, c_d=3.5
):
    """Performs triangulation to compute the position of the Robot.
    a, b: the world-space coordinate of the two reference points,
    a_d, b_d, the distance to points a and b respectively,
    returns: the real world coordinates of the robots position.
    """
    # TO DO: get mean of all 3 "correct" seeming intersections
    [ab_x, ab_y] = triangulation_2p(a, b, a_d, b_d)
    [ac_x, ac_y] = triangulation_2p(a, c, a_d, c_d)
    [bc_x, bc_y] = triangulation_2p(b, c, b_d, c_d)

    x = np.mean(np.array([ab_x, ac_x, bc_x]))
    y = np.mean(np.array([ab_y, ac_y, bc_y]))

    fig, ax = plt.subplots(figsize=(11, 7))

    draw_field(ax)

    # anchors
    for (px, py), col in zip([a, b, c], ["r", "m", "b"]):
        ax.plot(px, py, col + "o")

    # circles
    for (px, py), d, col in zip([a, b, c], [a_d, b_d, c_d], ["r", "m", "b"]):
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(px + d * np.cos(th), py + d * np.sin(th), col + "--", alpha=0.7)

    # intersection points
    ax.plot([ab_x, ac_x, bc_x], [ab_y, ac_y, bc_y], "kx")

    # final mean
    ax.plot(x, y, "m*", markersize=14)

    # padding around field
    ax.set_xlim(-K, A + K)
    ax.set_ylim(-K, B + K)

    ax.set_aspect("equal")
    ax.set_facecolor("green")
    plt.show()
    # ------------------


def triangulation_2p(a, b, a_d, b_d):
    """Performs triangulation to compute the position of the Robot."""

    # locations of tags
    # april_tags = {
    #     "tag1": [9.0, 3.0],
    #     "tag2": [7.350, 0.0],
    #     "tag3": [7.350, 6.0],
    #     "tag4": [4.5, 0.0],
    #     "tag5": [4.5, 6.0],
    #     "tag9": [0.0, 3.0],
    # }

    # # get detected tags
    # a = detections[0]["id"]
    # b = detections[1]["id"]
    # [ax, ay] = april_tags[a]
    # [bx, by] = april_tags[b]

    # [a_d, b_d] = distances
    [ax, ay] = a
    [bx, by] = b

    # get euclidean distance between a and b
    d = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    if d > a_d + b_d:
        print("circles dont intersect")

    # get triangle
    z = (a_d**2 - b_d**2 + d**2) / (2 * d)
    h = np.sqrt(a_d**2 - z**2)
    cx = ax + z * (bx - ax) / d
    cy = ay + z * (by - ay) / d

    # get possible robot coordinates
    qx = cx + h * (by - ay) / d
    qy = cy - h * (bx - ax) / d
    px = cx - h * (by - ay) / d
    py = cy + h * (bx - ax) / d

    # remove the extraneous solution that lies outside the field.
    if qx < 0 or qx > 9:  # field measured in m
        return [px, py]
    elif qy < 0 or qy > 6:  # field measured in m
        return [px, py]
    return [qx, qy]


def get_rotation(ax, ay, bx, by, robot_x, robot_y, camera_):
    # compute angles from each point to camera (in radians)
    theta1 = np.arctan2(ay - robot_y, ax - robot_x)
    theta2 = np.arctan2(by - robot_y, bx - robot_y)

    # take the average angle
    theta_camera = (theta1 + theta2) / 2

    # go from camera to robot orientation

    return theta_camera


def curling_algorithm(ducky_pos, robot_pos, image):
    twist = Twist()

    # calculate the angle between the rover and ducky.
    rotation = np.arctan2(ducky_pos[1] - robot_pos[1], ducky_pos[0] - robot_pos[0])
    # calculate how much the rover needs to turn and normalize (in radians)
    z_error = rotation - robot_pos[2]
    z_error = (z_error + np.pi) % (2 * np.pi) - np.pi

    # check how far away you are (in m)
    distance_to_ducky = np.sqrt(
        (robot_pos[0] - ducky_pos[0]) ** 2 + (robot_pos[1] - ducky_pos[1]) ** 2
    )

    # if you are close, reduce speed to be more precise
    if distance_to_ducky < 1:
        speed = 0.1
        turning = 0.3
    else:
        speed = 0.3
        turning = 0.5

    if z_error < 0.3:
        # drive fowards if in the right direction
        twist.linear.x = speed
        twist.angular.z = 0.0
    else:
        twist.linear.x = 0
        # otherwise turn to the right direction
        if z_error > 0.3:
            twist.angular.z = turning
        else:
            twist.angular.z = -turning

    if distance_to_ducky < 0.10:
        # if within 10 cm of the ducky, stop
        twist.linear.x = 0.0
        twist.angular.z = 0.0


def main(args=None):
    triangulation_3p()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
