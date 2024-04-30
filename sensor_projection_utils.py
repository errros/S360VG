import numpy as np
import pandas as pd
import math


# code for dataset in paper :  A Dataset for Exploring User Behaviors in VR Spherical Video Streaming

def euler_to_quaternion(roll, pitch, yaw,is_radian=True):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion vector.

    Args:
    - roll: Rotation around the x-axis in radians.
    - pitch: Rotation around the y-axis in radians.
    - yaw: Rotation around the z-axis in radians.

    Returns:
    - A tuple representing the quaternion (w, x, y, z).
    """
    if not is_radian:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def direction_vect_from_quaternion_in_cartesian(q):
    # Normalize the quaternion

    magnitude = np.linalg.norm(q)
    normalized_q = q / magnitude

    # Calculate the components of the forward vector
    x = 2 * (normalized_q[1] * normalized_q[3] - normalized_q[0] * normalized_q[2]) #+ q[4]
    y = 2 * (normalized_q[2] * normalized_q[3] + normalized_q[0] * normalized_q[1]) #+ q[5]
    z = 1 - 2 * (normalized_q[1] ** 2 + normalized_q[2] ** 2) #+ q[6]

    return [x, y, z]


def quaternion_from_direction_vect_cartesian(V):
    # Normalize the input vector
    V = np.array(V)
    V_norm = np.linalg.norm(V)
    if V_norm == 0:
        return [1, 0, 0, 0]  # Return identity quaternion if V is zero vector
    V /= V_norm

    # Original forward direction (z-axis)
    forward = np.array([0, 0, 1])

    # Compute the dot product and angle between vectors
    dot = np.dot(forward, V)

    # If the vectors are in the same direction, return the identity quaternion
    if dot > 0.999999:
        return [1, 0, 0, 0]

    # If the vectors are opposite, find an orthogonal vector for the rotation axis
    if dot < -0.999999:
        # Use an arbitrary vector to find the orthogonal axis
        axis = np.cross(forward, [1, 0, 0])
        if np.linalg.norm(axis) < 0.000001:  # If parallel, use another vector
            axis = np.cross(forward, [0, 1, 0])
        axis = axis / np.linalg.norm(axis)
    else:
        # Use cross product to find the rotation axis
        axis = np.cross(forward, V)

    axis_norm = np.linalg.norm(axis)
    if axis_norm:  # To avoid division by zero
        axis /= axis_norm

    # Compute quaternion components
    angle = np.arccos(dot)
    s = np.sin(angle / 2)
    q_w = np.cos(angle / 2)
    q_x, q_y, q_z = axis * s

    return [q_w, q_x, q_y, q_z]



##reduce the convertion levels hoping to get more accurate conversions
def quaternion_from_direction_vect_spherical(V_spherical,is_radian=True):
    # Convert angles from degrees to radians for computation
    theta_rad = V_spherical[0]
    phi_rad = V_spherical[1]
    if not is_radian:
        theta_rad = math.radians(theta_rad)
        phi_rad = math.radians(phi_rad)

    # Compute quaternion components
    q_w = math.cos(phi_rad / 2) * math.cos(theta_rad / 2)
    q_x = math.cos(phi_rad / 2) * math.sin(theta_rad / 2)
    q_y = math.sin(phi_rad / 2) * math.cos(theta_rad / 2)
    q_z = math.sin(phi_rad / 2) * math.sin(theta_rad / 2)

    return [q_w, q_x, q_y, q_z]


def vect_cartesian_to_spherical(V):
    # Ensure the vector is normalized
    V /= np.linalg.norm(V)

    # Convert vector to spherical coordinates (theta, phi)
    theta = np.arctan2(-V[0], -V[2])
    phi = np.arcsin(-V[1])
    return [theta, phi]


def vect_spherical_to_cartesian(V_spherical):
    theta, phi = V_spherical[0], V_spherical[1]

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return [x, y, z]


def project_spherical_vect_to_equirect_point(V_spherical):
    theta, phi = V_spherical[0], V_spherical[1]

    x = (theta + np.pi *  3/ 2) / (2 * np.pi)
    y = (phi + np.pi / 2) / np.pi

    return [x, y]


def equirect_point_to_spherical(p):
    x, y = p[0], p[1]

    theta = x * (2 * np.pi) - np.pi
    phi = y * np.pi - np.pi / 2

    return [theta, phi]


def equirect_point_to_pixels(p, resolution):
    x, y = p[0], p[1]
    x_pixel = x * resolution[0]
    x_pixel = resolution[0] - x_pixel
    if x_pixel < 0:
        x_pixel = resolution[0] + x_pixel

    y_pixel = y * resolution[1]

    return [x_pixel, y_pixel]


def pixels_to_equirect_point(pixels, resolution):
    x_pixel, y_pixel = pixels[0], pixels[1]

    x = (resolution[0] - x_pixel) / resolution[0]
    x = x * (2 * np.pi) - np.pi * 3 / 2

    y = y_pixel / resolution[1]
    y = y * np.pi - np.pi / 2

    return [x, y]


#
# q1 = [0.643,0.032,0.763,-0.047]
# print(f'original quaternion <{q1}>')
# direct_vect1 = direction_vect_from_quaternion_in_cartesian(q1)
# sph = vect_cartesian_to_spherical(direct_vect1)
# print(f'direction vector<{direct_vect1}>')
# q_inv1 = quaternion_from_direction_vect_cartesian(direct_vect1)
# q_inv2 = quaternion_from_direction_vect_spherical(sph)
# print(f'inverse quaternion <{q_inv1}>')
# print(f'inverse quaternion from sph <{q_inv2}>')
# print(np.abs(np.dot(q_inv1, q1)))
# print(np.abs(np.dot(q_inv2, q1)))
#
#
# print('------------------------------')
#
#
# q1 = [0.536,0.058,0.833,-0.127]
# print(f'original quaternion <{q1}>')
# direct_vect1 = direction_vect_from_quaternion_in_cartesian(q1)
# sph = vect_cartesian_to_spherical(direct_vect1)
# print(f'direction vector<{direct_vect1}>')
# q_inv1 = quaternion_from_direction_vect_cartesian(direct_vect1)
# q_inv2 = quaternion_from_direction_vect_spherical(sph)
# print(f'inverse quaternion <{q_inv1}>')
# print(f'inverse quaternion from sph <{q_inv2}>')
# print(np.abs(np.dot(q_inv1, q1)))
# print(np.abs(np.dot(q_inv2, q1)))
#
# print('------------------------------')
#
#
# q1 = [-0.149,-0.014,0.987,-0.054]
# print(f'original quaternion <{q1}>')
# direct_vect1 = direction_vect_from_quaternion_in_cartesian(q1)
# sph = vect_cartesian_to_spherical(direct_vect1)
# print(f'direction vector<{direct_vect1}>')
# q_inv1 = quaternion_from_direction_vect_cartesian(direct_vect1)
# q_inv2 = quaternion_from_direction_vect_spherical(sph)
# print(f'inverse quaternion <{q_inv1}>')
# print(f'inverse quaternion from sph <{q_inv2}>')
# print(np.abs(np.dot(q_inv1, q1)))
# print(np.abs(np.dot(q_inv2, q1)))



