import numpy as np


def rot2eul(r):
    """
    Method for computing Euler angles of X-Y-X rotation given rotation matrix r

    We assume that the first X rotation is performed by angle ψ, second Y rotation by angle θ
    and last X rotation by φ
    As this is X-Y-X rotation, matrix will have the form
    [      cos(θ),               sin(ψ)*sin(φ),                        sin(θ)*cos(φ)              ]
    [  sin(ψ)*sin(θ), cos(ψ)*cos(φ) - sin(ψ)*cos(θ)*sin(φ), -sin(ψ)*sin(φ) - sin(ψ)*cos(θ)*cos(φ) ]
    [ -cos(ψ)*sin(θ), sin(ψ)*cos(φ) + sin(φ)*cos(θ)*cos(ψ), -sin(ψ)*sin(φ) + cos(ψ)*cos(θ)*cos(φ) ]

    For the simpler explanation, we will assume the matrix above to be
    [ R11, R12, R13 ]
    [ R21, R22, R33 ]
    [ R31, R32, R33 ]

    θ = arccos(R11)
    ψ = arctan2(R21, R31)
    Angle φ might be trickier to find, as simple arctan2(R12, R13) won't be enough,
    as we will still have sin(ψ)/sin(θ). But, as we know the angles ψ and θ from
    previous equations, we can divide R12 by sin(ψ) and R13 by sin(θ).
    With that in mind,
    φ = arctan2(R12/sin(ψ), R13/sin(θ))

    If ψ = pi/2, then cos(ψ) = 0 and we cannot use arctan to solve for this angle.
    But, as we already know the value of θ, we can use it to compute angle ψ from R21.
    ψ = arcsin(R21/sin(θ))

    If ψ and θ are zero, then sin(ψ) and sin(θ) are zero too and we can't compute
    last angle by division. But, in this case, R22 becomes equal just to cos(φ),
    as sines become zero and other cosines become 1.
    φ = arccos(R22)

    If just one of ψ and θ is zero, we still can't use division as we wanted. But,
    in this case, we can take R12 or R13, depending on which angle is not zero,
    and use arcsin or arccos respectively.
    φ = arccos(R13/sin(θ))
    φ = arcsin(R12/sin(ψ))

    Parameters
    ----------
    r
        rotation matrix

    Returns
    -------
    Tuple of three Euclidian angles, in the order ψ - θ - φ
    """
    theta = 0
    psi = 0
    phi = 0

    theta = np.arccos(r[1][1])

    if r[3][1].equals(np.sin(theta)):
        psi = np.arcsin(r[2][1]/np.sin(theta))
    else:
        psi = np.arctan2(r[2][1], r[3][1])

    if theta.equals(0) and not psi.equals(0):
        phi = np.arcsin(r[1][2]/np.sin(psi))
    elif psi.equals(0) and not theta.equals(0):
        phi = np.arccos(r[1][3]/np.sin(theta))
    elif psi.equals(0) and theta.equals(0):
        phi = np.arccos(r[2][2])
    else:
        phi = np.arctan2(r[1][2]/np.sin(psi), r[1][3]/np.sin(theta))
    return psi, theta, phi


def fk(j1, j2, j3, j4, j5, j6):
    """
    Forward kinematics solution using matrices

    Implements forward kinematics solution for FANUC 2000iC through transformation and rotation matrices.
    All transformation matrices were carefully calculated using official datasheet available at fanuc.eu
    FANUC 2000iC is a yellow 6DOF robotic manipulator. We will use the following matrices:
    R1, R2, R3, R4, R5, R6, T1, T2, T34, T5

    R1 is a Z-rotation matrix with the following form:
    [ cos(j1), -sin(j1), 0, 0 ]
    [ sin(j1),  cos(j1), 0, 0 ]
    [    0,        0,    1, 0 ]
    [    0,        0,    0, 1 ]

    R2, R3 and R5 are X-rotation matrices with the following form:
    [ 1,   0,       0,    0 ]
    [ 0, cos(j), -sin(j), 0 ]
    [ 0, sin(j),  cos(j), 0 ]
    [ 0,   0,       0,    1 ]

    R4 and R6 are Y-rotation matrices with the following form:
    [  cos(j), 0, sin(j), 0 ]
    [    0,    1,   0,    0 ]
    [ -sin(j), 0, cos(j), 0 ]
    [    0,    0,   0,    1 ]

    T1 is a YZ-translation matrix from base point to center of J2.

    T2 is a Z-translation matrix from center of J2 to center of J3.

    T34 is a YZ-translation matrix from center of J3 to center of J5.
    The rationale behind combining translations from J3 to J4 and from J4 to J5 is due to
    lack of such dimensions in documentation and J4 making no effect on position of J5
    with respect to J3.

    T5 is a Y-translation matrix from center of J5 to end-effector (assumed to be endpoint of the manipulator).

    All translation matrices have the following form:
    [ 0, 0, 0, x ]
    [ 0, 0, 0, y ]
    [ 0, 0, 0, z ]
    [ 0, 0, 0, 1 ]
    x, y and z are given in mm.

    In order to obtain the final transformation, we must perform the following computation:
    T = R1 * T1 * R2 * T2 * R3 * R4 * T34 * R5 * T5 * R6

    Parameters
    ----------
    j1
        Rotation matrix for joint 1
    j2
        Rotation matrix for joint 2
    j3
        Rotation matrix for joint 3
    j4
        Rotation matrix for joint 4
    j5
        Rotation matrix for joint 5
    j6
        Rotation matrix for joint 6
    Returns
    -------
    Homogeneous transformation matrix for resulting end effector position
    """
    r1 = np.array([[np.cos(j1), -np.sin(j1), 0, 0],
                   [np.sin(j1), np.cos(j1), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    r2 = np.array([[1, 0, 0, 0],
                   [0, np.cos(j2), -np.sin(j2), 0],
                   [0, np.sin(j2), np.sin(j2), 0],
                   [0, 0, 0, 1]])
    r3 = np.array([[1, 0, 0, 0],
                   [0, np.cos(j3), -np.sin(j3), 0],
                   [0, np.sin(j3), np.sin(j3), 0],
                   [0, 0, 0, 1]])
    r4 = np.array([[np.cos(j4), 0, np.sin(j4), 0],
                   [0, 1, 0, 0],
                   [-np.sin(j4), 0, np.cos(j4), 0],
                   [0, 0, 0, 1]])
    r5 = np.array([[1, 0, 0, 0],
                   [0, np.cos(j5), -np.sin(j5), 0],
                   [0, np.sin(j5), np.sin(j5), 0],
                   [0, 0, 0, 1]])
    r6 = np.array([[np.cos(j6), 0, np.sin(j6), 0],
                   [0, 1, 0, 0],
                   [-np.sin(j6), 0, np.cos(j6), 0],
                   [0, 0, 0, 1]])

    t1 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 312],
                   [0, 0, 0, 670],
                   [0, 0, 0, 1]])
    t2 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1075],
                   [0, 0, 0, 1]])
    t34 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 1280],
                   [0, 0, 0, 275],
                   [0, 0, 0, 1]])
    t5 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 215],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1]])

    ht = np.dot(r6, np.dot(t5, np.dot(r5, np.dot(t34, np.dot(r4, np.dot(r3, np.dot(t2, np.dot(r2, np.dot(t1, r1)))))))))
    return ht
