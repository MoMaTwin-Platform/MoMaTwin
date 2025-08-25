import numpy as np
from scipy.spatial.transform import Rotation as R

def so3_to_lie_algebra(R):
    """
    Convert SO(3) rotation matrix to lie algebra representation.
    
    Args:
    R (np.array): 3x3 rotation matrix
    
    Returns:
    np.array: 3x1 lie algebra representation
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    if np.abs(theta) < 1e-4:  # Close to zero
        return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / 2
    elif np.abs(theta - np.pi) < 1e-4:  # Close to pi
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            w = np.array([R[0,0] + 1, R[1,0] + R[0,1], R[2,0] + R[0,2]])
        elif R[1,1] > R[2,2]:
            w = np.array([R[1,0] + R[0,1], R[1,1] + 1, R[2,1] + R[1,2]])
        else:
            w = np.array([R[2,0] + R[0,2], R[2,1] + R[1,2], R[2,2] + 1])
        w = w / np.sqrt(np.max([R[0,0], R[1,1], R[2,2]]) + 1) * theta
        return w
    else:  # General case
        w = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        return w * theta / (2 * np.sin(theta))

def lie_algebra_to_so3(w):
    """
    Convert lie algebra representation to SO(3) rotation matrix.
    
    Args:
    w (np.array): 3x1 lie algebra representation
    
    Returns:
    np.array: 3x3 rotation matrix
    """
    theta = np.linalg.norm(w)
    
    if theta < 1e-6:  # Close to zero
        return np.eye(3) + skew(w)
    else:
        K = skew(w / theta)
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

def skew(v):
    """
    Create a skew-symmetric matrix from a 3D vector.
    
    Args:
    v (np.array): 3x1 vector
    
    Returns:
    np.array: 3x3 skew-symmetric matrix
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrix.
    
    Args:
    eul_angle : Euler angles in degrees
    
    Returns:
    np.array: 3x3 rotation matrix
    """
    # Create a rotation object from Euler angles
    rot = R.from_euler('zyx', euler_angles, degrees=False)
    
    return rot.as_matrix()

def rotation_matrix_to_euler_angles(matrix, seq='zyx'):
    """
    Convert a rotation matrix to Euler angles.
    
    Args:
    matrix : 3x3 rotation matrix
    seq : Sequence of axes for rotation. Default is 'zyx' for intrinsic rotations.
    
    Returns:
    np.array: Euler angles in degrees [yaw, pitch, roll] for 'zyx' sequence
    """
    # Create a Rotation object from the matrix
    rot = R.from_matrix(matrix)

    # Convert to Euler angles
    euler_angles = rot.as_euler(seq, degrees=False)
    
    return euler_angles

if __name__ == "__main__":
    orientation = [0.04026075452566147, 0.6584011912345886, 0.5670892596244812]

    R0 = euler_to_rotation_matrix(orientation)
    w = so3_to_lie_algebra(R0)

    updated_R = lie_algebra_to_so3(w)
    new_orientation = rotation_matrix_to_euler_angles(updated_R)

    print("orient diff: ", np.sum(new_orientation-orientation))