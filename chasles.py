import numpy as np
import scipy.linalg as la

def hat(w: np.ndarray) -> np.ndarray:
    """ Skew symmetric operator """
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def vee(what: np.ndarray) -> np.ndarray:
    """ Inverse of the skew symmetric operator """
    return np.array([what[2,1], what[0,2], what[1,0]])

def wedge(twist: np.ndarray) -> np.ndarray:
    """ Convert the twist to an element of se(3) """

    v = twist[:3]
    w = twist[3:]

    se3 = np.zeros((4,4))
    se3[:3, :3] = hat(w)
    se3[:3, 3] = v

    return se3

def unwedge(se3: np.ndarray) -> np.ndarray:
    """ Convert the element of se(3) to a twist """
    v = se3[:3, 3]
    w = vee(se3[:3, :3])
    return np.concatenate([v, w])

def Rp_to_g(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ Convert p and R to an element of SE(3) """
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = p
    return g

def g_to_Rp(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Convert the element of SE(3) to p and R """
    R = g[:3, :3]
    p = g[:3, 3]
    return R, p

def check_R(R: np.ndarray) -> bool:
    """ Check if the rotation matrix is valid """
    return np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)

def screw_from_twist(twist: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert the twist to a screw.
    """
    v = twist[:3]
    w = twist[3:]

    theta = np.linalg.norm(w)
    if theta == 0:
        q = np.zeros(3)
        theta = np.linalg.norm(v)
        return v / theta, q, np.inf

    else:
        omega = w / theta
        v = v / theta

        q = hat(omega) @ v
        h = np.dot(omega, v)

        return omega, q, h

def interpolate_rbt(twist: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP style interpolation of the RBT from the twist and the index t in [0, 1]   
    """

    se3 = wedge(twist)
    return la.expm(se3 * t)


def generate_trajectory_from_twist(twist: np.ndarray,
                                   n_steps: int = 100) -> np.ndarray:
    """
    Generate a sequence of homogeneous transforms g(t) for t in [0, 1].

    Parameters
    ----------
    twist : np.ndarray
        6D twist vector (v, w).
    n_steps : int
        Number of interpolation steps.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps, 4, 4) with SE(3) matrices.
    """
    ts = np.linspace(0.0, 1.0, n_steps)
    trajectory = np.empty((n_steps, 4, 4))
    se3 = wedge(twist)
    for i, tau in enumerate(ts):
        trajectory[i] = la.expm(se3 * tau)
    return trajectory

def given_rbt(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ 
    Given the RBT, return the twist.
    """
    if not check_R(R):
        raise ValueError("Invalid rotation matrix")

    g = Rp_to_g(R, p)
    xihat = la.logm(g)
    twist = unwedge(xihat)
    return twist

def given_v_omega_theta(v: np.ndarray, omega: np.ndarray, theta: float) -> np.ndarray:
    """
    Given the twist and the index t in [0, 1], return the RBT.
    """
    twist = np.concatenate([v, omega])*theta
    return twist

def main():
    """
    Main function.
    """
    input_method = "v_omega_theta"

    if input_method == "v_omega_theta":
        v = np.array(input("Enter the v: "))
        omega = np.array(input("Enter the omega: "))
        theta = float(input("Enter the theta: "))
        twist = given_v_omega_theta(v, omega, theta)
    elif input_method == "rbt":
        R = np.array(input("Enter the R: "))
        p = np.array(input("Enter the p: "))
        twist = given_rbt(R, p)
    else:
        print("Invalid input method")
        return

    omega, q, h = screw_from_twist(twist)

    initial_rbt = np.eye(4)
    final_rbt = la.expm(wedge(twist))

    ### Create 3D plot
    ### Plot the line q + t*omega
    ### Print the screw 
    ### Plot the initial and final RBT

    ### Animate
    for t in np.linspace(0, 1, 100):
        rbt = interpolate_rbt(twist, t)
        ## Plot the rbt

if __name__ == "__main__":
    main()