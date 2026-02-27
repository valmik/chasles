from token import NEWLINE
import numpy as np
import streamlit as st

from chasles import (
    given_v_omega_theta,
    given_rbt,
    screw_from_twist,
    check_R,
    interpolate_rbt,
    g_to_Rp,
)
from animate_rbt import make_gif_for_twist


st.set_page_config(page_title="Chasles Theorem Visualizer", layout="wide")


def _init_state():
    if "v" not in st.session_state:
        st.session_state.v = np.zeros(3, dtype=float)
    if "omega" not in st.session_state:
        st.session_state.omega = np.array([0.0, 0.0, 1.0], dtype=float)
    if "theta" not in st.session_state:
        st.session_state.theta = 1.0
    if "R" not in st.session_state:
        st.session_state.R = np.eye(3, dtype=float)
    if "p" not in st.session_state:
        st.session_state.p = np.zeros(3, dtype=float)
    if "gif_bytes" not in st.session_state:
        st.session_state.gif_bytes = None


def _inputs_v_omega_theta():
    col_v, col_omega = st.columns(2)

    v_default = np.asarray(st.session_state.v, dtype=float)
    omega_default = np.asarray(st.session_state.omega, dtype=float)
    theta_default = float(st.session_state.theta)

    with col_v:
        st.subheader("Linear part v")
        v_x = st.number_input("v_x", value=float(v_default[0]))
        v_y = st.number_input("v_y", value=float(v_default[1]))
        v_z = st.number_input("v_z", value=float(v_default[2]))

    with col_omega:
        st.subheader("Angular part ω")
        omega_x = st.number_input("ω_x", value=float(omega_default[0]))
        omega_y = st.number_input("ω_y", value=float(omega_default[1]))
        omega_z = st.number_input("ω_z", value=float(omega_default[2]))

    theta = st.number_input("θ (theta)", value=theta_default)

    v = np.array([v_x, v_y, v_z], dtype=float)
    omega = np.array([omega_x, omega_y, omega_z], dtype=float)

    st.session_state.v = v
    st.session_state.omega = omega
    st.session_state.theta = float(theta)

    return v, omega, float(theta)


def _inputs_R_p():
    st.subheader("Rotation matrix R (3×3)")
    cols = st.columns(3)

    R_default = np.asarray(st.session_state.R, dtype=float)
    R = np.zeros((3, 3), dtype=float)
    for i in range(3):
        with cols[i]:
            for j in range(3):
                default_val = float(R_default[i, j])
                R[i, j] = st.number_input(f"R[{i+1},{j+1}]", value=default_val)

    st.subheader("Translation vector p")
    col_px, col_py, col_pz = st.columns(3)

    p_default = np.asarray(st.session_state.p, dtype=float)
    with col_px:
        p_x = st.number_input("p_x", value=float(p_default[0]))
    with col_py:
        p_y = st.number_input("p_y", value=float(p_default[1]))
    with col_pz:
        p_z = st.number_input("p_z", value=float(p_default[2]))

    p = np.array([p_x, p_y, p_z], dtype=float)

    st.session_state.R = R
    st.session_state.p = p

    return R, p


def _update_state_from_twist(twist: np.ndarray):
    """
    Given a twist where the rotational part encodes ω θ, update
    session_state with compatible (v, ω, θ) and (R, p).
    """
    v_part = twist[:3]
    w_part = twist[3:]

    theta = float(np.linalg.norm(w_part))
    if theta > 0:
        omega = w_part / theta
        v = v_part / theta
    else:
        # Pure translation: keep omega along z by convention
        omega = np.array([0.0, 0.0, 1.0], dtype=float)
        v = v_part
        theta = 0.0

    g = interpolate_rbt(twist, 1.0)
    R, p = g_to_Rp(g)

    st.session_state.v = v
    st.session_state.omega = omega
    st.session_state.theta = theta
    st.session_state.R = R
    st.session_state.p = p


def main():
    _init_state()

    st.title("Chasles Theorem Visualizer")
    st.markdown(
        "Explore rigid-body motions via Chasles' theorem!  \n"
        "Enter either a twist $(v, \\omega)\\theta$ or a rigid-body transform $(R, p)$, "
        "and visualize the corresponding screw motion.  \n"
        "By Valmik Prabhu (and Claude)"
    )

    # Narrow left column for inputs (≈20%), wide right column for animation (≈80%)
    col_left, col_right = st.columns([1, 4])

    with col_left:
        top_left, top_right = st.columns([3, 2])
        with top_left:
            mode = st.radio("Input mode", ("v, omega, theta", "R, p"))
        with top_right:
            if st.button("Random twist"):
                # Random v in [-3, 3]^3
                v_rand = np.random.uniform(-3.0, 3.0, size=3)
                # Random unit omega
                while True:
                    w = np.random.uniform(-1.0, 1.0, size=3)
                    n = np.linalg.norm(w)
                    if n > 1e-6:
                        omega_rand = w / n
                        break
                # Random theta in [-2π, 2π]
                theta_rand = float(np.random.uniform(-2.0 * np.pi, 2.0 * np.pi))

                st.session_state.v = v_rand
                st.session_state.omega = omega_rand
                st.session_state.theta = theta_rand

                twist_rand = given_v_omega_theta(v_rand, omega_rand, theta_rand)
                _update_state_from_twist(twist_rand)

        if mode == "v, omega, theta":
            v, omega, theta = _inputs_v_omega_theta()
        else:
            R, p = _inputs_R_p()

        compute = st.button("Compute & animate")

    if compute:
        try:
            if mode == "v, omega, theta":
                twist = given_v_omega_theta(v, omega, theta)
            else:
                if not check_R(R):
                    with col_left:
                        st.error(
                            "R is not a valid rotation matrix "
                            "(R^T R ≈ I, det(R) ≈ 1 required)."
                        )
                    return
                twist = given_rbt(R, p)

            _update_state_from_twist(twist)

            with col_right:
                try:
                    st.session_state.gif_bytes = make_gif_for_twist(
                        twist, n_frames=60, fps=20
                    )
                except Exception as e:
                    st.warning(f"Could not generate GIF animation: {e}")

        except Exception as e:
            with col_left:
                st.error(f"Error while computing screw/trajectory: {e}")

    # Always show the latest GIF (if any) on the right, even after mode changes.
    with col_right:
        if st.session_state.gif_bytes is not None:
            st.image(
                st.session_state.gif_bytes,
                caption="Screw motion animation",
            )


if __name__ == "__main__":
    main()


