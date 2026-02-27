import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from chasles import screw_from_twist, generate_trajectory_from_twist, hat


FRAME_COLORS = ['#e74c3c', '#2ecc71', '#3498db']   # x=red, y=green, z=blue
AXIS_COLOR   = '#f39c12'
TRACE_COLOR  = '#9b59b6'
FRAME_ALPHA  = 0.9


def _draw_frame(ax, g: np.ndarray, scale: float = 1.2,
                alpha: float = FRAME_ALPHA, linewidth: float = 2.5):
    """Draw an RGB triad for the frame defined by g (4×4 SE(3))."""
    origin = g[:3, 3]
    artists = []
    for i, color in enumerate(FRAME_COLORS):
        direction = g[:3, i]
        end = origin + scale * direction
        line, = ax.plot([origin[0], end[0]],
                        [origin[1], end[1]],
                        [origin[2], end[2]],
                        color=color, linewidth=linewidth, alpha=alpha)
        artists.append(line)
    return artists


def _draw_screw_axis(ax, omega: np.ndarray, q: np.ndarray,
                     h: float, axis_len: float = 3.0):
    """
    Draw the screw axis as a dashed line through q in direction omega.
    For pure translation (h=inf) draw the translation direction instead.
    """
    if np.isinf(h):
        start = q - axis_len / 2 * omega
        end   = q + axis_len / 2 * omega
        label = "Translation axis"
    else:
        start = q - axis_len / 2 * omega
        end   = q + axis_len / 2 * omega
        label = f"Screw axis  (h = {h:.3f})"

    ax.plot([start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=AXIS_COLOR, linewidth=2, linestyle='--',
            label=label, zorder=5)

    # small dot at q
    ax.scatter(*q, color=AXIS_COLOR, s=50, zorder=6)
    ax.text(q[0], q[1], q[2] + 0.15, ' q', color=AXIS_COLOR, fontsize=9)


def _auto_axis_limits(
    frames: list[np.ndarray],
    omega: np.ndarray,
    q: np.ndarray,
    padding: float = 0.2,
) -> float:
    """Return a reasonable half-extent for the plot cube, based on motion and q."""
    positions = np.array([f[:3, 3] for f in frames])
    pts = np.vstack([positions, q.reshape(1, 3)])
    extent = np.max(np.abs(pts)) + padding
    return max(extent, 0.5)


def make_screw_figure(frames: np.ndarray,
                      omega: np.ndarray,
                      q: np.ndarray,
                      h: float,
                      trace: bool = True):
    """
    Create a static 3D matplotlib figure visualizing the screw motion.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (N, 4, 4) of SE(3) transforms along the motion.
    omega : np.ndarray
        Unit rotation axis (3,).
    q : np.ndarray
        A point on the screw axis (3,).
    h : float
        Pitch of the screw (or np.inf for pure translation).
    trace : bool
        If True, draw the trajectory of the origin.
    """
    if frames.ndim != 3 or frames.shape[1:] != (4, 4):
        raise ValueError("frames must have shape (N, 4, 4)")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    extent = _auto_axis_limits(list(frames), omega, q)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    # Draw screw axis and q, scaled to be very prominent in the plot cube
    _draw_screw_axis(ax, omega, q, h, axis_len=6.0 * extent)

    # Draw initial and final frames
    _draw_frame(ax, frames[0])
    _draw_frame(ax, frames[-1], alpha=0.5)

    # Label origin of the initial frame as O
    origin0 = frames[0][:3, 3]
    ax.scatter(*origin0, color="white", s=20)
    ax.text(
        origin0[0],
        origin0[1],
        origin0[2],
        " O",
        color="white",
        fontsize=8,
    )

    # Label origin of the final frame as T
    originT = frames[-1][:3, 3]
    ax.scatter(*originT, color="white", s=20)
    ax.text(
        originT[0],
        originT[1],
        originT[2],
        " T",
        color="white",
        fontsize=8,
    )

    # Trace of origin
    if trace:
        positions = np.array([g[:3, 3] for g in frames])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                color=TRACE_COLOR, linewidth=2, alpha=0.8)

    ax.set_title("Screw Motion")

    return fig


def make_gif_for_twist(
    twist: np.ndarray,
    n_frames: int = 60,
    fps: int = 20,
    background: str = "black",
) -> bytes:
    """
    Generate an animated GIF visualizing the rigid-body motion
    corresponding to a twist.

    Parameters
    ----------
    twist : np.ndarray
        6D twist vector (v, w).
    n_frames : int
        Number of frames in the animation.
    fps : int
        Frames per second for the GIF.

    Returns
    -------
    bytes
        GIF image bytes suitable for display in Streamlit.
    """
    frames = generate_trajectory_from_twist(twist, n_steps=n_frames)
    omega, q, h = screw_from_twist(twist)

    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor(background)
    ax = fig.add_subplot(111, projection='3d')

    extent = _auto_axis_limits(list(frames), omega, q)

    text_color = "white" if background.lower() == "black" else "black"

    omega_str = f"ω = [{omega[0]: .2f}, {omega[1]: .2f}, {omega[2]: .2f}]"
    q_str = f"q = [{q[0]: .2f}, {q[1]: .2f}, {q[2]: .2f}]"
    h_str = f"h = {h:.2f}"

    def _setup_axes():
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_facecolor(background)
        ax.set_box_aspect([1, 1, 1])

    def init():
        ax.clear()
        _setup_axes()
        _draw_screw_axis(ax, omega, q, h, axis_len=6.0 * extent)
        return []

    def update(i):
        ax.clear()
        _setup_axes()
        _draw_screw_axis(ax, omega, q, h, axis_len=6.0 * extent)

        # Initial, final, and current frames
        _draw_frame(ax, frames[0])
        _draw_frame(ax, frames[-1], alpha=0.5)
        _draw_frame(ax, frames[i])

        # Label origin of the initial frame as O
        origin0 = frames[0][:3, 3]
        ax.scatter(*origin0, color="white", s=20)
        ax.text(
            origin0[0],
            origin0[1],
            origin0[2],
            " O",
            color="white",
            fontsize=8,
        )

        # Label origin of the final frame as T
        originT = frames[-1][:3, 3]
        ax.scatter(*originT, color="white", s=20)
        ax.text(
            originT[0],
            originT[1],
            originT[2],
            " T",
            color="white",
            fontsize=8,
        )

        # Trace up to current frame
        if i > 0:
            positions = np.array([g[:3, 3] for g in frames[: i + 1]])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color=TRACE_COLOR, linewidth=2, alpha=0.8)

        # Line between p and q + hat(omega)^2 (p - q)
        p_t = frames[i][:3, 3]
        v = p_t - q
        omega_hat = hat(omega)
        proj_point = p_t + omega_hat @ (omega_hat @ v) / np.linalg.norm(omega)**2
        ax.plot(
            [p_t[0], proj_point[0]],
            [p_t[1], proj_point[1]],
            [p_t[2], proj_point[2]],
            color="white",
            linewidth=1.5,
            alpha=0.9,
        )

        ax.set_title("Screw Motion Animation")

        # Text overlay in the bottom-left corner of the axes (in figure coordinates)
        ax.text2D(
            0.02,
            0.06,
            f"{omega_str}\n{q_str}\n{h_str}",
            transform=ax.transAxes,
            color=text_color,
            fontsize=8,
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames, blit=False
    )

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as tmp:
        anim.save(tmp.name, writer="pillow", fps=fps)
        tmp.seek(0)
        data = tmp.read()

    plt.close(fig)
    return data
