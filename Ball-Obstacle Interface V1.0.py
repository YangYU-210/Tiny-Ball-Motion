from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,QLabel, QLineEdit, QPushButton, QMessageBox, QFormLayout
import matplotlib.pyplot as plt
import numpy as np
import sys 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Ball:
    def __init__(self, x, y, vx, vy, r=0.2):
        # Position of the ball
        self.x = float(x)
        self.y = float(y)
        # Velocity of the ball
        self.vx = float(vx)
        self.vy = float(vy)
        # Radius of the ball
        self.r = float(r)

    def advance(self, dt):
        """
        Advance the ball position by one time step
        using simple Newtonian motion:
            x <- x + v * dt
        """
        self.x += self.vx * dt
        self.y += self.vy * dt


class RectObstacle:
    """
    Axis-aligned rectangular obstacle:
    region [x0, x1] × [y0, y1]
    """
    def __init__(self, x0, x1, y0, y1):
        # Ensure correct ordering of bounds
        self.x0, self.x1 = (min(x0, x1), max(x0, x1))
        self.y0, self.y1 = (min(y0, y1), max(y0, y1))

    def draw(self, ax):
        """Draw the rectangle boundary"""
        xs = [self.x0, self.x1, self.x1, self.x0, self.x0]
        ys = [self.y0, self.y0, self.y1, self.y1, self.y0]
        ax.plot(xs, ys, lw=2)

    def collide(self, ball: Ball):
        """
        Detect and resolve collision between the ball and the rectangle.

        Method:
        - Expand the rectangle by the ball radius (AABB method)
        - If the ball center enters this expanded region, a collision occurred
        - Reflect velocity along the direction of minimum penetration
        """

        # Expanded rectangle (accounts for ball radius)
        ex0 = self.x0 - ball.r
        ex1 = self.x1 + ball.r
        ey0 = self.y0 - ball.r
        ey1 = self.y1 + ball.r

        # Check if ball center is inside expanded rectangle
        inside = (ex0 <= ball.x <= ex1) and (ey0 <= ball.y <= ey1)
        if not inside:
            return False

        # Penetration depths to each side
        pen_left   = abs(ball.x - ex0)
        pen_right  = abs(ex1 - ball.x)
        pen_bottom = abs(ball.y - ey0)
        pen_top    = abs(ey1 - ball.y)

        # Choose the smallest penetration direction
        pens = [pen_left, pen_right, pen_bottom, pen_top]
        k = int(np.argmin(pens))

        # Reflect velocity based on collision normal
        if k == 0:          # collision with left side
            ball.x = ex0
            ball.vx *= -1
        elif k == 1:        # collision with right side
            ball.x = ex1
            ball.vx *= -1
        elif k == 2:        # collision with bottom side
            ball.y = ey0
            ball.vy *= -1
        else:               # collision with top side
            ball.y = ey1
            ball.vy *= -1

        return True


class World:
    """
    Simulation domain with square boundaries
    and (optionally) a single rectangular obstacle
    """
    def __init__(self, half_size=5.0, obstacle=None):
        # Half-size of the square domain [-h, h] × [-h, h]
        self.h = float(half_size)
        self.obstacle = obstacle

    def draw_bounds(self, ax):
        """Draw square simulation boundary"""
        h = self.h
        ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], lw=2)

    def bounce_on_walls(self, ball: Ball):
        """
        Reflect the ball on the domain boundaries
        by reversing the corresponding velocity component
        """
        h = self.h

        # Left / right walls
        if ball.x - ball.r < -h:
            ball.x = -h + ball.r
            ball.vx *= -1
        elif ball.x + ball.r > h:
            ball.x = h - ball.r
            ball.vx *= -1

        # Bottom / top walls
        if ball.y - ball.r < -h:
            ball.y = -h + ball.r
            ball.vy *= -1
        elif ball.y + ball.r > h:
            ball.y = h - ball.r
            ball.vy *= -1

    def step(self, ball: Ball, dt):
        """
        Perform one deterministic time step:
        1) Move the ball
        2) Handle wall collisions
        3) Handle obstacle collision
        """
        ball.advance(dt)
        self.bounce_on_walls(ball)
        if self.obstacle is not None:
            self.obstacle.collide(ball)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 6))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        
class ParticleGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ball + Obstacles Simulation")
        
        # Initial inputs
        self.edit_env_half = QLineEdit("5.0")   # environment half-size
        self.edit_ball_r = QLineEdit("0.3")     # ball radius
        self.edit_obs_w = QLineEdit("2.0")      # obstacle width
        self.edit_obs_h = QLineEdit("1.0")      # obstacle height
        self.edit_dt = QLineEdit("0.01")        # time step
        self.edit_step = QLineEdit("1000")      # total number of steps
        
        form = QFormLayout()
        form.addRow(QLabel("Environment half-size (H):"), self.edit_env_half)
        form.addRow(QLabel("Ball radius (r):"), self.edit_ball_r)
        form.addRow(QLabel("Obstacle width (w):"), self.edit_obs_w)
        form.addRow(QLabel("Obstacle height (h):"), self.edit_obs_h)
        form.addRow(QLabel("Time step (dt):"), self.edit_dt)
        form.addRow(QLabel("Total steps:"), self.edit_step)
        # Buttons
        self.btn_generate = QPushButton("Generate Image")
        self.btn_clear = QPushButton("Clear")
        
        
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_clear)
        
        # Canvas
        self.canvas = MplCanvas(self)
        
        # Layout
        left = QVBoxLayout()
        left.addLayout(form)
        left.addLayout(btn_row)
        left.addStretch(1)
        
        main = QHBoxLayout()
        main.addLayout(left, 0)
        main.addWidget(self.canvas, 1)
        self.setLayout(main)
        
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_clear.clicked.connect(self.on_clear)
        
    def on_generate(self):
        H = float(self.edit_env_half.text())
        r = float(self.edit_ball_r.text())
        w = float(self.edit_obs_w.text())
        h_obs = float(self.edit_obs_h.text())
        dt = float(self.edit_dt.text())
        steps = int(float(self.edit_step.text()))
            
        # Basic validation
        if H <= 0 or r <= 0 or w <= 0 or h_obs <= 0:
            raise ValueError("All sizes must be positive.")
        if r >= H:
            raise ValueError("Ball radius must be smaller than environment half-size.")
        if w >= 2 * H or h_obs >= 2 * H:
            raise ValueError("Obstacle is too large for the environment.")
                
        # Define a simple default placement:
        # - Ball at left-lower quadrant
        # - Obstacle centered at origin
        ball_x, ball_y = -0.6 * H, -0.6 * H
        obs_cx, obs_cy = 0.0, 0.0
            
        # Obstacle bounds
        x0, x1 = obs_cx - w / 2, obs_cx + w / 2
        y0, y1 = obs_cy - h_obs / 2, obs_cy + h_obs / 2
            
        # Initialize environment, obstacle, ball object
        obs = RectObstacle(x0, x1, y0, y1)
        world = World(H, obs)
        ball = Ball(ball_x, ball_y, vx = 1, vy = 1, r=r)
            
        traj = np.zeros((steps, 2), dtype=float)
        traj[0] = [ball.x, ball.y]
        for n in range(steps):
            world.step(ball, dt)
            traj[n] = [ball.x, ball.y]
            
        # Plot
        ax = self.canvas.ax
        ax.clear()

        # Draw environment boundary
        ax.plot([-H, H, H, -H, -H], [-H, -H, H, H, -H], lw=2)

        # Draw obstacle rectangle
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], lw=2)
            
        # Draw trajectory
        ax.plot(traj[:, 0], traj[:, 1], lw=1, alpha=0.9)
        ax.scatter([traj[0, 0]], [traj[0, 1]], label="start", zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], label="end", zorder=3)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-H, H)
        ax.set_ylim(-H, H)
        ax.grid(True)
        ax.legend()
        ax.set_title(f"Trajectory (dt={dt}, steps={steps})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
            
        self.canvas.draw()
        
    def on_clear(self):
        self.canvas.ax.clear()
        self.canvas.draw()
            
def main():
    app = QApplication(sys.argv)
    w = ParticleGui()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()  
        