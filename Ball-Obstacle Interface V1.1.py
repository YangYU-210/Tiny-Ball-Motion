from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,QLabel, QLineEdit, QPushButton, QMessageBox, QFormLayout
import matplotlib.pyplot as plt
import numpy as np
import sys 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle



class Ball:
    def __init__(self, x, y, vx, vy, mass = 1, r=0.2):
        # Position of the ball
        self.x = float(x)
        self.y = float(y)
        # Velocity of the ball
        self.vx = float(vx)
        self.vy = float(vy)
        # Radius of the ball
        self.r = float(r)
        # for collisions ball-ball
        self.mass = float(mass) 

    @property
    def position(self):
         # vectorial access to position
        return np.array([self.x, self.y])
    
    @property
    def velocity(self):
       # vectorial access to velocity
        return np.array([self.vx, self.vy])
    
    @property
    def kinetic_energy(self):
        # kinetic energy
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)

    def advance(self, dt):
        
       # Advance the ball position by one time step
        #using simple Newtonian motion:
       #     x <- x + v * dt
        
        self.x += self.vx * dt
        self.y += self.vy * dt


class RectObstacle:
    
    #Axis-aligned rectangular obstacle:
    #region [x0, x1] × [y0, y1]
    
    def __init__(self, x0, x1, y0, y1):
        # Ensure correct ordering of bounds
        self.x0, self.x1 = (min(x0, x1), max(x0, x1))
        self.y0, self.y1 = (min(y0, y1), max(y0, y1))

    def draw(self, ax):
        #Draw the rectangle boundary
        xs = [self.x0, self.x1, self.x1, self.x0, self.x0]
        ys = [self.y0, self.y0, self.y1, self.y1, self.y0]
        ax.plot(xs, ys, lw=2)

    def contains_expanded(self, x, y, margin):
        return (self.x0 - margin <= x <= self.x1 + margin) and (self.y0 - margin <= y <= self.y1 + margin)
    
    def collide(self, ball: Ball):
        
        #Detect and resolve collision between the ball and the rectangle.

       # Method:
       # - Expand the rectangle by the ball radius (AABB method)
       # - If the ball center enters this expanded region, a collision occurred
       # - Reflect velocity along the direction of minimum penetration
        

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
    
    #Simulation domain with square boundaries
    #and (optionally) a single rectangular obstacle
    
    def __init__(self, half_size=5.0, obstacle=None):
        # Half-size of the square domain [-h, h] × [-h, h]
        self.h = float(half_size)
        # Store obstacles as a list for iteration
        if obstacle is None:
            self.obstacles = []
        elif isinstance(obstacle, list):
            self.obstacles = obstacle
        else:
            self.obstacles = [obstacle]
        self.balls = []  # for multi-particles

    def add_ball(self, ball: Ball):
        # add a ball in the universe
        self.balls.append(ball)

    def check_ball_collision(self, b1: Ball, b2: Ball):
        # collision between 2 balls
        dx = b2.x - b1.x
        dy = b2.y - b1.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < b1.r + b2.r:
            # Collision détectée,  résolution élastique
            nx, ny = dx/dist, dy/dist  # Normale
            
            # Vitesses relatives
            dvx = b1.vx - b2.vx
            dvy = b1.vy - b2.vy
            dvn = dvx*nx + dvy*ny
            
            # Conservation de la quantité de mouvement
            m1, m2 = b1.mass, b2.mass
            b1.vx -= (2*m2/(m1+m2)) * dvn * nx
            b1.vy -= (2*m2/(m1+m2)) * dvn * ny
            b2.vx += (2*m1/(m1+m2)) * dvn * nx
            b2.vy += (2*m1/(m1+m2)) * dvn * ny
            
            # Séparation pour éviter l'interpénétration
            overlap = (b1.r + b2.r - dist) / 2
            b1.x -= overlap * nx
            b1.y -= overlap * ny
            b2.x += overlap * nx
            b2.y += overlap * ny

    def draw_bounds(self, ax):
        #Draw square simulation boundary
        h = self.h
        ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], lw=2)

    def bounce_on_walls(self, ball: Ball):
    
       # Reflect the ball on the domain boundaries
       # by reversing the corresponding velocity component
        
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

    def step(self, dt):
        
        # Mouvement
        for ball in self.balls:
            ball.advance(dt)
            self.bounce_on_walls(ball)
        
        # Collisions avec obstacles
        for ball in self.balls:
            for obstacle in self.obstacles:
                obstacle.collide(ball)
        
        # Collisions inter-balles
        for i, b1 in enumerate(self.balls):
            for b2 in self.balls[i+1:]:
                self.check_ball_collision(b1, b2)
                
                
def generate_obstacles(rng, n_obs, obs_size, half_size, margin=0.2, max_tries=10_000):
    """
    Randomly place square obstacles of size obs_size x obs_size
    inside the domain [-h,h] without going out of bounds.
    """
    h = half_size
    s = obs_size
    obstacles = []

    # allowed center region so that obstacle stays inside
    x_min = -h + margin + s / 2
    x_max =  h - margin - s / 2
    y_min = -h + margin + s / 2
    y_max =  h - margin - s / 2
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Environment too small for the given obstacle size.")

    for _ in range(n_obs):
        # simple random placement
        cx = rng.uniform(x_min, x_max)
        cy = rng.uniform(y_min, y_max)
        obstacles.append(RectObstacle(cx - s/2, cx + s/2, cy - s/2, cy + s/2))

    return obstacles


def generate_balls(rng, n_balls, r_ball, half_size, obstacles, v0=2.0, max_tries=50_000):
    """
    Place balls randomly (avoiding obstacles) and assign a simple initial velocity.
    """
    h = half_size
    balls = []

    tries = 0
    while len(balls) < n_balls:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Failed to place balls without overlapping obstacles. Try fewer balls or smaller radius.")

        x = rng.uniform(-h + r_ball, h - r_ball)
        y = rng.uniform(-h + r_ball, h - r_ball)

        # avoid starting inside any obstacle (expanded by ball radius)
        ok = True
        for obs in obstacles:
            if obs.contains_expanded(x, y, r_ball):
                ok = False
                break
        if not ok:
            continue

        # simple initial velocity: random direction with magnitude v0
        theta = rng.uniform(0, 2*np.pi)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        balls.append(Ball(x, y, vx, vy, r=r_ball))

    return balls

def simulate(world: World, dt: float, steps: int):
    """
    Run simulation and return trajectories array: (n_balls, steps, 2)
    """
    n = len(world.balls)
    traj = np.zeros((n, steps, 2), dtype=float)
    # initial positions
    for i, b in enumerate(world.balls):
        traj[i, 0] = [b.x, b.y]

    for t in range(steps - 1):
        world.step(dt)
        for i, b in enumerate(world.balls):
            traj[i, t + 1] = [b.x, b.y]

    return traj


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
        #self.edit_obs_h = QLineEdit("1.0")      # obstacle height
        self.edit_dt = QLineEdit("0.01")        # time step
        self.edit_step = QLineEdit("1000")      # total number of steps
        self.nb_balls = QLineEdit("3")          # total number of balls
        self.nb_obs = QLineEdit("1")            # total number of obstacles
        
        
        
        form = QFormLayout()
        form.addRow(QLabel("Environment half-size (H):"), self.edit_env_half)
        form.addRow(QLabel("Ball radius (r):"), self.edit_ball_r)
        form.addRow(QLabel("Obstacle width (w):"), self.edit_obs_w)
        #form.addRow(QLabel("Obstacle height (h):"), self.edit_obs_h)
        form.addRow(QLabel("Time step (dt):"), self.edit_dt)
        form.addRow(QLabel("Total steps:"), self.edit_step)
        form.addRow(QLabel("Total number of balls:"), self.nb_balls)
        form.addRow(QLabel("Total number of obstacles:"), self.nb_obs)
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
        #h_obs = float(self.edit_obs_h.text())
        dt = float(self.edit_dt.text())
        steps = int(float(self.edit_step.text()))
        nb_b = int(float(self.nb_balls.text()))
        nb_o = int(float(self.nb_obs.text()))
        
            
        # Basic validation
        if H <= 0 or r <= 0 or w <= 0:
            raise ValueError("All sizes must be positive.")
        if r >= H:
            raise ValueError("Ball radius must be smaller than environment half-size.")
        if w >= 2 * H:
            raise ValueError("Obstacle is too large for the environment.")
            
        # Initialize environment, obstacle, ball object
        rng = np.random.default_rng()
        
        obs = generate_obstacles(rng, nb_o, w, H)
        world = World(H, obs)
        ball = generate_balls(rng, nb_b, r, H, obs)
        for b in ball:
            world.add_ball(b)
            
        traj = simulate(world, dt, steps)
            
        # Plot
        ax = self.canvas.ax
        ax.clear()

        # Draw environment boundary
        world.draw_bounds(ax)
        for obs in world.obstacles:
            obs.draw(ax)


        # Draw obstacle rectangle
        #ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], lw=2)
            
        # plot trajectories
        for i in range(traj.shape[0]):
            # Plot trajectory line with a label for the legend
            (line,) = ax.plot(
                traj[i, :, 0],
                traj[i, :, 1],
                lw=1,
                alpha=0.8,
                label=f"Ball {i+1}",
            )
            color = line.get_color()

            # Draw "balls" with TRUE physical radius r (data units) using Circle patches
            start_circle = Circle(
                (traj[i, 0, 0], traj[i, 0, 1]),
                radius=r,
                edgecolor=color,
                facecolor=color,
                alpha=1,
                zorder=3,
            )
            end_circle = Circle(
                (traj[i, -1, 0], traj[i, -1, 1]),
                radius=r,
                edgecolor=color,
                facecolor=color,
                alpha=1,
                zorder=3,
            )
            ax.add_patch(start_circle)
            ax.add_patch(end_circle)

            # Add a small text notation along the trajectory
            mid_idx = traj.shape[1] // 2
            ax.annotate(
                f"{i+1}",
                (traj[i, mid_idx, 0], traj[i, mid_idx, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=color,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-H, H)
        ax.set_ylim(-H, H)
        ax.grid(True)
        ax.set_title(f"Trajectory (dt={dt}, steps={steps})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
            
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
        