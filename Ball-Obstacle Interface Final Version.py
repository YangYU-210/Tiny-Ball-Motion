from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QLineEdit, QPushButton, QFormLayout, QCheckBox)
import matplotlib.pyplot as plt
import numpy as np
import sys
from PyQt5.QtCore import QTimer
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



def generate_obstacles(rng, n_obs, obs_size, half_size, margin=0.2):
   # générer obstacles placés aléatoirement
    h = half_size
    s = obs_size
    obstacles = []

    x_min = -h + margin + s / 2
    x_max = h - margin - s / 2
    y_min = -h + margin + s / 2
    y_max = h - margin - s / 2

    if x_min >= x_max or y_min >= y_max:
        return []

    for _ in range(n_obs):
        cx = rng.uniform(x_min, x_max)
        cy = rng.uniform(y_min, y_max)
        obstacles.append(RectObstacle(cx - s/2, cx + s/2, cy - s/2, cy + s/2))

    return obstacles


def generate_balls(rng, n_balls, r_ball, half_size, obstacles, v0=2.0, max_tries=5000):
   # placer les balles en évitant les obstacles
    h = half_size
    balls = []

    tries = 0
    while len(balls) < n_balls:
        tries += 1
        if tries > max_tries:
            break

        x = rng.uniform(-h + r_ball, h - r_ball)
        y = rng.uniform(-h + r_ball, h - r_ball)

        ok = True
        for obs in obstacles:
            if (obs.x0 - r_ball <= x <= obs.x1 + r_ball) and (obs.y0 - r_ball <= y <= obs.y1 + r_ball):
                ok = False
                break
        if not ok:
            continue

        theta = rng.uniform(0, 2 * np.pi)
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        balls.append(Ball(x, y, vx, vy, mass=1.0, r=r_ball))

    return balls


class MplCanvas(FigureCanvas):
   # canvas matplotlib
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 6))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class ParticleGui(QWidget):
   # interface principale simulation

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ball + Obstacles Simulation")

        self.world = None
        self.running = False
        self.time = 0.0
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        # Timer pour l'animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)

        # Champs de saisie
        self.edit_env_half = QLineEdit("5.0")
        self.edit_ball_r = QLineEdit("0.3")
        self.edit_obs_w = QLineEdit("2.0")
        self.edit_dt = QLineEdit("0.02")
        self.nb_balls = QLineEdit("3")
        self.nb_obs = QLineEdit("1")

        # Formulaire avec QFormLayout (style de ton collègue)
        form = QFormLayout()
        form.addRow(QLabel("Environment half-size (H):"), self.edit_env_half)
        form.addRow(QLabel("Ball radius (r):"), self.edit_ball_r)
        form.addRow(QLabel("Obstacle width (w):"), self.edit_obs_w)
        form.addRow(QLabel("Time step (dt):"), self.edit_dt)
        form.addRow(QLabel("Number of balls:"), self.nb_balls)
        form.addRow(QLabel("Number of obstacles:"), self.nb_obs)

        # Checkbox pour trajectoires
        self.trails_checkbox = QCheckBox("Show trajectories")
        self.trails_checkbox.setChecked(True)
        form.addRow(self.trails_checkbox)

        # Boutons
        self.btn_start = QPushButton("Start")
        self.btn_reset = QPushButton("Reset")
        self.btn_clear = QPushButton("Clear")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_clear)

        # Labels d'info
        self.energy_label = QLabel("Total energy: 0.00")
        self.time_label = QLabel("Time: 0.00")
        self.balls_label = QLabel("Balls: 0")

        # Canvas
        self.canvas = MplCanvas(self)

        # Layout principal
        left = QVBoxLayout()
        left.addLayout(form)
        left.addLayout(btn_row)
        left.addWidget(self.energy_label)
        left.addWidget(self.time_label)
        left.addWidget(self.balls_label)
        left.addStretch(1)

        main = QHBoxLayout()
        main.addLayout(left, 0)
        main.addWidget(self.canvas, 1)
        self.setLayout(main)

        # Connexions
        self.btn_start.clicked.connect(self.toggle_simulation)
        self.btn_reset.clicked.connect(self.reset_simulation)
        self.btn_clear.clicked.connect(self.on_clear)

        # Initialiser la simulation
        self.reset_simulation()

    def reset_simulation(self):
       # réinitialisation
        self.running = False
        self.timer.stop()
        self.btn_start.setText("Start")
        self.time = 0.0

        try:
            H = float(self.edit_env_half.text())
            r = float(self.edit_ball_r.text())
            w = float(self.edit_obs_w.text())
            self.dt = float(self.edit_dt.text())
            nb_b = int(float(self.nb_balls.text()))
            nb_o = int(float(self.nb_obs.text()))
        except ValueError:
            return

        # ---- Clamp dt depending on obstacle size ----
        # We limit the distance travelled per step to a fraction of the obstacle width
        if w > 0:
            v0 = 2.0  # same default speed used in generate_balls
            safety_factor = 0.25  # ball moves at most 25% of obstacle width per step
            max_dt = safety_factor * w / v0
            # basic sanity on max_dt to avoid extremely small or huge values
            max_dt = max(1e-4, min(max_dt, 0.1))
            if self.dt <= 0 or self.dt > max_dt:
                self.dt = max_dt
                # reflect the clamped value in the UI
                self.edit_dt.setText(f"{self.dt:.4g}")

        self.world_size = H

        # Validation
        if H <= 0 or r <= 0 or w <= 0:
            return
        if r >= H:
            return

        # Générer obstacles et balles
        rng = np.random.default_rng()
        obstacles = generate_obstacles(rng, nb_o, w, H)
        self.world = World(half_size=H, obstacle=obstacles)

        balls = generate_balls(rng, nb_b, r, H, obstacles)
        for ball in balls:
            ball.trail = []
            self.world.add_ball(ball)

        self.draw_scene()
        self.update_info()

    def draw_scene(self):
       # dessiner la scene complete
        ax = self.canvas.ax
        ax.clear()
        H = self.world_size

        # Limites du monde
        self.world.draw_bounds(ax)

        # Obstacles
        for obs in self.world.obstacles:
            xs = [obs.x0, obs.x1, obs.x1, obs.x0, obs.x0]
            ys = [obs.y0, obs.y0, obs.y1, obs.y1, obs.y0]
            ax.fill(xs, ys, color='#34495e', alpha=0.8)
            ax.plot(xs, ys, color='#2c3e50', lw=2)

        # Trajectoires et balles
        for i, ball in enumerate(self.world.balls):
            color = self.colors[i % len(self.colors)]

            # Trajectoire
            if self.trails_checkbox.isChecked() and hasattr(ball, 'trail') and len(ball.trail) >= 2:
                trail = np.array(ball.trail)
                ax.plot(trail[:, 0], trail[:, 1], color=color, lw=1, alpha=0.6)

            # Balle (cercle avec le vrai rayon)
            circle = Circle(
                (ball.x, ball.y),
                radius=ball.r,
                edgecolor='black',
                facecolor=color,
                alpha=1,
                zorder=3,
                lw=1.5
            )
            ax.add_patch(circle)

            # Numéro de la balle
            ax.annotate(
                f"{i+1}",
                (ball.x, ball.y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=8,
                color='white',
                ha='center',
                va='center',
                fontweight='bold'
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-H * 1.05, H * 1.05)
        ax.set_ylim(-H * 1.05, H * 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Simulation (t = {self.time:.2f}s)")

        self.canvas.draw()

    def toggle_simulation(self):
       # lancer/arréter
        self.running = not self.running
        if self.running:
            self.btn_start.setText("Stop")
            self.timer.start(33)  # ~30 FPS
        else:
            self.btn_start.setText("Start")
            self.timer.stop()

    def animate(self):
       # boucle d'animation
        if not self.running or self.world is None:
            return

        self.world.step(self.dt)
        self.time += self.dt

        # Mise à jour des trajectoires
        for ball in self.world.balls:
            if hasattr(ball, 'trail'):
                ball.trail.append((ball.x, ball.y))
                if len(ball.trail) > 200:
                    ball.trail.pop(0)

        self.draw_scene()
        self.update_info()

    def update_info(self):
       # misa a jour
        if self.world:
            total_energy = sum(ball.kinetic_energy for ball in self.world.balls)
            self.energy_label.setText(f"Total energy: {total_energy:.2f}")
            self.time_label.setText(f"Time: {self.time:.2f}")
            self.balls_label.setText(f"Balls: {len(self.world.balls)}")

    def on_clear(self):
        # efface canva
        self.canvas.ax.clear()
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = ParticleGui()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

