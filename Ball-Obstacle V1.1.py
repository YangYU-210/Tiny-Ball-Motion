import numpy as np
import matplotlib.pyplot as plt

class Ball:
    def __init__(self, x, y, vx, vy, mass, r=0.2):
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


def simulate():
    # Create world and obstacle
    obs = RectObstacle(x0=-1.0, x1=1.0, y0=-0.5, y1=0.5)
    world = World(half_size=5.0, obstacle=obs)

    # Initialize 3 balls with different positions and velocities
    balls = [
        Ball(x=-4.0, y=-3.0, vx=2.2, vy=1.7, mass=1.0, r=0.2),
        Ball(x=3.0, y=2.0, vx=-1.5, vy=-2.0, mass=1.0, r=0.2),
        Ball(x=0.0, y=3.5, vx=1.0, vy=-1.2, mass=1.0, r=0.2),
    ]
    for ball in balls:
        world.add_ball(ball)

    dt = 0.01
    steps = 4000

    # Store trajectories for all balls
    trajs = [np.zeros((steps, 2)) for _ in balls]
    for n in range(steps):
        world.step(dt)
        for i, ball in enumerate(balls):
            trajs[i][n] = [ball.x, ball.y]

    # Plot result
    fig, ax = plt.subplots(figsize=(6, 6))
    world.draw_bounds(ax)
    obs.draw(ax)

    colors = ['blue', 'red', 'green']
    for i, traj in enumerate(trajs):
        ax.plot(traj[:, 0], traj[:, 1], lw=1, color=colors[i], alpha=0.7)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color=colors[i], marker='o', label=f"ball {i+1} start")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=colors[i], marker='x', s=100)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-world.h, world.h)
    ax.set_ylim(-world.h, world.h)
    ax.grid(True)
    ax.legend()
    ax.set_title("3 balls + obstacle (with collisions)")
    plt.show()


if __name__ == "__main__":
    simulate()