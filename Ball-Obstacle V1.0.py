import numpy as np
import matplotlib.pyplot as plt

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


def simulate():
    # Create world and obstacle
    obs = RectObstacle(x0=-1.0, x1=1.0, y0=-0.5, y1=0.5)
    world = World(half_size=5.0, obstacle=obs)

    # Initialize a single ball
    ball = Ball(x=-4.0, y=-3.0, vx=2.2, vy=1.7, r=0.2)

    dt = 0.01
    steps = 4000

    # Store trajectory
    traj = np.zeros((steps, 2))
    for n in range(steps):
        world.step(ball, dt)
        traj[n] = [ball.x, ball.y]

    # Plot result
    fig, ax = plt.subplots(figsize=(6, 6))
    world.draw_bounds(ax)
    obs.draw(ax)
    ax.plot(traj[:, 0], traj[:, 1], lw=1)
    ax.scatter([traj[0, 0]], [traj[0, 1]], label="start")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-world.h, world.h)
    ax.set_ylim(-world.h, world.h)
    ax.grid(True)
    ax.legend()
    ax.set_title("Single ball + single obstacle (deterministic)")
    plt.show()


if __name__ == "__main__":
    simulate()
    


