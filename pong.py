import numpy as np
import cv2
import math
import keyboard
from utils import normalize_states
import matplotlib.pyplot as plt
from random import choice, randint, random


class Pong:
    PADDING = 10
    MAX_SCORE = 21
    WIDTH = 160
    HEIGHT = 192
    SPEEDUP = 1
    ACTIONS = ["UP", "DOWN", "NONE"]
    GRACE_PADDLE = 10
    BALL_MARKER_SIZE = 4

    @staticmethod
    def read_key(up, down):
        if keyboard.is_pressed(up):
            return 0
        elif keyboard.is_pressed(down):
            return 1
        else:
            return 2

    @staticmethod
    def random_action():
        return choice(Pong.ACTIONS)

    class Paddle:
        EDGE_BUFFER = 0
        HEIGHT = 20
        SPEED = 1
        WIDTH = 2

        def __init__(self, side):
            self.side = side
            self.x = 0
            self.y = int(Pong.HEIGHT / 2)
            self.w = self.WIDTH
            self.h = self.HEIGHT
            self.speed = self.SPEED * Pong.SPEEDUP
            if side == "left":
                self.x = Pong.PADDING
            elif side == "right":
                self.x = Pong.WIDTH - Pong.PADDING

        def reset(self):
            self.x = 0
            self.y = int(Pong.HEIGHT / 2)
            self.w = self.WIDTH
            self.h = self.HEIGHT
            self.speed = self.SPEED * Pong.SPEEDUP
            if self.side == "left":
                self.x = Pong.PADDING
            elif self.side == "right":
                self.x = Pong.WIDTH - Pong.PADDING

        def up(self):
            self.y -= self.speed
            if self.y < Pong.Paddle.EDGE_BUFFER:
                self.y = Pong.Paddle.EDGE_BUFFER

        def down(self):
            self.y += self.speed
            max = Pong.HEIGHT - Pong.Paddle.EDGE_BUFFER
            if self.y > max:
                self.y = max

        def handle_action(self, action):
            if action == "UP":
                self.up()
            elif action == "DOWN":
                self.down()
            elif action == "NONE":
                pass

    class Ball:  # 0, 30, -45, 225
        DIAMETER = 6
        SPEED = 1
        BOUNCE_ANGLES = [0, 60, 45, 30, -30, -45, -60]

        def spawn_hit_practice(self):
            self.x = 5
            self.y = randint(0, Pong.HEIGHT)
            self.speed = self.SPEED * Pong.SPEEDUP
            self.velocity = ((random() * 0.5 + 0.1), (random() * 2) - 1)
            self.w = self.DIAMETER
            self.right = True
            self.h = self.DIAMETER

        def __init__(self, hit_practice=False):
            self.hit_practice = hit_practice
            if self.hit_practice:
                self.spawn_hit_practice()
            else:
                self.x = math.floor(Pong.WIDTH / 2)
                self.y = math.floor(Pong.HEIGHT / 2)
                self.speed = self.SPEED * Pong.SPEEDUP
                self.velocity = (0, 0)
                self.w = self.DIAMETER
                self.right = None
                self.h = self.DIAMETER

        def reset(self):
            if self.hit_practice:
                self.spawn_hit_practice()
            else:
                self.right = None
                self.x = math.floor(Pong.WIDTH / 2)
                self.y = math.floor(Pong.HEIGHT / 2)
                self.speed = self.SPEED * Pong.SPEEDUP
                self.velocity = (0, 0)
                self.w = self.DIAMETER
                self.h = self.DIAMETER

        def get_vector(self, deg, scale):
            rad = math.pi * deg / 180
            return scale * math.cos(rad), scale * math.sin(rad)

        def reverse_vector(self, vector):
            x, y = vector
            return -x, -y

        def bounce(self, x=False, y=False):
            xv, yv = self.velocity
            if x:
                xv = -xv
            if y:
                yv = -yv
            self.velocity = xv, yv

        def bounce_angle(self, pos):
            segment = int(round(pos * 3))
            angle = Pong.Ball.BOUNCE_ANGLES[segment]
            velocity = self.get_vector(angle, self.speed)
            if self.right:
                velocity = self.get_vector(-angle, self.speed)
                velocity = self.reverse_vector(velocity)
            self.velocity = velocity
            self.speed += 0.1 * Pong.SPEEDUP

        def update(self):
            if self.velocity == (0, 0):
                angle = choice(Pong.Ball.BOUNCE_ANGLES)
                self.right = True
                if randint(0, 1) == 1:
                    angle += 180
                    self.right = False
                self.velocity = self.get_vector(angle, self.speed)
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            if self.y > Pong.HEIGHT:
                self.y = Pong.HEIGHT
                self.bounce(y=True)
            if self.y < 0:
                self.y = 0
                self.bounce(y=True)

    def __init__(self, hit_practice=False, marker_h=False, marker_v=False):
        # Holds last raw screen pixels for rendering
        self.last_screen = None
        self.hit_practice = hit_practice
        self.score_left = 0
        self.marker_v = marker_v
        self.marker_h = marker_h
        self.score_right = 0
        self.left = Pong.Paddle("left") if not self.hit_practice else None
        self.right = Pong.Paddle("right")
        self.ball = Pong.Ball(hit_practice=hit_practice)

    def reset(self):
        self.score_left = 0
        self.score_right = 0
        if not self.hit_practice: self.left.reset()
        self.right.reset()
        self.ball.reset()
        screen = self.render()
        self.last_screen = screen
        return screen

    def get_score(self):
        return self.score_left, self.score_right

    def get_bot_data(self, left=False, right=False):
        if left:
            return self.left, self.ball
        if right:
            return self.right, self.ball

    def check_collision(self, ball, paddle):
        ball_left = ball.x - (ball.w / 2)
        ball_right = ball.x + (ball.w / 2)
        ball_top = ball.y + (ball.h / 2)
        ball_bottom = ball.y - (ball.h / 2)
        paddle_left = paddle.x - (paddle.w / 2)
        paddle_right = paddle.x + (paddle.w / 2)
        paddle_top = paddle.y + (paddle.h / 2)
        paddle_bottom = paddle.y - (paddle.h / 2)
        left_collide = ball_left > paddle_left and ball_left < paddle_right
        right_collide = ball_right > paddle_left and ball_right < paddle_right
        top_collide = ball_top > paddle_bottom and ball_top < paddle_top
        bottom_collide = ball_bottom < paddle_top and ball_bottom > paddle_bottom
        if left_collide or right_collide:
            if top_collide or bottom_collide:
                return True, (ball.y - paddle.y) / (paddle.h / 2)
        return False, 0

    def step_hit_practice(self, right_action, frames=3):
        reward_l = 0
        reward_r = 0
        done = False
        for i in range(frames):
            if not done:
                self.right.handle_action(right_action)

                collide_right, pos = self.check_collision(self.ball, self.right)
                if collide_right and self.ball.right:
                    self.ball.bounce_angle(pos)
                    self.ball.right = False

                if self.ball.x < 0:
                    self.score_right += 1
                    reward_l -= 1.0
                    reward_r += 1.0
                    self.ball.reset()
                    self.right.reset()
                elif self.ball.x > Pong.WIDTH:
                    self.score_left += 1
                    reward_l += 1.0
                    reward_r -= 1.0
                    self.ball.reset()
                    self.right.reset()
                self.ball.update()
                done = False
                if self.score_right >= Pong.MAX_SCORE or self.score_left >= Pong.MAX_SCORE:
                    done = True
        screen = self.render()
        self.last_screen = screen
        return screen, (reward_l, reward_r), done

    def step(self, left_action, right_action, frames=3):
        if self.hit_practice:
            return self.step_hit_practice(right_action, frames=frames)
        reward_l = 0
        reward_r = 0
        done = False
        for i in range(frames):
            if not done:
                self.left.handle_action(left_action)
                self.right.handle_action(right_action)

                collide_left, pos = self.check_collision(self.ball, self.left)
                if collide_left and not self.ball.right:
                    self.ball.bounce_angle(pos)
                    self.ball.right = True
                collide_right, pos = self.check_collision(self.ball, self.right)
                if collide_right and self.ball.right:
                    self.ball.bounce_angle(pos)
                    self.ball.right = False

                if self.ball.x < 0:
                    self.score_right += 1
                    reward_l -= 1.0
                    reward_r += 1.0
                    self.ball.reset()
                    self.left.reset()
                    self.right.reset()
                elif self.ball.x > Pong.WIDTH:
                    self.score_left += 1
                    reward_l += 1.0
                    reward_r -= 1.0
                    self.ball.reset()
                    self.left.reset()
                    self.right.reset()
                self.ball.update()
                done = False
                if self.score_right >= Pong.MAX_SCORE or self.score_left >= Pong.MAX_SCORE:
                    done = True

        screen = self.render()
        self.last_screen = screen
        return screen, (reward_l, reward_r), done

    def show(self, scale=1,  duration=1):
        l, r = self.get_score()
        to_render = cv2.resize(self.last_screen, (int(Pong.WIDTH * scale), int(Pong.HEIGHT * scale)))
        cv2.imshow(f"Pong", cv2.cvtColor(to_render, cv2.COLOR_RGB2BGR))
        cv2.waitKey(duration)

    def get_screen(self):
        return self.last_screen

    def draw_rect(self, screen, x, y, w, h, color):
        screen[max(y,0):y+h+1, max(x,0):x+w+1] = color

    def render(self):
        screen = np.zeros((Pong.HEIGHT, Pong.WIDTH, 3), dtype=np.float32)
        screen[:, :] = (0, 60, 140)

        # Draw middle grid lines
        #self.draw_rect(screen, 0, int(Pong.HEIGHT/2), int(Pong.WIDTH), 1, 255)
        #self.draw_rect(screen, int(Pong.WIDTH/2), 0, 1, int(Pong.HEIGHT), 255)
        if not self.hit_practice:
            self.draw_rect(screen, int(self.left.x - int(self.left.w / 2)), int(self.left.y - int(self.left.h / 2)),
                           self.left.w, self.left.h, 255)
        self.draw_rect(screen, int(self.right.x - int(self.right.w / 2)), int(self.right.y - int(self.right.h / 2)),
                       self.right.w, self.right.h, 255)
        self.draw_rect(screen, int(self.ball.x - int(self.ball.w / 2)), int(self.ball.y - int(self.ball.h / 2)),
                       self.ball.w, self.ball.h, 255)
        # Draw pixel markers on top and left aligned with ball
        if self.marker_h:
            marker_x = max(min(int(self.ball.x), Pong.WIDTH-1), 0)
            self.draw_rect(screen, int(marker_x - int(Pong.BALL_MARKER_SIZE / 2)), 0,
                           Pong.BALL_MARKER_SIZE, Pong.BALL_MARKER_SIZE, 255)
        if self.marker_v:
            marker_y = min(max(int(self.ball.y), 0), Pong.HEIGHT - 1)
            self.draw_rect(screen, 0, int(marker_y - int(Pong.BALL_MARKER_SIZE / 2)),
                           Pong.BALL_MARKER_SIZE, Pong.BALL_MARKER_SIZE, 255)
        return screen


