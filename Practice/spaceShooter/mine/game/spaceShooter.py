from __future__ import division

import pygame
import random
import global_res
import os
from os import path
import numpy as np


class GameState:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        os.environ['SDL_VIDEO_CENTERED'] = '1'  # 居中显示
        # 设置游戏界面大小、背景图片及标题
        # 游戏界面像素大小
        self.screen = pygame.display.set_mode(
            (global_res.SCREEN_WIDTH, global_res.SCREEN_HEIGHT))
        # 加载图片
        self.__initImg()
        # 储存陨石,管理多个对象
        self.mobs = pygame.sprite.Group()
        # group for bullets
        self.powerups = pygame.sprite.Group()
        # 爆炸特效
        self.explosion = pygame.sprite.Group()
        # 初始化分数
        self.score = 0
        # 初始化玩家
        self.player = Player()
        # 初始化射击及陨石频率
        self.shoot_frequency = 0
        self.mob_frequency = 0
        self.exp_frequency = 0
        self.power_frequency = 0

    def __initImg(self):
        # 初始化背景图片
        # global_res.BACKGROUND_IMAGE = pygame.image.load(
        #     path.join(global_res.IMG_DIR, 'starfield.png')).convert()
        # global_res.BACKGROUND_RECT = global_res.BACKGROUND_IMAGE.get_rect()
        # 初始化玩家图片
        global_res.PLAYER_IMAGE = pygame.image.load(
            path.join(global_res.IMG_DIR, 'playerShip1_orange.png')).convert()

        # 导弹图片
        global_res.MISSILE_IMAGE = pygame.image.load(
            path.join(global_res.IMG_DIR, 'missile.png')).convert_alpha()

        # 子弹图片
        global_res.BULLET_IMAGE = pygame.image.load(
            path.join(global_res.IMG_DIR, 'laserRed16.png')).convert()

        # 障碍物
        meteor_list = [
            'meteorBrown_big1.png',
            'meteorBrown_big2.png',
            'meteorBrown_med1.png',
            'meteorBrown_med3.png',
            'meteorBrown_small1.png',
            'meteorBrown_small2.png',
            'meteorBrown_tiny1.png'
        ]

        for i in range(len(meteor_list)):
            global_res.METEOR_IMAGE.append(pygame.image.load(
                path.join(global_res.IMG_DIR, meteor_list[i])).convert())

        # 初始化爆炸图片
        global_res.EXPLOSION_ANIM['big'] = []
        global_res.EXPLOSION_ANIM['small'] = []
        global_res.EXPLOSION_ANIM['player'] = []
        for i in range(9):
            # 陨石爆炸
            filename = 'regularExplosion0{}.png'.format(i)
            img = pygame.image.load(
                path.join(global_res.IMG_DIR, filename)).convert()
            img.set_colorkey(global_res.BLACK)
            # 调整大小。大爆炸
            img_lg = pygame.transform.scale(img, (75, 75))
            global_res.EXPLOSION_ANIM['big'].append(img_lg)
            # 小爆炸
            img_sm = pygame.transform.scale(img, (32, 32))
            global_res.EXPLOSION_ANIM['small'].append(img_sm)

            # 玩家爆炸
            filename = 'sonicExplosion0{}.png'.format(i)
            img = pygame.image.load(
                path.join(global_res.IMG_DIR, filename)).convert()
            img.set_colorkey(global_res.BLACK)
            global_res.EXPLOSION_ANIM['player'].append(img)

        # 初始化道具图片
        global_res.POWERUP_IMAGE['shield'] = pygame.image.load(
            path.join(global_res.IMG_DIR, 'shield_gold.png')).convert()
        global_res.POWERUP_IMAGE['gun'] = pygame.image.load(
            path.join(global_res.IMG_DIR, 'bolt_gold.png')).convert()

    # 逐帧绘制，返回每一帧的图像数据
    def draw_shield_bar(self, surf, x, y, pct):
        pct = max(pct, 0)
        fill = (pct / 100) * global_res.BAR_LENGTH
        outline_rect = pygame.Rect(
            x, y, global_res.BAR_LENGTH, global_res.BAR_HEIGHT)
        fill_rect = pygame.Rect(x, y, fill, global_res.BAR_HEIGHT)
        pygame.draw.rect(surf, global_res.GREEN, fill_rect)
        pygame.draw.rect(surf, global_res.WHITE, outline_rect, 2)

    def render(self):
        # 绘制
        self.screen.fill(global_res.BLACK)
        # 显示子弹
        self.player.bullets.draw(self.screen)
        # 显示陨石
        self.mobs.draw(self.screen)
        # 显示玩家
        if not self.player.ishit:
            self.screen.blit(self.player.image, self.player.rect)

        self.explosion.draw(self.screen)

        self.powerups.draw(self.screen)

        self.draw_shield_bar(self.screen, 5, 5, self.player.shield)

    def frame_step(self, input_actions):
        terminal = False
        reward = 0.1
        # 控制游戏最大帧率为 30
        # 生成子弹，需要控制发射频率
        # 首先判断玩家飞机没有被击中

        # 检测输入正确性
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        if input_actions[1] == 1:
            self.player.moveLeft()
        elif input_actions[2] == 1:
            self.player.moveRight()
        elif input_actions[3] == 1:
            self.player.moveDown()
        elif input_actions[4] == 1:
            self.player.moveUp()

        # 更新部分：
        # 更新爆炸特效
        for exp in self.explosion:
            exp.update()

        # 更新飞机状态
        self.player.update()
        # 子弹移动
        for bullet in self.player.bullets:
            # 以固定速度移动子弹
            bullet.move()

        # 道具移动
        for power in self.powerups:
            power.move()

        # 陨石移动
        for mob in self.mobs:
            mob.move()

        # 固定生成部分：
        # 如果没被击中就射击
        if not self.player.ishit:
            # 循环30次射击一次
            if self.shoot_frequency % global_res.SHOOT_EVERY_F == 0:
                self.player.shoot()
            self.shoot_frequency += 1
            if self.shoot_frequency >= global_res.SHOOT_EVERY_F:
                self.shoot_frequency = 0

        # # 隔一定时间生成一个道具
        # if self.power_frequency % 60 == 0:
        #     powerup = Pow()
        #     self.powerups.add(powerup)
        #     self.power_frequency = 0
        # self.power_frequency += 1

        # 生成陨石,一次生成多个
        if self.mob_frequency % 30 == 0:
            for i in range(global_res.MOB_NUMS):
                mob = Mob()
                self.mobs.add(mob)
            self.mob_frequency = 0
        self.mob_frequency += 1

        # 判断部分
        # 判断陨石是否和飞机相撞
        hits = pygame.sprite.spritecollide(self.player, self.mobs, True,
                                           pygame.sprite.collide_circle)  # gives back a list, True makes the mob element disappear
        for hit in hits:
            self.player.shield -= hit.radius * 2
            expl = Explosion(hit.rect.center, 'small')
            self.explosion.add(expl)
            self.mobs.remove(hit)

            # 处理坠毁情况
            if self.player.shield <= 0:
                expl = Explosion(hit.rect.center, 'player')
                self.explosion.add(expl)
                # player_die_sound.play()
                self.player.shield = 100
                self.player.ishit = True
                reward = -1  # 撞毁后退出
                terminal = True
                self.__init__()

        # 判断子弹是否击中陨石
        hits = pygame.sprite.groupcollide(
            self.mobs, self.player.bullets, True, True)
        for hit in hits:
            expl = Explosion(hit.rect.center, 'big')
            self.explosion.add(expl)
            self.mobs.remove(hit)
            # give different scores for hitting big and small metoers
            self.score += 50 - hit.radius
            # random.choice(expl_sounds).play()
            if random.random() <= global_res.POW_RATE:
                power = Pow(hit.rect.center)
                self.powerups.add(power)

            reward = 1

        # if the player hit a power up
        hits = pygame.sprite.spritecollide(self.player, self.powerups, True)
        for hit in hits:
            if hit.type == 'shield':
                self.player.shield += random.randrange(10, 30)
                if self.player.shield >= 100:
                    self.player.shield = 100
            if hit.type == 'gun':
                self.player.powerup()

        self.render()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        clock = pygame.time.Clock()
        clock.tick(30)  # 限制最大帧率为30，防止用掉所有cpu资源
        return image_data, reward, terminal


class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # scale the player img down
        self.image = pygame.transform.scale(global_res.PLAYER_IMAGE, (50, 38))
        self.image.set_colorkey(global_res.BLACK)
        self.rect = self.image.get_rect()
        self.radius = 20
        self.rect.centerx = global_res.SCREEN_WIDTH / 2
        self.rect.bottom = global_res.SCREEN_HEIGHT - 10
        self.speed = 5
        self.shield = 100
        self.hidden = False
        self.hide_timer = pygame.time.get_ticks()
        self.power = 1
        self.power_timer = pygame.time.get_ticks()
        self.ishit = False
        self.bullets = pygame.sprite.Group()

    def shoot(self):
        if self.power == 1:
            bullet = Bullet(self.rect.centerx, self.rect.top)
            self.bullets.add(bullet)
            # shooting_sound.play()
        if self.power == 2:
            bullet1 = Bullet(self.rect.left, self.rect.centery)
            bullet2 = Bullet(self.rect.right, self.rect.centery)
            self.bullets.add(bullet1)
            self.bullets.add(bullet2)
            # shooting_sound.play()

        """ MOAR POWAH """
        if self.power >= 3:
            bullet1 = Bullet(self.rect.left, self.rect.centery)
            bullet2 = Bullet(self.rect.right, self.rect.centery)
            # Missile shoots from center of ship
            missile1 = Missile(self.rect.centerx, self.rect.top)
            self.bullets.add(bullet1)
            self.bullets.add(bullet2)
            self.bullets.add(missile1)
            # shooting_sound.play()
            # missile_sound.play()

    def powerup(self):
        self.power += 1
        self.power_time = pygame.time.get_ticks()

    def update(self):
        # time out for powerups
        if self.power >= 2 and pygame.time.get_ticks() - self.power_time > global_res.POWERUP_TIME:
            self.power -= 1
            self.power_time = pygame.time.get_ticks()

    # 向上移动，需要判断边界
    def moveUp(self):
        if self.rect.top <= 0:
            self.rect.top = 0
        else:
            self.rect.top -= self.speed

    # 向下移动，需要判断边界
    def moveDown(self):
        if self.rect.top >= global_res.SCREEN_HEIGHT - self.rect.height:
            self.rect.top = global_res.SCREEN_HEIGHT - self.rect.height
        else:
            self.rect.top += self.speed

    # 向左移动，需要判断边界
    def moveLeft(self):
        if self.rect.left <= 0:
            self.rect.left = 0
        else:
            self.rect.left -= self.speed

    # 向右移动，需要判断边界
    def moveRight(self):
        if self.rect.left >= global_res.SCREEN_WIDTH - self.rect.width:
            self.rect.left = global_res.SCREEN_WIDTH - self.rect.width
        else:
            self.rect.left += self.speed


# defines the sprite for Powerups
# defines the enemies
class Mob(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image_orig = random.choice(global_res.METEOR_IMAGE)
        self.image_orig.set_colorkey(global_res.BLACK)
        self.image = self.image_orig.copy()
        self.rect = self.image.get_rect()
        self.radius = int(self.rect.width * .90 / 2)
        self.rect.x = random.randrange(
            0, global_res.SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(-150, -100)
        # for randomizing the speed of the Mob
        self.speedy = random.randrange(5, 20)

        # randomize the movements a little more
        self.speedx = random.randrange(-3, 3)

        # adding rotation to the mob element
        self.rotation = 0
        self.rotation_speed = random.randrange(-8, 8)
        # time when the rotation has to happen
        self.last_update = pygame.time.get_ticks()

    def rotate(self):
        time_now = pygame.time.get_ticks()
        if time_now - self.last_update > 50:  # in milliseconds
            self.last_update = time_now
            self.rotation = (self.rotation + self.rotation_speed) % 360
            new_image = pygame.transform.rotate(self.image_orig, self.rotation)
            old_center = self.rect.center
            self.image = new_image
            self.rect = self.image.get_rect()
            self.rect.center = old_center

    def move(self):
        self.rotate()
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        # now what if the mob element goes out of the screen

        if (self.rect.top > global_res.SCREEN_HEIGHT + 10) or (self.rect.left < -25) or (self.rect.right > global_res.SCREEN_WIDTH + 20):
            self.rect.x = random.randrange(
                0, global_res.SCREEN_WIDTH - self.rect.width)
            self.rect.y = random.randrange(-100, -40)
            # for randomizing the speed of the Mob
            self.speedy = random.randrange(1, 8)


class Pow(pygame.sprite.Sprite):
    def __init__(self, center):
        pygame.sprite.Sprite.__init__(self)
        self.type = random.choice(['shield', 'gun'])
        self.image = global_res.POWERUP_IMAGE[self.type]
        self.image.set_colorkey(global_res.BLACK)
        self.rect = self.image.get_rect()
        # place the bullet according to the current position of the player
        self.rect.center = center
        self.speedy = 2

    def move(self):
        """should spawn right in front of the player"""
        self.rect.y += self.speedy
        # kill the sprite after it moves over the top border
        if self.rect.top > global_res.SCREEN_HEIGHT:
            self.kill()


# defines the sprite for bullets
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = global_res.BULLET_IMAGE
        self.image.set_colorkey(global_res.BLACK)
        self.rect = self.image.get_rect()
        # place the bullet according to the current position of the player
        self.rect.bottom = y
        self.rect.centerx = x
        self.speedy = -10

    def move(self):
        """should spawn right in front of the player"""
        self.rect.y += self.speedy
        # kill the sprite after it moves over the top border
        if self.rect.bottom < 0:
            self.kill()

        # now we need a way to shoot
        # lets bind it to "spacebar".
        # adding an event for it in Game loop


# FIRE ZE MISSILES
class Missile(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = global_res.MISSILE_IMAGE
        self.image.set_colorkey(global_res.BLACK)
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speedy = -10

    def move(self):
        """should spawn right in front of the player"""
        self.rect.y += self.speedy
        if self.rect.bottom < 0:
            self.kill()


class Explosion(pygame.sprite.Sprite):
    def __init__(self, center, size):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.image = global_res.EXPLOSION_ANIM[self.size][0]
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 75

    def update(self):
        if self.frame >= len(global_res.EXPLOSION_ANIM[self.size]):
            self.frame = 0
            self.kill()

        else:
            center = self.rect.center
            self.image = global_res.EXPLOSION_ANIM[self.size][self.frame]
            self.rect = self.image.get_rect()
            self.rect.center = center
            self.frame += 1


if __name__ == "__main__":
    G = GameState()
    while True:
        input_actions = [1, 0, 0]
        G.frame_step(input_actions)
