import pygame
import pytest

def test_hello():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("purple")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    assert 1 == 1

@pytest.mark.current
def test_move_circle():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill("purple")

        pygame.draw.circle(screen, "orange", player_pos, 30)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            player_pos.x -= 300 * dt
        if keys[pygame.K_RIGHT]:
            player_pos.x += 300 * dt
        if keys[pygame.K_UP]:
            player_pos.y -= 300 * dt
        if keys[pygame.K_DOWN]:
            player_pos.y += 300 * dt

        pygame.display.flip()

        dt = clock.tick(60) / 1000

    pygame.quit()