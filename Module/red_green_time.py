import time

start_time = time.time()
CYCLE_TIME = 20

class redAndGreen:
    def red_and_green_light():
            elapsed = time.time() - start_time

            cycle = elapsed % (2 * CYCLE_TIME)

            if cycle < CYCLE_TIME:
                return False
            else:
                return True