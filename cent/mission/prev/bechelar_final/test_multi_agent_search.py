import pytest 

import numpy as np 
import taichi as ti
import os 

@pytest.mark.app
def test_multi_agent_search():
    ti.init(arch=ti.gpu)
    STOP = False 
    AGENT_N = 10
    OBSTACLE_N = 10
    INTEREST_N = 10

    GRID_SIZE = 512
    ORIGIN = ti.Vector([GRID_SIZE / 2, GRID_SIZE / 2])

    SEARCH_R = 200
    INTEREST_R = 100
    SEARCH_AREA = ti.Vector([GRID_SIZE/2-SEARCH_R, GRID_SIZE/2-SEARCH_R, GRID_SIZE/2+SEARCH_R, GRID_SIZE/2+SEARCH_R])

    INIT_R = 20

    SENSE_ANGLE = 0.40 * np.pi
    SENSE_DIST = 8.0
    EVAPORATION = 0.95
    MOVE_ANGLE = 0.2 * np.pi
    MOVE_STEP = 1.0

    MainCogMap = ti.field(dtype=ti.f32, shape=[2, GRID_SIZE, GRID_SIZE])
    TrailGrid = ti.field(dtype=ti.f32, shape=[GRID_SIZE, GRID_SIZE])
    TruthGrid = ti.field(dtype=ti.i32, shape=[GRID_SIZE, GRID_SIZE])
    CoverGrid = ti.field(dtype=ti.f32, shape=[GRID_SIZE, GRID_SIZE])
    # Visiualization
    TruthGridDisp = ti.Vector.field(3, dtype=ti.f32, shape=[GRID_SIZE, GRID_SIZE])
    TrailGridDisp = ti.Vector.field(3, dtype=ti.f32, shape=[GRID_SIZE, GRID_SIZE])

    AgentPosition = ti.Vector.field(2, dtype=ti.f32, shape=[AGENT_N])
    AgentState = ti.field(dtype=ti.i32, shape=[AGENT_N])
    AgentHeading = ti.field(dtype=ti.f32, shape=[AGENT_N])
    ObstaclePos = ti.Vector.field(2, dtype=ti.f32, shape=[OBSTACLE_N])
    ObstacleState = ti.field(dtype=ti.i32, shape=[OBSTACLE_N])
    InterestPos = ti.Vector.field(2, dtype=ti.f32, shape=[INTEREST_N])
    InterestState = ti.field(dtype=ti.i32, shape=[INTEREST_N])

    @ti.func
    def inArea(pos, area):
        return pos[0] > area[0] and pos[0] < area[2] and pos[1] > area[1] and pos[1] < area[3]

    @ti.kernel
    def init():
        # initialize
        for p in ti.grouped(MainCogMap):
            MainCogMap[p] = 0.0
    
        for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
            TruthGrid[i, j] = 0
            TrailGrid[i, j] = 0.0
            CoverGrid[i, j] = 0.0
    
        # initalize the obstacles
        for i in ObstaclePos:
            ObstaclePos[i] = ti.Vector([ti.random(), ti.random()]) * GRID_SIZE
            ObstacleState[i] = 0
            ipos = ObstaclePos[i].cast(int)
            TruthGrid[ipos] = 1

        # intialize interesting position
        for i in InterestPos:
            InterestPos[i] = ti.Vector([2 * ti.random() - 1, 2 * ti.random() - 1]) * INTEREST_R + ORIGIN
            InterestState[i] = 0
            ipos = InterestPos[i].cast(int)
            TruthGrid[ipos] = 2
    
        # initialize agents 
        for i in AgentPosition:
            AgentPosition[i] = ti.Vector([2 * ti.random() - 1, 2 * ti.random() - 1]) * INIT_R + ORIGIN
            AgentHeading[i] = ti.random() * np.pi * 2.0
            AgentState[i] = 1
            ipos = AgentPosition[i].cast(int)
            TruthGrid[ipos] = 3

    @ti.func
    def sense(phase, pos, ang):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * SENSE_DIST
        res = MainCogMap[phase, p.cast(int) % GRID_SIZE]
        return res

    @ti.func
    def dis(pos1, pos2):
        return max(abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1]))

    @ti.kernel
    def step(phase: ti.i32):
        # sense and move
        for i in AgentPosition:
            pos, ang = AgentPosition[i], AgentHeading[i]
            l = sense(phase, pos, ang - SENSE_ANGLE)
            c = sense(phase, pos, ang)
            r = sense(phase, pos, ang + SENSE_ANGLE)
            b = sense(phase, pos, -ang)
            if b < c:
                ang = - ang
            elif l < c < r :
                ang -= MOVE_ANGLE
            elif l > c > r :
                ang += MOVE_ANGLE
            elif c < l and c < r:
                ang += MOVE_ANGLE * (2 * (ti.random() < 0.5) - 1)
            # update the truth

            TruthGrid[pos.cast(int)] = 0
            pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * MOVE_STEP
            AgentPosition[i], AgentHeading[i] = pos, ang
            ipos = pos.cast(int) % GRID_SIZE
            TruthGrid[ipos] = 3

        # deposit
        for i in AgentPosition:
            ipos = AgentPosition[i].cast(int) % GRID_SIZE
            TrailGrid[ipos] += 1.0
            MainCogMap[phase, ipos] += 1.0
            for di in ti.static(range(-4, 5)):
                for dj in ti.static(range(-4, 5)):
                    CoverGrid[ipos[0] + di, ipos[1] + dj] += 1.0
        # INTEREST  
        for i in InterestPos:
            ipos = InterestPos[i].cast(int)
            for j in ti.static(range(AGENT_N)):
                apos = AgentPosition[j].cast(int)
                if dis(apos, ipos) < 10:
                    InterestState[i] += 1
        
                if InterestState[i] > 0 and InterestState[i] < 3:
                    MainCogMap[phase, ipos] -= 100.0
                    TruthGrid[ipos] = 2
        # OBSTACLES
        for i in ObstaclePos:
            ipos = ObstaclePos[i].cast(int)
            for j in ti.static(range(AGENT_N)):
                apos = AgentPosition[j].cast(int)
                if dis(apos, ipos) < 10:
                    ObstacleState[i] = 1
                if ObstacleState[i] == 1:
                    MainCogMap[phase, ipos] += 10.0
                    TruthGrid[ipos] = 1

        # diffuse
        for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
            a = 0.0
            # limit 
            if not inArea([i, j], SEARCH_AREA):
                a = 1.0
                for di in ti.static(range(-2, 3)):
                    for dj in ti.static(range(-2, 3)):
                        a += MainCogMap[phase, (i + di) % GRID_SIZE, (j + dj) % GRID_SIZE]
            
            a *= EVAPORATION / 25.0
            MainCogMap[1 - phase, i, j] = a 
            


    @ti.kernel
    def visualize():
        for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
            if TruthGrid[i, j] == 0:
                TruthGridDisp[i, j] = ti.Vector([0.0, 0.0, 0.0])
            elif TruthGrid[i, j] == 1:
                for di in ti.static(range(-4, 5)):
                    for dj in ti.static(range(-4, 5)):
                        TruthGridDisp[i+di, j+dj] = ti.Vector([0.9, 0.1, 0.1])
            elif TruthGrid[i, j] == 2:
                for di in ti.static(range(-4, 5)):
                    for dj in ti.static(range(-4, 5)):
                        TruthGridDisp[i+di, j+dj] = ti.Vector([0.1, 0.3, 0.8])  
            elif TruthGrid[i, j] >= 3:
                TruthGridDisp[i, j] = ti.Vector([0.3, 0.9, 0.5])



    print("[Hint] Press A/Z to change the simulation speed.")
    gui = ti.GUI('MAIN')
    init()

    i = 0
    step_per_frame = gui.slider('step_per_frame', 1, 100, 1)
    folder = "output/int10_obs10_agt10"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    while gui.running and not gui.get_event(gui.ESCAPE):
        # check if stop
        for _ in range(int(step_per_frame.value)):
            if not STOP:
                step(i % 2)
                # Desp
                visualize()
                # SAVE
                if (i % 100 == 0):
                    filename = folder + "/cog_{}.png".format(i)
                    ti.imwrite(MainCogMap.to_numpy()[0], filename)
                    filename = folder + "/cover_{}.png".format(i)
                    ti.imwrite(CoverGrid.to_numpy(), filename)
                    filename = folder + "/trail_{}.png".format(i)
                    ti.imwrite(TrailGrid.to_numpy(), filename)
                    filename = folder + "./truth_{}.png".format(i)
                    ti.imwrite(TruthGridDisp, filename)
            else:
                print("FINISHED when {} step found {} with {} agents".format(i, INTEREST_N, AGENT_N))
                gui.EXIT()

            if (i > 2000):
                print("Mission Fails")
                STOP = True
                gui.EXIT()
            STOP = True
            for state in InterestState.to_numpy():
                if state < 3:
                    STOP = False

            i += 1

        gui.set_image(TruthGridDisp)
        gui.show()