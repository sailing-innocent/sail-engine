import pytest 
from cent.app.scene.gym_env.jss.jss_env import JssEnv 
import numpy as np 
import imageio 
from pathlib import Path

@pytest.mark.app
def test_job_shop_env():
        print("")
        env_config = {
            "instance_path": Path(__file__).parent.absolute() / "instances" / "ta01"
        }
        env = JssEnv(env_config)
        env.reset()
        assert(env.current_time_step == 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [7, 11, 9, 10, 8, 3, 12, 2, 14, 5, 1, 6, 4, 0, 13],
            [2, 8, 7, 14, 6, 13, 9, 11, 4, 5, 12, 3, 10, 1, 0],
            [11, 9, 3, 0, 4, 12, 8, 7, 5, 2, 6, 14, 13, 10, 1],
            [6, 5, 0, 9, 12, 7, 11, 10, 14, 1, 13, 2, 3, 4, 8],
            [10, 13, 0, 4, 1, 5, 14, 3, 7, 6, 12, 8, 2, 9, 11],
            [5, 7, 3, 12, 13, 10, 1, 11, 8, 4, 2, 6, 0, 9, 14],
            [9, 0, 4, 8, 3, 11, 13, 14, 6, 12, 10, 2, 1, 7, 5],
            [4, 6, 7, 10, 0, 11, 1, 9, 3, 5, 13, 14, 8, 2, 12],
            [13, 4, 6, 2, 9, 14, 12, 11, 7, 10, 0, 1, 3, 8, 5],
            [9, 3, 2, 4, 13, 11, 12, 1, 0, 7, 8, 5, 14, 10, 6],
            [8, 14, 4, 3, 11, 12, 9, 0, 10, 13, 5, 1, 6, 2, 7],
            [7, 9, 8, 5, 6, 0, 2, 3, 1, 13, 14, 12, 4, 10, 11],
            [6, 0, 7, 11, 5, 14, 10, 2, 4, 13, 8, 9, 3, 12, 1],
            [13, 10, 7, 9, 5, 3, 11, 1, 12, 14, 2, 4, 0, 6, 8],
            [13, 11, 6, 8, 7, 4, 1, 5, 3, 10, 0, 14, 9, 2, 12],
        ]
        done = False
        job_nb = len(solution_sequence[0])
        machine_nb = len(solution_sequence)
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        images = []
    
        cum_reward = 0.0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if (
                        env.needed_machine_jobs[action_to_do] == machine
                        and env.legal_actions[action_to_do]
                    ):
                        print("machine {} do job {}".format(machine, action_to_do))
                        assert(env.legal_actions[action_to_do])
                        assert(sum(env.legal_actions[:-1]) ==  env.nb_legal_actions)
                        no_op = False
                        state, reward, done, _ = env.step(action_to_do)
                        cum_reward += reward
                        index_machine[machine] += 1
                        step_nb += 1
            
            if no_op and not done:
                previous_time_step = env.current_time_step
                env.increase_time_step()
                assert(env.current_time_step > previous_time_step)
        print("cummulated reward: ", cum_reward)
        env.reset()



