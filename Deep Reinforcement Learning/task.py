import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

#         self.state_size = self.action_repeat * 6
        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose and vel of sim to return reward."""
#         reward = 1-0.001*(abs(self.sim.pose[:3] - self.target_pos)).sum()
#         reward = reward + self.sim.v[2]
#         reward = reward - 0.000001*((abs(self.sim.angular_v[1:])).sum())
#         reward = reward - 0.000001*((abs(self.sim.pose[4:])).sum())
#         reward = np.clip(reward, -100, 100)
        
        ## Test
        #reward for continue flying and penalty for being far from the target
        reward = 0.001*self.sim.v[2]
        reward += 5-0.00001*((self.sim.pose[:3] - self.target_pos)**2).sum()
        reward -= 0.0001*(abs(self.sim.angular_v[0])).sum()
        reward -= 0.0001*(abs(self.sim.angular_v[1])).sum()
        

        # penalty for terminating the simulation before the runtime is over    
        if self.sim.time < self.sim.runtime and self.sim.done == True:
            reward -= 500
            
#         reward = np.clip(reward, -1000, 1000) # Ensure there are no extreme outliers
        ## End test
        
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            
            # Extra parameters for exploring purposes
            pose_all.append(self.sim.v)
            pose_all.append(self.sim.angular_v)
            # Up to here
            
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v] * self.action_repeat) 
        return state