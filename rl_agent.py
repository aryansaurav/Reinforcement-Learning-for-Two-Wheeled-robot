import numpy as np

class agent:
    """
    class containing agent's policy and value function for model-based RL
    """

    def __init__(self, discrete_mdp, reward, T = 50, discount = 1,
                 temp_discount=1, trigger_t=0, trigger_duration = 1, trigger_pos = []):
        """
        initialization
        :param discrete_mdp: class containing the environment MDP information
        :param reward: class containing information about reward
        :param T: Time horizon (default = 50)
        """
        self.reward = reward
        self.mdp = discrete_mdp
        self.states = discrete_mdp.state_vals
        self.actions = discrete_mdp.action_vals
        self.nbstates = self.states.shape[0]
        self.nbactions = self.actions.shape[0]
        self.sdim = self.states.shape[1]
        self.adim = self.actions.shape[1]
        self.T = T
        self.policy = np.zeros((self.nbstates))
        self.R = np.zeros((self.nbstates, self.nbactions))
        self.value_function = np.zeros((self.nbstates, self.T))
        self.discount = discount        #discount factor
        self.temp_discount = temp_discount # Temporary discount factor at specific time-step(s)
        self.trigger_t = trigger_t         # Time step at which trigger is pressed
        self.trigger_duration = trigger_duration    #Duration of trigger press (in time steps)
        self.trigger_pos = trigger_pos      # Position of obstacle feature where trigger is likely pressed


    def generate_value_function(self, initial_state=None):
        """
        Function to generate value function
        :return: stores value function as class attribute
        """
        if initial_state is None:
            print('generating agents value function without solving MDP (initial state not given)')
        else:
            print('generating agents value function and solving MDP for given initial state')
            states = np.zeros((self.T + 1, self.sdim))
            actions = np.zeros((self.T, self.adim))
            # val = np.zeros(self.nbactions)
            states[0, :] = initial_state


        next_states = np.zeros((self.nbstates, self.nbactions, self.sdim))


        # Activate this section for parallel computing

        # state_range = range(self.nbstates)
        # action_range = range(self.nbactions)
        # paramlist = list(itertools.product(state_range,action_range))
        # print('Multiprocessing pool initiated.. CPU resources loaded')
        # pool = mp.Pool(mp.cpu_count()-1)
        # result = pool.map(self.reward_compute_parallel, paramlist)
        # self.R = np.array(result).reshape(self.R.shape)
        # print(self.R.shape)
        # pool.close()
        # print('Multiprocessing pool shutdown.. CPU resources released')

        if self.mdp.R is not None:          #if R was built in rl_agent
            self.R = self.mdp.R
        else:
            for a in range(self.nbactions):
                for s in range(self.nbstates):
                    ns_ss, ns_sp = self.mdp.control(self.states[s,:], self.actions[a,:])
                    next_states[s,a,:] = ns_sp.dot(self.mdp.state_vals[ns_ss,:] )
                    self.R[s, a] = self.reward.evaluate(next_states[s,a,:], self.actions[a,:])

        temp = np.amax(self.R, axis =1)

        self.value_function[:,self.T-1] = temp

        for t in range(self.T -2, 0, -1):
            # Introducing temporary discount factor
            # if self.trigger_t< t < self.trigger_t + self.trigger_duration:
            #     gamma = self.temp_discount
            #     print("Temporary discount factor applied at t = ", t)
            # else:
            #     gamma = self.discount
            gamma = self.discount
            temp = np.amax(self.R + gamma*np.sum(temp[self.mdp.sa_s] * self.mdp.sa_p, axis =2) , axis=1 )
            self.value_function[:,t] = temp





    def solve_mdp(self, initial_state):
        # Greedy policy

        print('Generating trajectory using greedy policy... ')
        states = np.zeros((self.T +1, self.sdim))
        actions = np.zeros((self.T, self.adim))
        # val = np.zeros(self.nbactions)
        states[0,:] = initial_state

        for t in range(self.T):
            max_val = -1000

            for a in range(self.nbactions):
                [ss, sp,next_state] = self.mdp.control(states[t,:], self.mdp.action_vals[a,:])
                val = np.sum((self.R[ss, a] + self.value_function[ss,t])*sp)
                # Introducing discount factor
                # if np.linalg.norm(states[t,0:2] - self.trigger_pos)< 0.5 :
                #     print("applying temp discount factor, ", self.temp_discount, " at t = ", t)
                #     val = np.sum((self.R[ss, a] + self.temp_discount* self.value_function[ss, t]) * sp)

                if val> max_val:
                    max_val = val
                    actions[t,:] = self.mdp.action_vals[a,:]
                    states[t+1,:] = next_state
        return states, actions


    def soft_max_policy(self, initial_state):

        print('Generating trajectory using soft-max policy... ')
        states = np.zeros((self.T + 1, self.sdim))
        actions = np.zeros((self.T, self.adim))
        val = np.zeros(self.nbactions)
        states[0, :] = initial_state

        for t in range(self.T):

            for a in range(self.nbactions):
                [ss, sp, next_state] = self.mdp.control(states[t, :], self.mdp.action_vals[a, :])
                val[a] = np.sum((self.R[ss, a] + self.value_function[ss, t]) * sp)
                # Introducing discount factor
                if  0<(next_state[0] - self.trigger_pos[0])< 0.5 :
                    print("applying temp discount factor at t = ", t)
                    val[a] = np.sum((self.R[ss, a] + self.temp_discount* self.value_function[ss, t]) * sp)

            soft_val = compute_soft_max(val)         # compute softmax of values
            actions[t,:] =  soft_val.dot(self.mdp.action_vals)
            [_, _, states[t+1,:]] = self.mdp.control(states[t, :], actions[t, :])
        return states, actions

    def reward_compute_parallel(self, sa):
        s= sa[0]
        a= sa[1]
        ns_ss, ns_sp = self.mdp.control(self.states[s,:], self.actions[a,:])
        next_state = ns_sp.dot(self.mdp.state_vals[ns_ss,:] )
        reward_val = self.reward.evaluate(next_state, self.actions[a,:])
        return reward_val


def compute_soft_max(x, tau =  0.05):
    """Compute softmax values for each sets of scores in x."""
    x = x - max(x)
    output = np.exp(x/tau) / np.sum(np.exp(x/tau), axis=0)
    if output.max() < 1:
        return output
    else:                   # To deal with NAN in soft-max computation
        output = np.zeros_like(x)
        output[x.argmax()]= 1

        print('NAN encountered. Output sum, ', output.sum())
        return output
