import numpy as np
import math
import multiprocessing as mp
import itertools
from reward import reward

class mdp_qolo:

    def __init__(self, reward =None):
        self.dt = 0.1     #timestep for control functions
        self.sdim = 3       # x, y, theta
        self.adim = 2
        self.sbounds = np.array([[0, 0, 0],[5, 5, 2*math.pi]])
        self.abounds = np.array([[0,  -1],[1, 1]])       # Speed limits TO BE SET (linear, angular)
        self.cells_state = np.array([5,5,11])           # 25 cells in theta at every 15 degrees
        self.cells_action = np.array([10,10])
        self.state_vals = self.buildgrid(self.sbounds, self.cells_state)
        self.action_vals = self.buildgrid(self.abounds, self.cells_action)
        self.nb_states = self.cells_state.prod()
        self.nb_actions = self.cells_action.prod()
        # self.sa_s = np.zeros((self.nb_states, self.nb_actions), dtype='i')    #discrete case
        self.sa_s = np.zeros((self.nb_states, self.nb_actions, 2**self.sdim), dtype = 'i')   #Continuous case
        self.sa_p = np.zeros((self.nb_states, self.nb_actions, 2**self.sdim), dtype = 'i')

        #Build reward matrix on state, action pairs if reward given
        if reward is not None:
            print('Reward supplied to MDP, building R(s,a) matrix with sa_s, sa_p')
            self.reward = reward
            self.R = np.zeros((self.nb_states, self.nb_actions))
        else:
            print('Reward not supplied to MDP, R(s,a) to be built by rl_agent class')
            self.R = None
# Build state space and action space grids using this function
    def buildgrid(self, bounds, state):
        dim = bounds.shape[1]
        nb_state = int(state.prod())
        grid = np.zeros((nb_state, dim))

        temp = np.zeros((dim, state.max()))
        for k in range(dim):
            temp[k,range(state[k])] = np.linspace(bounds[0,k], bounds[1,k], state[k])

        v = np.zeros((dim,1))
        for i in range(nb_state):
            for j in range(dim):
                v[j] = math.floor(i/(state[range(j)].prod()))
                if v[j] > state[j]-1:
                    v[j] = v[j] % state[j]
                grid[i,j] = temp[j,int(v[j])]

        return grid


    def control(self, current_state, action = None, dt = None):

        # Function to define the control law,
        # INPUT---
        # current_state : (1 x dims)
        # action: (1 x dima)
        # dt: time-step scalar
        # OUTPUT---
        # sa_s, sa_p (next_state) : (1 x dims)

        if dt is None:
            dt = self.dt

        if action is None:          # No action given, then Parallel processing
            # print(current_state)
            action = self.action_vals[current_state[1],:]
            current_state = self.state_vals[current_state[0],:]

        next_state = np.zeros(current_state.shape)
        next_state[0] = current_state[0] + action[0]*math.cos(current_state[2])*dt
        next_state[1] = current_state[1] + action[0]*math.sin(current_state[2])*dt
        next_state[2] = current_state[2] + action[1]*dt

        if next_state[2]> 2* math.pi:
            next_state[2] -= 2* math.pi
        if next_state[2]<0:
            next_state[2] += 2* math.pi

        next_state = next_state.clip(self.sbounds[0,:], self.sbounds[1,:])
        [ss_ns, sp_ns] = self.interpolate_state(next_state)
        next_state = sp_ns.dot(self.state_vals[ss_ns, :])
        return ss_ns, sp_ns, next_state

    def control_indices(self, s, a = None, dt = None):

        # Function to define the control law,
        # INPUT---
        # s : (1 x 1) state index
        # a: (1 x 1) action index
        # dt: time-step scalar
        # OUTPUT---
        # sa_s, sa_p (next_state) : (1 x dims)
        if dt is None:
            dt = self.dt

        if a is None:          # No action given, then Parallel processing
            a= s[1]
            s= s[0]

        action = self.action_vals[a,:]
        current_state = self.state_vals[s,:]

        next_state = np.zeros(current_state.shape)
        next_state[0] = current_state[0] + action[0]*math.cos(current_state[2])*dt
        next_state[1] = current_state[1] + action[0]*math.sin(current_state[2])*dt
        next_state[2] = current_state[2] + action[1]*dt

        if next_state[2]> 2* math.pi:
            next_state[2] -= 2* math.pi
        if next_state[2]<0:
            next_state[2] += 2* math.pi

        next_state = next_state.clip(self.sbounds[0,:], self.sbounds[1,:])

        [ss_ns, sp_ns] = self.interpolate_state(next_state)
        # next_state = sp_ns.dot(self.state_vals[ns_ss, :])
        r = self.reward.evaluate(next_state, action) * np.ones(ss_ns.shape)
        # self.sa_s[s,a,:] = ss_ns
        # self.sa_p[s,a,:] = sp_ns
        return ss_ns, sp_ns, r



    def build_transition_prob(self):
        # Function to generate transition probability matrix
        # OUTPUT---
        # sa_s : (nb_states, nb_actions, 2**dims)

        state_range = range(self.nb_states)
        action_range = range(self.nb_actions)
        paramlist = list(itertools.product(state_range,action_range))
        # print(paramlist)
        print('Multiprocessing pool initiated.. CPU resources loaded')
        pool = mp.Pool(mp.cpu_count()-1)
        result = pool.map(self.control_indices, paramlist)
        result = np.array(result)
        self.sa_s = np.array(result[:,0,:],dtype= 'i').reshape(self.sa_s.shape)
        self.sa_p = np.array(result[:,1,:]).reshape(self.sa_p.shape)
        self.R = np.array(result[:,2,0]).reshape(self.R.shape)
        pool.close()
        print('Multiprocessing pool shutdown.. CPU resources released')

        # for a in range(self.nb_actions):
        #     for s in range(self.nb_states):
        #         # next_state = self.control(self.state_vals[s,:], self.action_vals[a,:])
        #         [self.sa_s[s,a,:], self.sa_p[s,a,:]] = self.control(self.state_vals[s,:], self.action_vals[a,:])   #Continuous case
        #         # self.sa_s[s,a] = self.state2index(next_state)     #Discrete case
        #         # if self.sa_s[s,a] == -1:        #if taking action a in state s leads to outside bound
        #         #     self.sa_s[s,a] = s
        #         #     print('going out of bounds')


    def interpolate_state(self, state):
        """
        Function to interpolate given state onto the grid
        :param state: (sdim X 1) states to be interpolated
        :return:
            sa_s: (2^sdim X 1) possible states in indices
            sa_p: (2^sdim X 1) probabilities of possible states
        """
        bounds = self.sbounds
        cells = self.cells_state
        dim = state.shape[0]
        global_grid = cells * np.ones(dim)

        sa_s = np.zeros((2**dim), dtype= 'i')
        sa_p = np.zeros((2**dim))

        l_id = np.zeros((dim))
        u_id = np.zeros((dim))
        l_prob = np.zeros((dim))
        u_prob = np.zeros((dim))

        state = state - bounds[0,:]

        for i in range(dim):
            step_size = (bounds[1,:] - bounds[0,:])/(cells[:]-1)
            l_id[i] = math.floor(state[i]/step_size[i] )
            u_id[i] = math.ceil(state[i]/step_size[i] )
            u_prob[i] = math.modf(state[i]/step_size[i])[0]
            l_prob[i] = (1 - math.modf(state[i] / step_size[i])[0] )


        pts = np.zeros((2,dim))
        pts[0,:] = l_id
        pts[1,:] = u_id

        probs = np.zeros((2, dim))
        probs[0,:] = l_prob
        probs[1,:] = u_prob

        for i in range(2**dim):
            local_grid = np.ones((dim))*2 # specifying two elements along each dimension
            nncoord = self.ind2coords(i, local_grid)
            nextprob = 1
            nextcoord = np.zeros((1, dim))
            for m in range(dim):
                nextprob = nextprob * probs[int(nncoord[m]),m]
                nextcoord[:,m] = pts[int(nncoord[m]),m]
            sa_s[i] = self.coord2ind(nextcoord, global_grid)
            sa_p[i] = nextprob
        return sa_s, sa_p


    def ind2coords(self, indices, dims):
        """
        Helper function to convert list of indices to list of coordinates along each dimension in dims
        :param indices: (N,1) index array
        :param dims: (dim, 1) number of elements along each dimension
        :return:    (N, dim) N array of coordinate values corresponding to the indices array
        """
        coords = np.zeros(len(dims))

        for k in range(len(dims)):
            coords[k] = indices % dims[k]
            indices = math.floor(indices/ dims[k])

        return coords


    def coord2ind(self, coords, dims):
        """
        Function to convert list of coords into indices
        :param coords: (N, dim) list of N coordinates
        :param dims: (dim,1) list of number of elements along each dimension
        :return: (N, 1) list of N indices corresponding to the coordinates
        """

        indices = np.zeros((len(coords),1))
        fac = 1

        for k in range(len(dims)):
            indices = indices + coords[:,k]*fac
            fac = fac*dims[k]

        return indices


    def state2index(self, state):
        # Function to convert the given state into the index on state_vals list
        # state : (dims, T)
        # indices: (1, T)
        step_size = (self.sbounds[1,:]- self.sbounds[0,:])/(self.cells_state[:]-1)
        if state.ndim>1:
            T = state.shape[1]
        else:
            T = 1
            state = state.reshape(self.sdim,1)

        indout = []
        indices = np.zeros(( T), dtype= 'i')

        for t in range(T):
            # for i in range(self.sdim):
            #     print(np.where(np.logical_and(state[i, t] >= self.state_vals[:,i], state[i, t] < self.state_vals[:,i] + step_size[i] )))
            temp = (np.where(np.all(np.logical_and(state[:, t] >= self.state_vals[:,:], state[:, t] < self.state_vals[:,:] + step_size[:] ),axis=1)))[0]
            if temp[0] != None:
                indices[t] = temp[0]
            else:
                indices[t] = -1
        return indices
