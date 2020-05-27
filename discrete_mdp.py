import numpy as np
import math



class discrete_mdp:

    def __init__(self, test_params):
        self.sdim = test_params.sdim
        self.adim = test_params.adim
        self.sbounds = test_params.sbounds
        self.abounds = test_params.abounds
        self.cells_state = test_params.cells_state
        self.cells_action = test_params.cells_action
        self.state_vals = self.buildgrid(self.sbounds, self.cells_state)
        self.action_vals = self.buildgrid(self.abounds, self.cells_action)
        self.nb_states = self.state_vals.shape[0]
        self.nb_actions = self.action_vals.shape[0]
        # self.sa_s = np.zeros((self.nb_states, self.nb_actions), dtype='i')    #discrete case
        self.sa_s = np.zeros((self.nb_states, self.nb_actions, 2**self.sdim), dtype = 'i')   #Continuous case
        self.sa_p = np.zeros((self.nb_states, self.nb_actions, 2**self.sdim), dtype = 'i')

# Build state space and action space grids using this function
    def buildgrid(self, bounds, state):
        dim = bounds.shape[1]
        grid = np.zeros((state ** dim, dim))

        temp = np.zeros((dim, state))
        for k in range(dim):
            temp[k,:] = np.linspace(bounds[0,k], bounds[1,k], state)

        v = np.zeros((dim,1))
        for i in range(state**dim):
            for j in range(dim):
                v[j] = math.floor(i/(state**j))
                if v[j] > state-1:
                    v[j] = v[j] % state
                grid[i,j] = temp[j,int(v[j])]

        return grid


    def control(self, current_state, action, dt = 0.2):

        # Function to define the control law,
        # INPUT---
        # current_state : (T x dims)
        # action: (T x dima)
        # dt: time-step scalar
        # OUTPUT---
        # next_state: (T x dims)


        dims = current_state.shape[0]
        if action.ndim > 1:     # multiple time steps
            T = action.shape[0]
            dima = action.shape[1]
        else:
            T = 1
            dima = action.shape[0]

        next_state = np.zeros((T, dims))

        #Control law
        next_state = np.clip(current_state + action * dt, self.sbounds[0,:], self.sbounds[1,:])

        return next_state


    def state2index(self, state):
        # Function to convert the given state into the index on state_vals list
        # state : (dims, T)
        # indices: (1, T)
        step_size = (self.sbounds[1,:]- self.sbounds[0,:])/(self.cells_state-1)
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

    def build_transition_prob(self):
        # Function to generate transition probability matrix
        # OUTPUT---
        # sa_s : (nb_states, nb_actions, 2**dims)


        for a in range(self.nb_actions):
            for s in range(self.nb_states):
                next_state = self.control(self.state_vals[s,:], self.action_vals[a,:])
                [self.sa_s[s,a,:], self.sa_p[s,a,:]] = self.interpolate_state(next_state)   #Continuous case
                # self.sa_s[s,a] = self.state2index(next_state)     #Discrete case
                # if self.sa_s[s,a] == -1:        #if taking action a in state s leads to outside bound
                #     self.sa_s[s,a] = s
                #     print('going out of bounds')


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
        sa_p = np.ones((2**dim))

        l_id = np.zeros((dim))
        u_id = np.zeros((dim))
        l_prob = np.zeros((dim))
        u_prob = np.zeros((dim))

        state = state - bounds[0,:]

        for i in range(dim):
            step_size = (bounds[1,:] - bounds[0,:])/(cells-1)
            l_id[i] = math.floor(state[i]/step_size[i] )
            u_id[i] = math.ceil(state[i]/step_size[i] )
            l_prob[i] = math.modf(state[i]/step_size[i])[0]
            u_prob[i] = (1 - math.modf(state[i] / step_size[i])[0] )


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
