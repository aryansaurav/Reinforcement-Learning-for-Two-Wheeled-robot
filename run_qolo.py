import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import cm

from mdp_qolo import mdp_qolo
from rl_agent import agent
from reward import reward


import matplotlib as mpl




def reward_construct():
    myrbfreward = reward('rbf')
    pos = np.array([0.5, 2])
    width = 0.5
    myrbfreward.setparams(pos, width)

    myrbfobstacle1 = reward('rbf')
    pos = np.array([3, 3])
    width = 0.3
    myrbfobstacle1.setparams(pos, width)

    myrbfobstacle2 = reward('rbf')
    pos = np.array([3, 1.5])
    width = 0.3
    myrbfobstacle2.setparams(pos, width)

    mydistreward = reward('dist')

    mysumreward = reward('sum')

    weights = [0, 10, -10, -10]
    mysumreward.setparams([mydistreward, myrbfreward, myrbfobstacle1, myrbfobstacle2], weights)
    return mysumreward


if __name__ == '__main__':

    my_reward = reward_construct()

    my_mdp = mdp_qolo(my_reward)

    my_mdp.build_transition_prob()
    # my_mdp.sa_s.tofile('data/qolo_sa_s5.dat')
    # my_mdp.sa_p.tofile('data/qolo_sa_p5.dat')
    # my_mdp.R.tofile('data/qolo_R5.dat')
    # my_mdp.sa_s= np.fromfile('data/qolo_sa_s5.dat', dtype='i').reshape(my_mdp.sa_s.shape)
    # my_mdp.sa_p= np.fromfile('data/qolo_sa_p5.dat').reshape(my_mdp.sa_p.shape)
    # # my_mdp.sa_p= np.fromfile('data/qolo_sa_p2.dat', dtype= 'f').reshape(my_mdp.sa_p.shape)
    # my_mdp.R = np.fromfile('data/qolo_R5.dat').reshape(my_mdp.R.shape)

    #Plotting setups

    # mpl.style.use('default')



    color = iter(cm.viridis(np.linspace(0, 1, 5)))

    fig, ax = plt.subplots()

    plt.title('Trajectory updates for varying  \n'
              ' $\gamma_t$ with reward weights = ( {:.02f},{:.02f}, {:.02f}, {:.02f})'
              .format(my_reward.theta[0],my_reward.theta[1], my_reward.theta[2], my_reward.theta[3]) )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim(my_mdp.sbounds[0,0], my_mdp.sbounds[1,0])
    plt.ylim(my_mdp.sbounds[0,1], my_mdp.sbounds[1,1])
    for i in range(1, len(my_reward.features)):
        plt.scatter(my_reward.features[i].pos[0],my_reward.features[i].pos[1])
        # plt.scatter(my_reward.features[1].pos[0],my_reward.features[1].pos[1])
    # plt.scatter(initial_state[0],initial_state[1])

    # Plotting reward function
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            z[j, i] = my_reward.evaluate(x=np.array([x[i], y[j]]))

    plt.imshow(z, extent=[0, 5, 0, 5], origin='lower')
    plt.set_cmap('pink')
    plt.colorbar()
    plt.show()


    initial_state = np.array([5,  1.3,  3.14])

    # temp_gamma_list = [0.2, 0.1, 0.05, 0.03, 0.01]
    temp_gamma_list = [0.5,  0.7, 0.9, 1.0]
    # temp_gamma_list = [ 0.5]
    # temp_gamma = 0.5

    # trig_t_list = [10,15,20, 40]
    trig_t = 15


    trig_dur = 3
    # trig_dur_list = [1, 3, 5, 7, 9]
    for temp_gamma in temp_gamma_list:
    # for trig_t in trig_t_list:

        myagent = agent(discrete_mdp=my_mdp, reward= my_reward, T=50, discount= 1, temp_discount= temp_gamma,
                        trigger_t= trig_t, trigger_duration=trig_dur, trigger_pos = my_reward.features[3].pos )
        myagent.generate_value_function()


        [traj_states, traj_actions] = myagent.soft_max_policy(initial_state)


        # traj_actions = -1*np.ones((20,2))
        # traj_actions[:,1] = -1* np.ones((20))
        # traj_states = np.zeros((11,3))
        # traj_states[0,:] = [3,3,2]
        #
        # for i in range(10):
        #     traj_ss, traj_sp,traj_states[i + 1, :]  = my_mdp.control(traj_states[i,:], traj_actions[i,:],dt=0.2)
        #     traj_states[i + 1, :] = traj_sp.dot(my_mdp.state_vals[traj_ss,:] )
        # print(traj_actions)
        # print(traj_states)
        #
        # print(myagent.reward.features[0].pos)
        # print('Time taken to generate the trajectory=', duration)

        c = next(color)
        ax.plot(traj_states[:, 0], traj_states[:, 1], c=c, marker='.', label='$\gamma_t = {:.02f}$'.format(temp_gamma), zorder=1)
        # plt.scatter(traj_states[trig_t, 0], traj_states[trig_t, 1], marker='o', color='r', zorder=2, s=120)         # Position where trigger was pressed

        # ax.plot(traj_states[:,0], traj_states[:,1], 'co')
        # ax.plot(traj_states[:,0], traj_states[:,1], 'C3-.x', label='$\gamma_t = {:.02f}$' .format(temp_gamma))
        # ax.plot(traj_states[:,0], traj_states[:,1], 'b-.x', label='$t_g = {:d}, \quad \delta t = {:d}$' .format(trig_t, trig_dur))
        # ax.plot(traj_states[:,0], traj_states[:,1], 'c--x', label=' Constant $\gamma = 1$')
        ax.legend()



