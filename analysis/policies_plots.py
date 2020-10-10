import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# How to run this script:
#
#   1) Go to the agent file and add some code to store the Q-values or policy in a pickle file.
#
#       DQN example:
#           (Add the following or equivalent lines in the acme_agent.py file)
#
#           init() method:
#               Q_net = snt.Sequential([network,])
#               self._q_net = actors_tf2.FeedForwardActor(Q_net)
#
#           deterministic_action() method:
#
#                import pickle
#                import numpy as np
#
#                Zs_array = np.zeros([7,61,121])    # Setup this depending on the MDP.
#                X = np.linspace(0,10,61)           # Setup this depending on the MDP.
#                Y = np.linspace(0,30,121)          # Setup this depending on the MDP.
#
#                for idx_x, x in enumerate(X):
#                    for idx_y, y in enumerate(Y):
#
#                        q_vals = self._q_net.select_action(np.array([x, y], dtype=np.float32))
#                        Zs_array[:,idx_x,idx_y] = q_vals
#
#                data = {'X': X, 'Y': Y, 'Zs_array': Zs_array}
#
#                with open('q_vals.pickle', 'wb') as f:
#                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#           (Run models/rollout.py script pointing to the desired experiment)
#           E.g.: python models/rollout.py -p /home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_high/20201002062037.273324/intersection_20201002-0719121601619552.1452537 -e False -n 20001
#
#           [At this point Q-values are saved in the pickle file.]
#
#
#        DDPG example:
#
#           deterministic_action() method:
#
#               import pickle
#               import numpy as np
#       
#               Zs_array = np.zeros([61,121])
#               X = np.linspace(0,10,61)
#               Y = np.linspace(0,30,121)
#       
#               for idx_x, x in enumerate(X):
#                   for idx_y, y in enumerate(Y):
#       
#                       picked_action = self._deterministic_actor.select_action(np.array([x, y], dtype=np.float32))
#                       Zs_array[idx_x,idx_y] = picked_action[0] # Store phase-0 allocations only.
#       
#               data = {'X': X, 'Y': Y, 'Zs_array': Zs_array}
#       
#               with open('ddpg_actions.pickle', 'wb') as f:
#                   pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#       
#               exit()
#       
#           (Run models/rollout.py script pointing to the desired experiment)
#           E.g.: python models/rollout.py -p /home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_high/20201002062037.273324/intersection_20201002-0719121601619552.1452537 -e False -n 20001
#
#           [At this point the policy preferences are stored in the pickle file.]
#
#
#
#   2) Setup the `DATA_PATH`, `OUTPUT_DIR`, `AGENT_TYPE` global variables bellow and run this script.
# 
#        `DATA_PATH` points to the pickle file.
#
#        `OUTPUT_DIR` points to the output directory.
#
#        `AGENT_TYPE` indicates the agent type.
#
#

# DQN setup.
# DATA_PATH = '/home/ppsantos/ILU/ILU-RL/q_vals.pickle'
# OUTPUT_DIR = 'analysis/plots/policies/'
# AGENT_TYPE = 'DQN'

# DDPG setup.
DATA_PATH = '/home/ppsantos/ILU/ILU-RL/ddpg_actions.pickle'
OUTPUT_DIR = 'analysis/plots/policies/'
AGENT_TYPE = 'DDPG'

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATA_PATH, 'rb') as handle:
        data = pickle.load(handle)

    if AGENT_TYPE == 'DQN':

        # Unpack data.
        X = data['X']
        Y = data['Y']
        Zs_array = data['Zs_array']
        NUM_ACTIONS = Zs_array.shape[0]

        X, Y = np.meshgrid(X, Y, indexing='ij')

        # Q-values plot.
        fig, axes = plt.subplots(nrows=3, ncols=3)
        fig.subplots_adjust(hspace=0.5, wspace=0.35)
        fig.set_size_inches(10.0, 9.0)

        for idx, ax in enumerate(axes.flat):
            if idx in range(NUM_ACTIONS):
                ax.set_title('Action {0}'.format(idx))
                ax.set_xlabel('Delay phase 0')
                ax.set_ylabel('Delay phase 1')
                im = ax.pcolormesh(X, Y, Zs_array[idx],
                    cmap=cm.jet, shading='gouraud',
                    vmin=np.min(Zs_array), vmax=np.max(Zs_array))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.title('Q-values')
        plt.savefig(OUTPUT_DIR + 'q_vals.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'q_vals.pdf', bbox_inches='tight', pad_inches=0)


        # Maximising actions plot.
        argmax_Zs = np.argmax(Zs_array, axis=0)

        fig = plt.figure()

        im = plt.pcolormesh(X, Y, argmax_Zs, cmap=plt.get_cmap('Set2', 7), vmin=0, vmax=6)
        plt.xlabel('Delay phase 0')
        plt.ylabel('Delay phase 1')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.title('Maximizing\naction')

        plt.savefig(OUTPUT_DIR + 'maximizing_action.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'maximizing_action.pdf', bbox_inches='tight', pad_inches=0)

        # Max q-values.
        max_Zs = np.max(Zs_array, axis=0)

        fig = plt.figure()

        im = plt.pcolormesh(X, Y, max_Zs, cmap=cm.jet, shading='gouraud',
                    vmin=np.min(Zs_array), vmax=np.max(Zs_array))
        plt.xlabel('Delay phase 0')
        plt.ylabel('Delay phase 1')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(OUTPUT_DIR + 'q_val_maximizing_action.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'q_val_maximizing_action.png', bbox_inches='tight', pad_inches=0)

    elif AGENT_TYPE == 'DDPG':

        # Unpack data.
        X = data['X']
        Y = data['Y']
        Zs_array = data['Zs_array']

        X, Y = np.meshgrid(X, Y, indexing='ij')

        # Policy (actions) plot.
        fig = plt.figure()

        im = plt.pcolormesh(X, Y, Zs_array, cmap=cm.jet, shading='gouraud')
        plt.xlabel('Delay phase 0')
        plt.ylabel('Delay phase 1')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.title('Phase-0\nallocation')

        plt.savefig(OUTPUT_DIR + 'ddpg_policy.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'ddpg_policy.pdf', bbox_inches='tight', pad_inches=0)

    else:
        raise ValueError('Agent type not implemented')


if __name__ == "__main__":
    main()
