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
DATA_PATH = '/home/ppsantos/ILU/ILU-RL/q_vals.pickle'
OUTPUT_DIR = 'analysis/plots/policies/'
AGENT_TYPE = 'DQN'

# DDPG setup.
# DATA_PATH = '/home/ppsantos/ILU/ILU-RL/ddpg_actions.pickle'
# OUTPUT_DIR = 'analysis/plots/policies/'
# AGENT_TYPE = 'DDPG'

# QL setup.
# DATA_PATH = '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_high/20201012000831.234665/intersection_20201012-0008451602457725.6145556/checkpoints/20001/247123161.chkpt'
# OUTPUT_DIR = 'analysis/plots/policies/'
# AGENT_TYPE = 'QL'

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
                ax.set_xlabel('Waiting time phase 1')
                ax.set_ylabel('Waiting time phase 2')
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
        plt.xlabel('Waiting time phase 1')
        plt.ylabel('Waiting time phase 2')

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
        plt.xlabel('Waiting time phase 1')
        plt.ylabel('Waiting time phase 2')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(OUTPUT_DIR + 'q_val_maximizing_action.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'q_val_maximizing_action.pdf', bbox_inches='tight', pad_inches=0)

    elif AGENT_TYPE == 'DDPG':

        # Unpack data.
        X = data['X']
        Y = data['Y']
        Zs_array = data['Zs_array']

        X, Y = np.meshgrid(X, Y, indexing='ij')

        # Policy (actions) plot.
        fig = plt.figure()

        im = plt.pcolormesh(X, Y, Zs_array, cmap=cm.jet, shading='gouraud')
        plt.xlabel('Delay phase 1')
        plt.ylabel('Delay phase 2')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.title('Phase-1\nallocation')

        plt.show()

        plt.savefig(OUTPUT_DIR + 'ddpg_policy.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'ddpg_policy.pdf', bbox_inches='tight', pad_inches=0)

    elif AGENT_TYPE == 'QL':

        import operator

        bins = [0.05, 0.75, 2.25, 4.15]

        data_max_actions_matrix = np.zeros((len(bins)+1, len(bins)+1))
        data_max_q_vals_matrix = np.zeros((len(bins)+1, len(bins)+1))

        for i in range(len(bins)+1):
            for j in range(len(bins)+1):
                if sum(data[(i,j)].values()) == 0:
                    data_max_actions_matrix[i,j] = np.nan
                else:
                    data_max_actions_matrix[i,j] = max(data[(i,j)].items(), key=operator.itemgetter(1))[0]
                
                data_max_q_vals_matrix[i,j] =  max(data[(i,j)].values())

        print(data)
        print(data_max_actions_matrix)
        print(data_max_q_vals_matrix)

        # Policy (actions) plot.
        fig = plt.figure()

        data_max_actions_matrix[0,0] = np.nan

        im = plt.pcolormesh(data_max_actions_matrix, cmap=plt.get_cmap('Set2', 7), vmin=0, vmax=6)

        plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['0', '1', '2', '3', '4'])
        plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], ['0', '1', '2', '3', '4'])

        # im = plt.pcolormesh(X, Y, data_max_actions_matrix, cmap=cm.jet, shading='gouraud')

        plt.xlabel('Waiting time phase 1 (bins)')
        plt.ylabel('Waiting time phase 2 (bins)')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.title('Phase-1\nallocation')

        plt.savefig(OUTPUT_DIR + 'ql_policy.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'ql_policy.pdf', bbox_inches='tight', pad_inches=0)


        ################################################

        bins_x = [0.0, 0.05, 0.75, 2.25, 4.15]
        bins_y = [0.0, 0.05, 0.75, 2.25, 4.15]

        # Policy (actions) plot.
        fig = plt.figure()

        colors = plt.get_cmap('Set2', 7)

        ax = plt.subplot(111)
        # for x,y,w,h,c in zip(xs,ys,ws,hs,colors):
        rect = plt.Rectangle((1.0,1.0), 2.0, 3.0, color='black')
        ax.add_patch(rect)

        # cax, _ = plt.colorbar.make_axes(ax) 
        # cb2 = plt.colorbar.ColorbarBase(cax, cmap=pl.cm.jet) 

        plt.xlim(0,5)
        plt.ylim(0,5)
        plt.xlabel('Waiting time phase 1')
        plt.ylabel('Waiting time phase 2')

        plt.savefig(OUTPUT_DIR + 'ql_policy_v2.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(OUTPUT_DIR + 'ql_policy_v2.pdf', bbox_inches='tight', pad_inches=0)

    else:
        raise ValueError('Agent type not implemented')


if __name__ == "__main__":
    main()
