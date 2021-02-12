import os
import re
import json
import tarfile
import pandas as pd
import argparse
from pathlib import Path
import configparser
import tempfile
import shutil

# Path to emissions folders.
EMISSIONS_PATH_1 = '/home/psantos/ILU/ILU-RL/data/emissions/'
EMISSIONS_PATH_2 = '/home/psantos/ILU/ILU-RL-2/data/emissions/'
EMISSIONS_PATH_3 = '/home/gvarela/ilu/ilurl/data/emissions/'
EMISSIONS_PATH_4 = '/home/gvarela/ilu/altrl/data/emissions/'

# Intersection + constant (high).
""" TOP_K = 3
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Static': '20201205150734.745147',
    'Webster': '20201206020205.268465',
    'Max-pressure': '20201205221657.264759',
    'Actuated': '20201205232304.764324',
}
RL = {
    'QL': {
        'Random': '20201203002410.465059',
        'Min. speed delta': '20201119150051.524184',
        'Min. delay': '20201120015727.261873',
        'Max. delay red.': '20201122163308.836162',
        'Min. waiting time': '20201228005357.773218',
        'Min. queue': '20201211170525.971782',
        'Max. queue red.': '20201120091613.848725',
        'Min. pressure': '20201228122903.543258',
    },
    'DQN': {
        'Random': '20201203002410.465059',
        'Min. speed delta': '20201019201927.705392',
        'Min. delay': '20201019121818.176705',
        'Max. delay red.': '20201212023738.171221',
        'Min. waiting time': '20201020120029.196691',
        'Min. queue': '20201125193144.350360',
        'Max. queue red.': '20201020043124.208366',
        'Min. pressure': '20201125120732.210657',
    },
    'DDPG': {
        'Random': '20201206010857.551559',
        'Min. speed delta': '20201112104325.247084',
        'Min. delay': '20201111095306.078216',
        'Max. delay red.': '20201126074429.705712',
        'Min. waiting time': '20201113074152.431465',
        'Min. queue': '20201126125852.876314',
        'Max. queue red.': '20201112231009.228489',
        'Min. pressure': '20201126023707.247087',
    }
} """

# Intersection + constant (low).
""" TOP_K = 3
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Static': '20201206011109.441105',
    'Webster': '20201206194519.870350',
    'Max-pressure': '20201206201147.378733',
    'Actuated': '20201206212635.207017',
}
RL = {
    'QL': {
        'Random': '20201206230544.309790',
        'Min. speed delta': '20201228185736.808873',
        'Min. delay': '20201123205612.190239',
        'Max. delay red.': '20201125093216.415102',
        'Min. waiting time': '20201124091359.128610',
        'Min. queue': '20201212094406.166684',
        'Max. queue red.': '20201124035929.239557',
        'Min. pressure': '20201124203350.027420',
    },
    'DQN': {
        'Random': '20201206230544.309790',
        'Min. speed delta': '20201120112049.546333',
        'Min. delay': '20201120162302.197023',
        'Max. delay red.': '20201127204521.381258',
        'Min. waiting time': '20201121022041.160455',
        'Min. queue': '20201128014914.794302',
        'Max. queue red.': '20201120211130.274071',
        'Min. pressure': '20201127164149.877018',
    },
    'DDPG': {
        'Random': '20201206235012.632390',
        'Min. speed delta': '20201116162934.032056',
        'Min. delay': '20201116213013.033741',
        'Max. delay red.': '20201227160831.998469',
        'Min. waiting time': '20201117073004.695373',
        'Min. queue': '20201128210244.601379',
        'Max. queue red.': '20201117022052.704456',
        'Min. pressure': '20201128120514.594441',
    }
} """

# Intersection + variable.
TOP_K = 3
IS_VARIABLE_DEMAND = True
BASELINES = {
    'Webster': '20201214011935.784512',
    'Max-pressure': '20201209165406.757524',
    'Actuated': '20201209162903.206448',
}
RL = {
    'QL': {
        'Random': '20201209130259.367330',
        'Min. speed delta': '20201118093331.672748',
        'Min. delay': '20201118153928.572475',
        'Max. delay red.': '20201112110459.748237',
        'Min. waiting time': '20201119035223.600767',
        'Min. queue': '20201218201048.350916',
        'Max. queue red.': '20201118211334.654831',
        'Min. pressure': '20201111193154.062514',
    },
    'DQN': {
        'Random': '20201209130259.367330',
        'Min. speed delta': '20201121170515.395018',
        'Min. speed delta +': '20201122135938.826760',
        'Min. delay': '20201121221710.122457',
        'Min. delay +': '20201122181740.712720',
        'Max. delay red.': '20201228225324.556912',
        'Max. delay red. +': '20201127033948.250453',
        'Min. waiting time': '20201122083107.504787',
        'Min. waiting time +': '20201123043023.331301',
        'Min. queue': '20201229033621.666694',
        'Min. queue +': '20201127074043.189437',
        'Max. queue red.': '20201122031433.851053',
        'Max. queue red. +': '20201122232654.297279',
        'Min. pressure +': '20201126233524.503159',
    },
    'DDPG': {
        'Random': '20201209155831.283414',
        'Min. speed delta': '20201124120612.235044',
        'Min. speed delta +': '20201123123324.874708',
        'Min. delay': '20201124171233.699657',
        'Min. delay +': '20201123164321.906268',
        'Max. delay red. +': '20201220205521.246754',
        'Min. waiting time': '20201125033006.751704',
        'Min. waiting time +': '20201124013704.246797',
        'Min. queue': '20201214022910.446978',
        'Max. queue red.': '20201124221348.729902',
        'Max. queue red. +': '20201123210144.728393',
        'Min. pressure +': '20201219021801.330996',
    }
}


# Intersection + cyclical.
""" TOP_K = 3
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Webster': '20201209011830.238844',
    'Adaptive-Webster': '20201209002445.668105',
    'Max-pressure': '20201207153625.330330',
    'Actuated': '20201207143756.671053',
}
RL = {
    'QL': {
        'Random': '20201208182059.479869',
        'Min. speed delta': '20201115172334.909118',
        'Min. delay': '20201116010251.913711',
        'Max. delay red.': '20201214065748.211529',
        'Min. waiting time': '20201116155318.608873',
        'Min. queue': '20201227173335.091130',
        'Max. queue red.': '20201116084643.482144',
        'Min. pressure': '20201117090440.547649',
    },
    'DQN': {
        'Random': '20201208182059.479869',
        'Min. speed delta': '20201118172555.618638',
        'Min. delay': '20201119000945.325697',
        'Max. delay red.': '20201119192507.625641',
        'Min. waiting time': '20201119124205.944699',
        'Min. queue': '20201220114333.124004',
        'Max. queue red.': '20201119065027.922979',
        'Min. pressure': '20201220010320.228470',
    },
    'DDPG': {
        'Random': '20201208202119.374929',
        'Min. speed delta': '20201117145441.965866',
        'Min. delay': '20201117213248.372374',
        # 'Max. delay red.': '',
        'Min. waiting time': '20201118081605.895669',
        # 'Min. queue': '',
        'Max. queue red.': '20201118031158.389225',
        # 'Min. pressure': '',
    }
} """

# Arterial + constant.
""" TOP_K = 3
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Webster': '20201210132209.358017',
    'Max-pressure': '20201210125059.851038',
    'Actuated': '20201210114217.937617',
}
RL = {
    'DQN': {
        'Random': '20201210105714.687376',
        'Min. speed delta': '20201202215415.753237',
        'Min. delay': '20201203021258.093153',
        # 'Max. delay red.': '',
        'Min. waiting time': '20201203065823.878991',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201220234357.879908',
    },
    'DDPG': {
        'Random': '20201216174726.529215',
        'Min. speed delta': '20201129235804.452288',
        'Min. delay': '20201130082448.166997',
        #'Max. delay red.': '',
        'Min. waiting time': '20201130155029.716761',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201222211332.508037',
    }
} """

# Arterial + variable.
""" TOP_K = 3
IS_VARIABLE_DEMAND = True
BASELINES = {
    'Webster': '20201210135438.786714',
    'Max-pressure': '20201205024623.091722',
    'Actuated': '20201205031053.139542',
}
RL = {
    'DQN': {
        'Random': '20201205125759.725263',
        'Min. speed delta +': '20201203225910.235100',
        'Min. delay +': '20201204075354.513620',
        'Max. delay red. +': '20201205182021.469367',
        'Min. waiting time +': '20201205013839.706850',
        #'Min. queue +': '',
        'Max. queue red. +': '20201204164239.706407',
        'Min. pressure +': '20201205093430.293706',
    },
    'DDPG': {
        'Random': '20201220181129.342083',
        'Min. speed delta +': '20201203212704.126277',
        'Min. delay +': '20201204042723.292757',
        # 'Max. delay red. +': '',
        'Min. waiting time +': '20201204114008.696502',
        # 'Min. queue +': '',
        # 'Max. queue red. +': '',
        'Min. pressure +': '20201204184344.125427',
    }
} """

# Arterial + cyclical.
""" TOP_K = 3
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Webster': '20201210144720.677525',
    'Adaptive-Webster': '20201210163144.133986',
    'Max-pressure': '20201210040620.824887',
    'Actuated': '20201210032656.058303',
}
RL = {
    'DQN': {
        'Random': '20201210051915.285597',
        'Min. speed delta': '20201206025355.629688',
        'Min. delay': '20201206133052.764158',
        # 'Max. delay red.': '',
        'Min. waiting time': '20201207013645.768954',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201207143116.987306',
    },
    'DDPG': {
        'Random': '20201219200303.825986',
        'Min. speed delta': '20201208022123.343251',
        'Min. delay': '20201208125219.278285',
        #'Max. delay red.': '',
        'Min. waiting time': '20201208224114.043723',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201209085435.658598',
    }
} """

# Grid + constant.
""" TOP_K = 1
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Webster': '20201215113333.410427',
    'Max-pressure': '20201214181126.171920',
    'Actuated': '20201214172713.564156',
}
RL = {
    'DQN': {
        'Random': '20201214194844.533431',
        'Min. speed delta': '20201214233231.894269',
        'Min. delay': '20201215094728.503560',
        #'Max. delay red.': '',
        'Min. waiting time': '20201215200235.033052',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201216055902.460132',
    },
    'DDPG': {
        'Random': '20201219172040.147693',
        'Min. speed delta': '20201218120012.988925',
        'Min. delay': '20201218222958.537780',
        #'Max. delay red.': '',
        'Min. waiting time': '20201219095046.019000',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201219203143.842703',
    }
} """

# Grid + variable.
""" TOP_K = 1
IS_VARIABLE_DEMAND = True
BASELINES = {
    'Webster': '20201215021707.422089',
    'Max-pressure': '20201214210144.215551',
    'Actuated': '20201214203505.667905',
}
RL = {
    'DQN': {
        'Random': '20201214221653.755194',
        'Min. speed delta +': '20201214202159.111828',
        'Min. delay +': '20201215143424.382793',
        #'Max. delay red. +': '',
        'Min. waiting time +': '20201215222207.141950',
        #'Min. queue +': '',
        #'Max. queue red. +': '',
        'Min. pressure +': '20201215045810.712956',
    },
    'DDPG': {
        'Random': '20201219010808.609596',
        'Min. speed delta +': '20201218102506.222433',
        'Min. delay +': '20201216174634.394295',
        #'Max. delay red. +': '',
        'Min. waiting time +': '20201217012200.004624',
        #'Min. queue +': '',
        #'Max. queue red. +': '',
        'Min. pressure +': '20201217171325.606227',
    }
} """

# Grid + cyclical.
""" TOP_K = 1
IS_VARIABLE_DEMAND = False
BASELINES = {
    'Webster': '20201211001720.355868',
    'Adaptive-Webster': '20201214155654.724833',
    'Max-pressure': '20201214145521.620478',
    'Actuated': '20201214135149.717254',
}
RL = {
    'DQN': {
        'Random': '20201214115540.279231',
        'Min. speed delta': '20201212191339.997485',
        'Min. delay': '20201213071149.324384',
        #'Max. delay red.': '',
        'Min. waiting time': '20201213180132.060303',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201219203448.568805',
    },
    'DDPG': {
        'Random': '20201219180730.564815',
        'Min. speed delta': '20201216180104.862810',
        'Min. delay': '20201217042328.635702',
        #'Max. delay red.': '',
        'Min. waiting time': '20201217143302.435898',
        #'Min. queue': '',
        #'Max. queue red.': '',
        'Min. pressure': '20201218002644.946340',
    }
} """

# Test.
# EMISSIONS_PATH_1 = '/home/ppsantos/ILU/ILU-RL/data/emissions/'
# EMISSIONS_PATH_2 = '/home/ppsantos/ILU/ILU-RL/data/experiments/chapter_1/demand_high/'
# BASELINES = {
#     'Static': '20201214011935.784512',
#     'Actuated': '20201209162903.206448',
# }

# RL = {
#     'QL': {
#         'Min. speed delta': '20201020120029.196691',
#         'Min. delay': '20201020120029.196691',
#     },
#     'DQN': {
#         'Min. speed delta': '20201020120029.196691',
#         'Min. delay': '20201020120029.196691',
#     },
#     # 'DDPG': [],
# }

repo_1_files = list(Path(EMISSIONS_PATH_1).rglob('*.tar.gz'))
repo_1_files = [p.name for p in repo_1_files]

repo_2_files = list(Path(EMISSIONS_PATH_2).rglob('*.tar.gz'))
repo_2_files = [p.name for p in repo_2_files]

repo_3_files = list(Path(EMISSIONS_PATH_3).rglob('*.tar.gz'))
repo_3_files = [p.name for p in repo_3_files]

repo_4_files = list(Path(EMISSIONS_PATH_4).rglob('*.tar.gz'))
repo_4_files = [p.name for p in repo_4_files]

# print(repo_1_files)
# print(repo_2_files)
# print(repo_3_files)
# print(repo_4_files)


out = ""

# Baselines.
for (controller, exp_name) in BASELINES.items():

    out += f'{controller} & N/A & N/A & '

    if (exp_name + '.tar.gz') in repo_1_files:
        path = EMISSIONS_PATH_1
    elif (exp_name + '.tar.gz') in repo_2_files:
        path = EMISSIONS_PATH_2
    elif (exp_name + '.tar.gz') in repo_3_files:
        path = EMISSIONS_PATH_3
    elif (exp_name + '.tar.gz') in repo_4_files:
        path = EMISSIONS_PATH_4
    else:
        raise ValueError(f'Unknown experiment: {exp_name}')

    exp_path = path +  exp_name + '.tar.gz'

    print(controller, exp_name)
    # print(exp_path)
    tar = tarfile.open(exp_path)

    # Num stops.
    df = pd.read_csv(
        tar.extractfile("{0}/plots/test/stops_stats.csv".format(exp_name))
    )
    mean = df.iloc[0,1]
    out += f'{round(mean,2):.2f} & '
    
    # Waiting time.
    df = pd.read_csv(
        tar.extractfile("{0}/plots/test/waiting_time_stats.csv".format(exp_name))
    )
    mean = df.iloc[0,1]
    std  = df.iloc[1,1]
    out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '


    if IS_VARIABLE_DEMAND:

        # Travel time free-flow.
        df = pd.read_csv(
            tar.extractfile("{0}/plots/test/travel_time_free_flow_stats.csv".format(exp_name))
        )
        mean = df.iloc[0,1]
        std  = df.iloc[1,1]
        out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '

        # Travel time congested.
        df = pd.read_csv(
            tar.extractfile("{0}/plots/test/travel_time_congested_stats.csv".format(exp_name))
        )
        mean = df.iloc[0,1]
        std  = df.iloc[1,1]
        out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '

    # Travel time.
    df = pd.read_csv(
        tar.extractfile("{0}/plots/test/travel_time_stats.csv".format(exp_name))
    )
    mean = df.iloc[0,1]
    std  = df.iloc[1,1]
    out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '
    
    # Top-k travel time.
    df_metrics = pd.read_csv(
        tar.extractfile(f'{exp_name}/plots/test/{exp_name}_metrics.csv')
    )
    top_k = df_metrics.nsmallest(TOP_K, 'travel_time')
    top_k_mean = top_k.mean()
    out += f'{round(top_k_mean["travel_time"], 1):.1f}'


    out += ' \\\\ \\hline \n'

for (rl_type, mdps) in RL.items():

    print(rl_type) #, mdps)

    for (mdp, exp_name) in mdps.items():

        print(mdp, exp_name)

        if (exp_name + '.tar.gz') in repo_1_files:
            path = EMISSIONS_PATH_1
        elif (exp_name + '.tar.gz') in repo_2_files:
            path = EMISSIONS_PATH_2
        elif (exp_name + '.tar.gz') in repo_3_files:
            path = EMISSIONS_PATH_3
        elif (exp_name + '.tar.gz') in repo_4_files:
            path = EMISSIONS_PATH_4
        else:
            raise ValueError(f'Unknown experiment: {exp_name}')

        exp_path = path + exp_name + '.tar.gz'

        # print(mdp, exp_path)
        tar = tarfile.open(exp_path)

        if mdp == 'Random':
            out += f'{rl_type} (Random) & N/A & N/A & '
        else:
            out += f'{rl_type} & \\textit{{{mdp}}} & '

            # Reward.
            df = pd.read_csv(
                tar.extractfile("{0}/plots/test/cumulative_reward.csv".format(exp_name))
            , header=None)
            mean = df.iloc[0,1]
            out += f'{round(mean,2):.2f} & '

        # Num stops.
        df = pd.read_csv(
            tar.extractfile("{0}/plots/test/stops_stats.csv".format(exp_name))
        )
        mean = df.iloc[0,1]
        out += f'{round(mean,2):.2f} & '
        
        # Waiting time.
        df = pd.read_csv(
            tar.extractfile("{0}/plots/test/waiting_time_stats.csv".format(exp_name))
        )
        mean = df.iloc[0,1]
        std  = df.iloc[1,1]
        out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '

        if IS_VARIABLE_DEMAND:

            # Travel time free-flow.
            df = pd.read_csv(
                tar.extractfile("{0}/plots/test/travel_time_free_flow_stats.csv".format(exp_name))
            )
            mean = df.iloc[0,1]
            std  = df.iloc[1,1]
            out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '

            # Travel time congested.
            df = pd.read_csv(
                tar.extractfile("{0}/plots/test/travel_time_congested_stats.csv".format(exp_name))
            )
            mean = df.iloc[0,1]
            std  = df.iloc[1,1]
            out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '

        # Travel time.
        df = pd.read_csv(
            tar.extractfile("{0}/plots/test/travel_time_stats.csv".format(exp_name))
        )
        mean = df.iloc[0,1]
        std  = df.iloc[1,1]
        out += f'({round(mean,1):.1f}, {round(std,1):.1f}) & '
        
        # Top-k travel time.
        df_metrics = pd.read_csv(
            tar.extractfile(f'{exp_name}/plots/test/{exp_name}_metrics.csv')
        )
        if mdp != 'Random':
            df_metrics = df_metrics.groupby(['train_run']).mean()
        top_k = df_metrics.nsmallest(TOP_K, 'travel_time')
        top_k_mean = top_k.mean()
        out += f'{round(top_k_mean["travel_time"], 1):.1f}'

        out += ' \\\\ \\hline \n'


print('-'*30)
print('\n')
print(out)
