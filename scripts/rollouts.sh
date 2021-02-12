#!/bin/bash

# 20201124120612.235044
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201124120612.235044.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201124120612.235044/ -name eval -exec rm -rf {} \;
sed -i 's/XXXX.YYY/20201124120612.235044/g' jobs/run.py
python jobs/run.py

# 20201123123324.874708
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201123123324.874708.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201123123324.874708/ -name eval -exec rm -rf {} \;
sed -i 's/20201124120612.235044/20201123123324.874708/g' jobs/run.py
python jobs/run.py

# 20201124171233.699657
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201124171233.699657.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201124171233.699657/ -name eval -exec rm -rf {} \;
sed -i 's/20201123123324.874708/20201124171233.699657/g' jobs/run.py
python jobs/run.py

# 20201123164321.906268
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201123164321.906268.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201123164321.906268/ -name eval -exec rm -rf {} \;
sed -i 's/20201124171233.699657/20201123164321.906268/g' jobs/run.py
python jobs/run.py

# 20201125033006.751704
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201125033006.751704.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201125033006.751704/ -name eval -exec rm -rf {} \;
sed -i 's/20201123164321.906268/20201125033006.751704/g' jobs/run.py
python jobs/run.py

# 20201124013704.246797
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201124013704.246797.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201124013704.246797/ -name eval -exec rm -rf {} \;
sed -i 's/20201125033006.751704/20201124013704.246797/g' jobs/run.py
python jobs/run.py

# 20201124221348.729902
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201124221348.729902.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201124221348.729902/ -name eval -exec rm -rf {} \;
sed -i 's/20201124013704.246797/20201124221348.729902/g' jobs/run.py
python jobs/run.py

# 20201123210144.728393
tar -xf /home/psantos/ILU/ILU-RL/data/emissions/20201123210144.728393.tar.gz --directory /home/psantos/ILU/ILU-RL/data/emissions/
find /home/psantos/ILU/ILU-RL/data/emissions/20201123210144.728393/ -name eval -exec rm -rf {} \;
sed -i 's/20201124221348.729902/20201123210144.728393/g' jobs/run.py
python jobs/run.py
