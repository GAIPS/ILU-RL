# ILU - (I)ntegrative (L)earning from (U)rban Data: Reinforcement learning-based traffic signal controllers

This experimental project researches the development of RL-based traffic signal controllers.

## Installation 
This project requires the installation of the computational framework for reinforcement learning (RL) in traffic control [FLOW](https://github.com/flow-project/flow) and the RL algorithms package [OpenAI Baselines](https://github.com/openai/baselines).

### 1) Flow 
Locally install the flow package. An installation guide can be found [here](https://flow.readthedocs.io/en/latest/flow_setup.html).
 1. Create a python 3 virtual environment for flow installation (tested with python 3.6.8)
	```bash
	virtualenv -p python3 env_flow
	source env_flow/bin/activate
	```
 2. Clone [FLOW-Project](https://github.com/flow-project/flow) repository
	```bash
	git clone https://github.com/flow-project/flow
	```
 3. Install FLOW
	```bash
	cd flow
	pip install -e
	```
	Depending on the operating system run:
		- For Ubuntu 14.04: ```scripts/setup_sumo_ubuntu1404.sh```
		- For Ubuntu 16.04: ```scripts/setup_sumo_ubuntu1604.sh```
		- For Ubuntu 18.04: ```scripts/setup_sumo_ubuntu1804.sh```
		- For Mac: ```scripts/setup_sumo_osx.sh```

4. Test installation		
	```bash
	which sumo
	sumo --version
	sumo-gui
	python examples/simulate.py ring
	```
	Note that, if the above commands did not work, you may need to run `source  ~/.bashrc` or open a 		new terminal to update your $PATH variable.
5. Exit virtual env	
	```bash
	deactivate
	cd ..
	```
 [Troubleshooting](https://flow.readthedocs.io/en/latest/flow_setup.html)
	
### 2) ILU-RL project
Locally install the ILU-RL package.
 1. Create a python 3 virtual environment for flow installation (tested with python 3.6.8)
	```bash
	virtualenv -p python3 env_ILU-RL
	source env_ILU-RL/bin/activate
	```
 2. Clone OpenAI baselines and install tensorflow
	```bash
	git clone https://github.com/PPSantos/baselines
	```
	Install tensorflow:
	```bash
	pip install tensorflow-gpu==1.14
	```
	or
	```bash
	pip install tensorflow==1.14
	```
 3. Clone [ILU-RL](https://github.com/PPSantos/ILU-RL) repository
	```bash
	git clone https://github.com/PPSantos/ILU-RL
	```
4. Install packages
	```bash
	cd ILU-RL
	pip install -r requirements.txt
	pip install -e ../flow
	pip install -e ../baselines
	pip install -e .
	```
4. Export root ILU-RL directory enviroment variable (configure .bashrc file)
	```bash
	export ILURL_HOME=path/to/ILU-RL/root/dir
	```
5. Test installation
	```bash
	python models/train.py 
	```