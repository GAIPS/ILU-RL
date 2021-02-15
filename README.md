# ILU - (I)ntegrative (L)earning from (U)rban Data: Reinforcement learning-based traffic signal controllers

This experimental project researches the development of RL-based traffic signal controllers.

## Installation 
This project requires the installation of the computational framework for reinforcement learning (RL) in traffic control [FLOW](https://github.com/flow-project/flow) and the RL framework [acme](https://github.com/deepmind/acme).

### 1) Flow 
Locally install the flow package. An installation guide can be found [here](https://flow.readthedocs.io/en/latest/flow_setup.html).
 1. Create a python 3 virtual environment for flow installation (tested with python 3.6.10)
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
	pip install -e flow/
	```
	Depending on the operating system run:
	- For Ubuntu 14.04:
		```flow/scripts/setup_sumo_ubuntu1404.sh```
	- For Ubuntu 16.04:
		```flow/scripts/setup_sumo_ubuntu1604.sh```
	- For Ubuntu 18.04:
		```flow/scripts/setup_sumo_ubuntu1804.sh```
	- For Mac:
		```flow/scripts/setup_sumo_osx.sh```

4. Test installation		
	```bash
	which sumo
	sumo --version
	sumo-gui
	python flow/examples/simulate.py ring
	```
	Note that, if the above commands did not work, you may need to run `source  ~/.bashrc` or open a 		new terminal to update your $PATH variable.
5. Exit virtual env	
	```bash
	deactivate
	```
 [Troubleshooting](https://flow.readthedocs.io/en/latest/flow_setup.html)
	
### 2) ILU-RL project
Locally install the ILU-RL package.
 1. Create a python 3 virtual environment for flow installation (tested with python 3.6.10)
	```bash
	virtualenv -p python3 env_ILU-RL
	source env_ILU-RL/bin/activate
	```
 2. Clone [ILU-RL](https://github.com/PPSantos/ILU-RL) repository
	```bash
	git clone https://github.com/PPSantos/ILU-RL
	```
3. Install packages
	```bash
	pip install -r ILU-RL/requirements.txt
	pip install -e flow/
	pip install -e ILU-RL/
	```
4. Install acme framework (checkout to b7340da4fbd1bde7d39a99e7379d1f2ce5e70965)
	```bash
	git clone https://github.com/deepmind/acme.git
	pip install -e acme/
	pip install dm-acme[reverb]
	pip install dm-acme[tf]
	pip install dm-acme[jax]
	```
4. Export root ILU-RL directory enviroment variable (configure .bashrc file)
	```bash
	export ILURL_HOME=path/to/ILU-RL/root/dir
	```
5. Test installation
	```bash
	python ILU-RL/models/train.py 
	```
