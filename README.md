# 2022 COG RoboMaster Sim2Real Challenge
This repository is related to the 2022 CoG Robomaster Sim2Real Challenge which is organized by CASIA DRL Team.In this repository, we will provide the competiton environment for you, and you can also find the Instructions of the necessary environment API in `api_test.py`.            


## Install

We recommend using conda to create a virtual environment. 

Step 1.We provide two environments for you to use different os(Windows/Linux) to paticipate in this competition.You can get the environment by tags on top.  On linux platform, you need to give the simulator executable permission

        chmod +x linux_v1/cog_sim2real_env.x86_64
        
Step 2. You can install the environment dependent using the command(you must have installed anaconda/miniconda first.):

        conda env create -f environment_*.yml.  

Step 3. Install the COG_API package with command:

        pip install CogEnvDecoder

Step 4. Run the api_test.py, you will see our simulation environment.  

## Important notes

The data from the simulator is clear, but it will be perturbed with biases and noises during the test stage. More information can be found in the [rulebook](https://github.com/DRL-CASIA/COG-sim2real-challenge/blob/main/CoG%20Challenge%20Rules-v1.2.pdf).

## Submission Guide
For submission, `cog_agent.py` must be included in the submitted files, and participants reconstruct `agent_cotrol()` function. When testing, `agent_control()` will be called to test the model. A simple test pipline can be found in `submit_test.py`. 

In addition, if the submitted code depends on additional packages, it is best to add `readme.md` files so that we can configure the environment easily.

               
## Thanks

The simulator is modified from the projects [Ausdroid RoboMaster Simulation](https://github.com/Webb-Bing/ARMS_RMUA2021_SImulation). 
We appreciate the open source efforts.

