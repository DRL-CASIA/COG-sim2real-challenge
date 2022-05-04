## COG-baseline-track1
This repository is related to the 2022 CoG Robomaster Sim2Real Challenge which is organized by CASIA DRL Team.In this repository, we will provide the competiton environment for you, and you can also find the Instructions of the necessary environment API in api_test. py.            

We recommend using conda to create a virtual environment.  

How to use the code we provided?  

Step 1.We provide two environments for you to use different os(Windows/Linux) to paticipate in this competition.You can get the environment by tags on top.  On linux platform, you need to give the simulator executable permission

        chmod +x linux_v1/og_sim2real_env.x86_64
        
Step 2. You can install the environment dependent using the command(you must have installed anaconda/miniconda first.):

        conda env create -f environment_*.yml.  

Step 3. Install the COG_API package with command:

        pip install CogEnvDecoder==0.1.28

Step 4. Run the api_test.py, you will see our simulation environment.  
               
Special Note：this is our first demo environment, we will update some new version according to the performance of contestants .  
        
When the game officially starts, we will provide the baseline algrithom on this repository.
