## Background
This is a package for COG participants to get necessary image and vector data to train and test their models.
##Install
pip install CogEnvDecoder
##Usages
The code is able to get observations generated from ml-agents in the COG stimulation environment.
With the code, you can get:
- Blue one rover
	- Image generated from virtual camera setted up on Blue one
	- Blue one rover position & angle
	- Enemy rover position & angle
	- Remaining HP & bullets
	- Goals' position
- Judgement system information
	- Current score
	- Number of goals have been achieved
	- Whether enemy rover has been activated
	- Enemy rover remaining HP & bullets
	- Blue one caused damage
	- Time passed in the round
	- collision information