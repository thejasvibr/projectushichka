# Project progress
This document will document progress made on various fronts, including some associated research work. 

The work never ends: here's the [TODO list](TODO.md)

## Week 1: 15-19/3

* Setting up this repository
* Decision to implement all data into a relational database (sqlite,mysql,etc) for data safety and consistency
* Initiation of work on inferring call direction using beam-shape models using approximate Bayesian computing. Ideas for a paper

## Week 2:  22-26/3 

* Cluster retreat (9-12am) for the whole week 
* Gave talk introducing the Ushichka dataset on 25/3
* Further work on inferring call direction using beam-shape models 
* Started inspection of the dataset/files that are spread across 4 smaller hard drives
	* Looking at redundancies between the 4 drives: in ```ushichka_docs/What is in the Ushichka dataset.ipynb``` notebook
* Looking up computers + 6 TB hard drives to be ordered

## Week 3: 29/3 - 01/4

* Decision to drop the idea of using a database management system as the learning curve may not be worth it. 
* 6TB hard drive delivered
* All unique audio and video raw data files listed: 2095 WAV files, 8163 TMC files. All copies of the WAV and TMC files spread on the 4 harddisks are identical
(same MD5 hash)
* Brainstorm25 meeting + chat with Pranav Khandelwal to see how to track bats in videos. See [Notes](./brainstorm25.md)
* Meeting with Iain, agreed on developing methods mainly for single/few bat audio+video data. Will focus on methods development and doing beamshape modelling. 

## Week 4: 6/4-9/4

* Placed order for PC and laptop
* Raised [issue on Stack Overflow ](https://stackoverflow.com/q/66934803/4955732) to see if there's a way to speed up the SymPy code which recreates the 
beam-shapes for various models. Got a reply, it seems like working at such high decimal precision is the main cause - there doesn't seem to be a way around, at least with SymPy
* Wrote to specialists in sound-field modelling to ask what software/packages they use to run their code. Those who replied use Mathematica or MATLAB.
	* I however, am starting to learn FriCAS as it promises to be a powerful language/framework to handle multiple acoustic models. It's able to 
	handle a larger variety of integrations than most packages out there, including Mathematica, and this is a big plus. The main disadvantage however is
	that it has a somewhat steep learning curve, and variable types play a really important role -- needs some getting used to. 

* Started microphone calibrations - Grossflugraum cleaned and basic equipment brought together.

## Week 5: 12/4 - 16/4

* Start proper implementation of piston in sphere in FriCAS. Raised issue on the emailing-list. 
* Chat with potential Master's student to work on automated trajectory matching. 
* Microphone calibrations started. 

## Week 6: 19/4 - 23/4

* Meeting scheduled with Bastian Goldluecke on 22/4
* Confirmatory meeting scheduled with Master's student
* Microphone calibration + troubleshooting underway 
* Succesful implementation of piston in a rigid sphere!!!
* Group meeting presentation on 23/4 - v cool ideas and suggestions (see email conversations)
* 2nd Meeting with Giray - Master's project confirmed

## Week 7: 26/4-30/4 

* Succesful implementation of oscillating cap of sphere. Some progress made optimising for speed too. 
* Analysing 3 mic calibration data- troubleshooting still going on. 
	* Building a pipeline and making sure things are as expected is taking a bit more time than I'd thought of!! (speaker also seems to show some variation?)
	
## Week 8: 3/5-7/5
* Spontaneous meeting with HRG about issues related to mic calbiration 
and where to take Ushichka. 
	* Focus on getting the pipeline working for ONE audio-video recording. Acoustic tracking, video tracking, video--> acoustic tracking etc. 
	* don't invest too much time documenting *everything*. It's a bit sad, but perform focussed data documentation -- focussing on
	single bat recordings. 
* contributed to Cluster grant w Michael Griesser, Hemal Naik, Kano Fumihiro, Chase Nunez for video tracking software development.
* Some more experiments to figure out whether the wall makes a difference etc. 
* Troubleshooting work on the piston in a sphere

## Week 8: 10/5 - 14/5

* Successfully troubleshot piston in a sphere + had a fast FLINT implementation 
	* there was a typo in the code + textbook itself -- leading to the apparently deviant output
* Implemented multiple beamshape models - piston in infinite baffle, point source on sphere, and speed optimisations
* Bachelor's student found to work on the LiDAR-thermal alignment. Meeting planned. 

## Week 9: 17/5 - 21/5
* Implementation of 3 all-round sound radiation models in *beamshapes*. Continuous integration implemented. Package seems to be ready for release
* LiDAR+thermal video data shared. Initial camera calibrations made for 2018-08-17
* 1st meeting with Julian+Bastian on aligning the thermal camera system and LiDAR scans. Data described and basic approaches discussed.

## Week 10: 24/5 - 28/5 
* Sent emails informing colleagues about `beamshapes`
* Some more microphone calibrations 
* Troubleshooting piston in sphere with `beamshapes`

## Week 11: 31/5 - 4/6
* raised question in Math Exchange for the piston in sphere question https://math.stackexchange.com/q/4156607/933933
* preparation for PhD defence 

## Week 12: 07/6 - 11/6
* Meeting with Giray on 7/6. He's implemented projection and re-projection workflows, and accuracy estiamtion. Next is to implement noise+ correspondence matching of points between cameras
* Preparations for PhD defence on 10/6

## Week 13: 14/6-18/6

* Clearing out Haus 10 
* Some more isues with playback variation in the speaker recordings -- is it mic level recording now?
* Visit to Konstanz to meet Iain and pick up computers that were ordered 
* Met Ben Koger, who agreed to share the 2D tracking code he used for the Kasanka bats - trying to install and run the code as is now. 

## Week 14: 21/6-25/6 

* First succesful 2D tracks of bats from 2018-07-28 P00/67000.TMC 1-75 frames. Thanks to the code Ben shared, the tracking happens fairly well. 
* Talk given at MPI-AB on 22 June 2021 10:30
* Shared bat detections with Giray for two videos 2018-07-28 P00/97000.TMC and 2018-07-28 P00/35000.TMC (frames 1-75 only)
* Giray progress - so far has established 3dtrajectory matching using a) known 2D trajectories across videos AND has also implemented
b) 3D trajectory matching with only points and no 2D traj labels!
* Acoustic tracking shown to work for speaker playbacks 2018-07-28/P02/02000.TMC and '...multichirp_2018-07-29_09-42-59' audio. 


