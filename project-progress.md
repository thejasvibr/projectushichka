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
