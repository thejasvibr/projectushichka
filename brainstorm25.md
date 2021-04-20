# Brainstorm 25 session + meeting with Pranav Khandelwal: 2021-03-31


The challenging thing about the videos for me is to keep track of bats that fly in and out 
of the view all the time. 

## Brainstorm25:

* Hemal:
	* the method can be broken down into a system which first detects the bats in each video stream, and then using epipolar projection matches each 2D 
detection to corresponding points on other cameras. 
	* Qualisys? -  a 3D tracking system used in the barn 
	* Detectron2 ? 
* Mathias: 
	* Suite2P package which is used for cell tracking in neuroscience
* ? : T-Rex (Tristan's software)

## Chat with Pranav Khandelwal:

* Not entirely sure if DLTdv8 can handle dynamic dis/appearance of points in a video
* He knows that tracking of a point *in* the view happens well - but the issue I have is that bats dis/appear in/out of the video 
very quickly, and doing it all manually gets relatively cumbersome. 
* Suggested DeepLabCut  - it can handle 2 camera 3d tracking. One potetntial issue is that it expects a checker-board pattern as a calibration object
	* The >2 camera version of DeepLabCut is AniPose, which handles multi-camera 3D pose estimation - however this too expects board pattern calibration workflows. 


## Summary

* Deep-learning based methods may detect objects out of the box with minimal/no training. 
* Stereo-matching or some kind of epipolar projection method will need to be used to match the 2D tracklets across the cameras
* The overall workflow would be:
	1. Run algorithms to detect the bats across the three cameras.
	1. 'Join' 2D points into tracklets
	1.  Match tracklets between cameras
	1. Generate the 3d positions for each tracklet

	



	
