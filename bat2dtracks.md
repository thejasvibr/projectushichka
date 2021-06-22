# Building 2D bat tracks

The 2d tracks of bats need to be generated from each of the video channels. This document will track my progress on 
the 2D tracking front. 

In general the code I'll use needs to implement the following tasks:

1. Bat detection over multiple frames
1. Track generation (assigning the same object ID to an object over multiple frames) -- this can be done with Giray's code 
for now. 

I'm assuming Ben's code (See Updates) does both right now - I know that it does bat detection at least.

## The plan 
Right now I've decided to use 2018-07-28 evening (as this was the hard-disk I had with me while starting to test). 
I can use the video data to send to Giray to test his trajectory assignment software. 

* Recording K1,2,3 P000/35000.TMC has 376 frames and is busy and ideal for a 2-3 bat test scenario
* Recording K1,2,3 P000/97000.TMC has 376 frames and is busy and ideal for a multi-bat test scenario. 

For the matching audio files see [here](audio-video-pairs.md).








## Updates:

* 2021-06-18 : Ben Koger shared code used to track fruit bat blobs in the Kasanka flights. A bunch of interesting 
measurements can be made with the code - blob size, bounding rectangles - from which wing flapping data can also be 
extracted. 