# Building 2D bat tracks

The 2d tracks of bats need to be generated from each of the video channels. This document will track my progress on 
the 2D tracking front. 

In general the code I'll use needs to implement the following tasks:

1. Bat detection over multiple frames
1. Track generation (assigning the same object ID to an object over multiple frames)

I'm assuming Ben's code (See Updates) does both right now - I know that it does bat detection at least.










## Updates:

* 2021-06-18 : Ben Koger shared code used to track fruit bat blobs in the Kasanka flights. A bunch of interesting 
measurements can be made with the code - blob size, bounding rectangles - from which wing flapping data can also be 
extracted. 