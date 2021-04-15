# Multi-camera trajectory matching
* Date of document initiation: 2021-04-14*

*Related post*: [brainstorm25](brainstorm25.md)

The Ushichka dataset has 3 video cameras that were sync'ed and recording videos every trigger. 
To generate 3D trajectories, two problems need to be solved : 1) 2D detection of bat trajectories in all videos
and 2) correspondence matching of the 2D trajectories into one 3D trajectory. As of 13/4, Hemal has perhaps found 
a potential Master's student who may be working on the problem of matching 2D tracks across cameras into 3D trajectories. 

The idea is to use the already collected data using the Vicon system for ground-truthing, and then applying it to various
other datasets. 

## Data of potential interest
As of now, I can only think of the wand data that has already been digitised + the ground-truth xyz data from 
the DLTdv package. Admittedly the wand is only 2 points to be tracked, though this should be enough to test
if the routines work or not. 

In Disk #1 (THEJASVI_DATA_BACKUP), ```fieldwork_2018_001/actrackdata/video/2018-07-25/wand``` , I see the 
video : P03_1000TMC has been processed during the field season. The XY points, along with the easyWand outputs are 
all there - which emans this will be a good dataset to send for initial checking. 

The data has been uploaded at: https://owncloud.gwdg.de/index.php/s/k19sgwvcSPcvHEY



 
 



