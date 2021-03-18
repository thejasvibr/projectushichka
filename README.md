# Project Ushichka 

*Date of document initiation 2021-03-18*

This document and others in this repo have details and notes on how
I'm managing the documentation and progress of work on the Uschichka dataset. 
It is the first time I'll be handling so many files of so many different 
types (audio, video, LiDAR, derived data) and I'm realising it is better to 
invest in a well thought-out system before the growing complexity of the 
enterprise tumbles over itself. 


## Table of contents
1. [Data management](ushichka-management.md).
1. [Supporting data](supporting-datasets.md)

## What are the various types of data? 

The broad types of primary data are: 
1. Audio : multichannel audio files from the acoustic arrays. They also need
    to be processed to synchronise channels recorded across multiple soundcards. This leads to a derived data-form, the *cross-device synced audio*
    Both the raw and sync'ed multichannel audio are .WAV files sampled at 192 kHz
    from a mix of MEMs and condenser microphones. The output from analysing the
    multichannel audio files will be the acoustic tracking data.
1. Video : multi- camera recordings from three thermal cameras. Each camera 
    generates one file. The recordings are in the form of .TMC files that
    need to be exported onto .avi or other image formats. Video was recorded
    at 25Hz. The processed form of the video data will be .avi files. The output
    from analysing the video files from >=2 cameras will be the video tracking
    data. 
1. LiDAR : A LiDAR scan of the recording volume exists in multiple formats
    and resolutions. The details of the LiDAR scanning can be accessed [here](http://symp2018.geodesy-union.org/wp-content/uploads/2018/11/20.pdf)

## How the primary data types are linked, and technical challenges:
The three data types are inherently linked with each other, but will require
some technical effort in getting them aligned (spatially/temporally):

* Audio-Video: The audio and video recording were triggered simultaneously.
  Here there are some details and challenges to consider:
    1. The audio data lags a little bit behind the video data recording - they need to be *temporally aligned*
    1. The output from the audio and video tracking data needs to be *spatially aligned* to a common coordinate system. 
       The objects used to set a common  coordinate system could be either the microphone positions themselves (many nights have video recordings of mic positions), or
       the positions of one/multiple bats as captured in both tracking systems. 
* Video-LiDAR: The video and LiDAR recordings capture the physical structure of
the recording volume. Video recordings were made of the volume across multiple nights
with slight shifts in the positions of the cameras. The video and LiDAR scans need to be
*spatially aligned*. The LiDAR scan was performed on only one night, and that too with a big array in the scene. The 
details to consider are: 
    1. The big tristar array in the raw LiDAR scan may need to be 'erased' or, at least the version of the 
LiDAR scan without the array may help alignment.

## Other relevant data types
Along with the three main data types, we will also need a host of other 
supporting data that are required for audio-video tracking.

### Other raw data types
* Weather data: temperature, humidity, pressure data from a weather logger in the cave. 
* Inter-mic-distance data: measurements of some inter-mic distances taken with a laser range finder. These measurements help set mic self-positioning
algorithms (like in StructureFromSound). 
* Photographs: images of the cave, setup, notebook pages, etc. 

### Downstream processed data
There are also a host of  processed data  at various stages - all of which needs to 
be kept track of. How did the data come into place - what were the inputs?

* Spatial alignment data: the required data to bring acoustic, video and LiDAR data
positions into a common coordinate system. 
* Acoustic tracking associated: 
    1. Estimated microphone positions : output from StructureFromSound
    1. Trajectory data: bat position and call timing
* Video tracking associated: 
    1. Calibration associated data: wand calibrations, manually annotated wand positions, etc.
    1. Estimated camera locations: output from DLTdv
    1. Trajectory data: bat position
* Audio-video data:
    1. Trajectory information: bat ID, position and call timing




