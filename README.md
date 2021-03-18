# Project Ushichka 

This document and others in this repo have details and notes on how
I'm managing the documentation and progress of work on the Uschichka dataset. 
It is the first time I'll be handling so many files of so many different 
types (audio, video, LiDAR, derived data) and I'm realising it is better to 
invest in a well thought-out system before the growing complexity of the 
enterprise tumbles over itself. 

## What are the various types of data? 

The broad types of data are: 
1. Audio : multichannel audio files from the acoustic arrays. They also need
    to be processed to synchronise channels recorded across multiple soundcards. This leads to a derived data-form, the *cross-device synced audio*
    Both the raw and sync'ed multichannel audio are .WAV files sampled at 192 kHz
    from a mix of MEMs and condenser microphones.
1. Video : multi- camera recordings from three thermal cameras. Each camera 
    generates one file. The recordings are in the form of .TMC files that
    need to be exported onto .avi or other image formats. Video was recorded
    at 25Hz. 
1. LiDAR : A LiDAR scan of the recording volume exists in multiple formats
    and resolutions. The details of the LiDAR scanning can be accessed [here](http://symp2018.geodesy-union.org/wp-content/uploads/2018/11/20.pdf)

The three raw data types are linked with each other in the following manner:

