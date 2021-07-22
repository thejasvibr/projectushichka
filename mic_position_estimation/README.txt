Ushichka data for mic self-positioning
======================================
Last updated: 2021-07-22



Audio data
----------
The set of folders contains multi-channel audio of the 'speaker playbacks', which typically consist
of 1-2 minute long recordings of speaker playing a signal repeatedly as it was moved in the recording 
chamber. 

Mic-to-mic measurements
-----------------------
The mic-to-mic measurements were performed with a laser range finder with 1 mm accuracy. However, in reality, 
the 'pointing' error is likely to be upto +/- 2 cm. I would expect that the measurement error is somehow directly
proportional to the distance, as with larger distance the literal jitter introduced by shaking hands and lowered
pointing accuracy must be considered. 



Folder structure and contents
-----------------------------

Each experimental session has the following structure and contents:
	* YYYY-MM-DD

		* mic2mic_measurements:
			* YYYY-MM-DD_field_notes.txt : text file with expected measurement error and any other observations 
			that might help with troubleshooting mic self-positioning
			* mic2mic_RAW_YYYY-MM-DD.csv : CSV file with a mic-to-mic distance matrix. 
			* YYYY-MM-DD_processing....py : .py file with code used to generate the final distance matrices

		* wav:
			* composite_speaker_playback_<POSIX-TIMESTAMP>.WAV : multichannel audio file 
			The 'composite_speaker_playback..' file will be there for all sessions. In addition there may be other
			files of interest too. 
			* digital_playbacksignal_<TIMESTAMP>.WAV : single channel audio file 
			On some experiment sessions, the actual digital signal that was fed to the speaker was recorded at the 
			same time. The 'digital_playbacksignal' file has this data. 
			* YYYY-MM-DD_alignment.py : .py file with code used to generate the final composite speaker playback file

		* playback_signals:
			* Folder with all the playback signals that are repeated. The silences are an important part of the signal 
			as they decide its periodicity!
			* The file 'session_wise_signalfiles.csv' maps the experimental session (the YYYY-MM-DD folders) to the playback signal used on that session. 
			* The playback signals are all derived from the 'https://github.com/thejasvibr/AV_calibration_playback.git' repo.
