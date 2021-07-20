Ushichka data for mic self-positioning
======================================

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
