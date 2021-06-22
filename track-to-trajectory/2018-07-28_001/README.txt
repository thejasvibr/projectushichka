This folder has the following data:

'object_detections':
--------------------
Two csv files with the following columns. For each csv file, one row corresponds to a bat detection in a frame of a camera video file.
The columns represent the following:

	* col_index: the 'x' position in pixels, with the leftmost pixel = 0 
	* row_index: the 'y' position in pixels, with the topmost pixel = 0
	* camera_id : one of K1, K2 or K3
	* frame_number: the frame number starting from 1
	* file_name: the raw image file from which objects were detected
	* tracking_parameters: the input parameters used to generate the object detections. 	

'tracked_videos':
-----------------

The stitched MP4 files for 2 recordings (2018-07-28_P00_35000 and 2018-07-28_P00_97000) have the 
3 camera views placed side by side. The order of the views is K3, K2, K1 (camera 3,2,1) from left to right. 

'video_calibration_data':
-------------------------

* The DLT coefficients and camera intrinsic parameters are stored in the 'round3_2018-07-28_dltCoefs.csv'. Each column is for one camera (K1, K2, K3 in that order)
For more details on what each row corresponds to see reference [1]

* The camera intrinsic parameters are in 'round3_cameraprofiles_profile.txt'. Each row is for one camera. 
The columns (left to right) stand for: 
	* camera number
	* focal length estimate (pixels)
	* image height (pixels)
	* image width (pixels)
	* principle point x (pixels)
	* principle point y (pixels)
	* primary-camera? (1/0) FOR yes/no. (Not relevant for most purposes)
	* r2 coefficient
	* r4 coefficient
	* tan 1 coefficient
	* tan 2 coefficient
	* r6 coefficient


References
----------
1. Kwon3D: Y. Kwon, Camera Calibration - DLT Method, 1998, http://www.kwon3d.com/theory/dlt/dlt.html