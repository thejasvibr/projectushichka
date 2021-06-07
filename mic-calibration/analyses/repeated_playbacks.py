"""
Module to perform repeated playbacks to see if playback level is constant. 

"""
import random
import time 
import datetime as dt

from playback_code.microphone_calib_playbacks import make_playback_sounds, perform_playback

pbk_sounds, _ = make_playback_sounds()
mic_num = 'gras'
angle = '0' # +ve elevation angles --> sound from top
gain = '46'
#orientation= 'elevation'
orientation= 'azimuth'


timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
mic_rec_name = mic_num+'_'+'gaindB_'+gain+'_'+orientation+'_angle_'+angle+'_'+timestamp+'.wav'
print('playback starting....')
perform_playback(pbk_sounds, mic_rec_name, save_path='./')
print('playback done...')
