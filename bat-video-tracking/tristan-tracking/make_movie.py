# Thanks https://stackoverflow.com/a/62434934
import glob 
import natsort
import os
import moviepy.video.io.ImageSequenceClip
image_folder='./'
fps=25
starting_filename = 'K3_frame'
image_files = [os.path.join(image_folder,img) for img in natsort.natsorted(glob.glob(image_folder+f'{starting_filename}*.png'))]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(f'{starting_filename}_video.mp4')