#######################
### Frame Extractor ###
#######################

The frame extractor allows you to process *.avi files fully automated and extract individual frames from mice.
The frame extractor saves the extracted frames and further provides statistics and histograms.
Additionally, a ratings.csv and mappings.csv file is generated, which can be used to rate the mice on the MGS.
After an initial run, you can add further videos and restart the script. Only newly added content will be processed.

This includes:
- Converting the *.avi videos into *.mp4 videos using ffmpeg
- Extracting *.h5 files containing coordinates of the mouse and the grimace box using SLEAP networks
- Extracting individual frames from the videos

#####################
### Prerequisites ###
#####################

- Have a running SLEAP environment
- Have ffmpeg installed
- Have pretrained models for the mouse and the grimace box
- Have *.avi videos of a front facing and a top view camera
- Know the distance of the front of the grimace box

#####################
### How to use it ###
#####################

If you stick to the default parameters, you can run the script with almost no parameterization necessary.
This assumes that the front of the grimace box is 70mm wide.

1: Activate your conda environment in the terminal
> conda activate sleap_1.2.3

2: Copy the frame_extractor.py file into any folder and create two empty subfolders called input and output. Additionally,
copy the mouse model you want to use and the grimace box model into the same directory and rename the folders to mouse_model
and grimace_box_model. (Copy the whole folder and rename -> e.g: C:\Users\XYZ\Desktop\SLEAP3\Networks\Mouse\models\220707_102540.single_instance.n=257)
- XYZ (anything you want)
-- frame_extractor.py
-- input
-- output
-- mouse_model
-- grimace_box_model

3: Copy the *.avi files into the input folder. Make sure the names of the top and front view match except the
   part which determines the camera used. Use 'c1Grimace' for the front view and 'c2Grimace' for the top view.
   e.g: d0a79c1Grimace.avi and d0a79c2Grimace.avi (If you have other names, see Examples & Tips Section #6)

4: Navigate in the terminal with the active conda session to your main folder (XYZ in this example).

5: Run this command:
> python frame_extractor.py


###################################
### How to customize parameters ###
###################################

The Frame Extractor allows for customization of many parameters (e.g. File name patterns, front of the grimace box width, minimal frame distance, etc.)
You can either change these parameters directly in the python script by changing the default values of the specific arguments
of the parser at the bottom of the script. Or you can specify them using command line arguments (recommended!).

A list of the available parameters can be found by executing the follwing command:
> python frame_extractor.py --help


#######################
### Examples & Tips ###
#######################

1: The Frames Extractor only processes videos that were not previously processed. If you want to rerun the extraction you either have to
   delete the content of the output folder or use the --force_clean_output_directory or -f flag. Be careful if you already filled in the ratings.csv,
   since this file would then be deleted!
> python frame_extractor.py -f

2: You can also initially run the Frames Extractor with the --statistics_only or -s flag enabled. This gives you some information about how many frames
   would be extracted with the current settings and allows you to finetune your parameters, without acutally extracting the frames. Once you are happy
   with the result, you can run the extractor without the flag and extract the images.
> python frame_extractor.py -s

3: You can also tell the Frames Extractor to extract the frames from the unconverted *.avi videos instead of the converted *.mp4 videos using the
   --extract_from_original or -orig flag
> python frame_extractor.py -orig

4: You can change parameters determining the minimum likelihood of certain points, the allowed deviation of the parallelism of the line connecting both
   ears of the mouse and the front of the grimace box, the width of the grimace box, and so on... (full list -> python frame_extractor.py --help)
> python frame_extractor.py --norm_dist_mm=80.0 --min_pt_predict_likelihood=0.6 --max_allowed_parallel_deviation=0.2

5: You can generate debug plots that show the top view annotated with coordinates of landmarks of the mouse and the associated front view frame.
> python frame_extractor.py --debug_plots

6: Use other names for your file patterns (e.g. for files: day1_mouse1_camera_front.avi and day1_mouse1_camera_top.avi):
> python frame_extractor.py --front_camera_file_pattern="camera_front" --top_camera_file_pattern="camera_top"
