# An Algorithm for Tracking Drifters Dispersion Induced by Wave Turbulence
# using Optical Cameras

Scripts:

1. track_drifter.py - 9.7 kB

Description: Track drifters released in a wave tank.
Input: video file with initial and final time
Output: paths with x and y position

Functions:
    - read_video
    - read_frame
    - find_first_and_last_frames
    - frames_preproc
    - find_circles
    - find_initial_balls
    - track_min_dist
    - make_gif

2. proc_drifter.py - 19.3 kB

Description: Qualify the paths and calculate the mean
             and relative dispersion of the drifters.
Input: dictionary with path file and valid balls
Output: qualified paths

Functions:
    - calculate_paths_dists_vels
    - calculate_mean_path_xy
    - calculate_relative_dispersion
    - calculate_velocity_statistics
    - exponential_func
    - adjust_fit_rel_disp
    - plot_paths_vels
    - plot_distances
    - plot_distances_log
    - plot_mean_distances
    - plot_rel_disp
    - plot_adjust_rel_disp
    - plot_adjust_dist_t0

Developers:

- Henrique P. P. Pereira
- Nelson Violante-Carvalho
- Ricardo Fabbri
- Alex Babanin
- Uggo Pinho
- Alex Skvortsov

Concatct Adress:

Adress: Ocean Engineering Program, Rio de Janeiro Federal University, Brazil
Telephone: 

Hardware required:

- Ubuntu 18.04 LTS

Language:

- Python 3.6 with libraries:
- Libraries
    - os
    - numpy
    - matplotlib
    - cv2
    - pandas
    - scipy