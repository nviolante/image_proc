"""
Acha circulos e cria caminho dos circulos encontrados
- o numero de circulos a serem rasterados sera a quantidade
de circulos encontrados no primeiro frame

Henrique Pereira
ONR
03/05/2018

with the arguments:

src_gray: Input image (grayscale)
circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
dp = 1: The inverse ratio of resolution
min_dist = src_gray.rows/8: Minimum distance between detected centers
param_1 = 200: Upper threshold for the internal Canny edge detector
param_2 = 100*: Threshold for center detection.
min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy import spatial

# cv2.destroyAllWindows()

def read_video(pathname, filename):
    cap = cv2.VideoCapture(pathname + filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def read_frame(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    frame = frame[250:-130,:]
    return frame

# def read_paths_xy(filename):
#     df_list = read_csv(filename)
#     df = np.vstack(df_list)
#     return df

def find_first_and_last_frames(dict_videos, filename, fps):
    # first and final time in datetime format
    dtime = pd.to_datetime(dict_videos[filename_video], format='%M:%S')
    # time in timedelta (to convert to total_seconds)
    timei = dtime[0] - pd.Timestamp('1900')
    timef = dtime[1] - pd.Timestamp('1900')
    # video duration in time_delta format
    dur = dtime[1] - dtime[0]
    # video duration in seconds
    durs = dur.total_seconds()
    # number of first and last frames to be reaed (based of fps)
    nframei = int(timei.total_seconds() * fps)
    nframef = int(timef.total_seconds() * fps)
    return nframei, nframef

def frames_preproc(frame1, frame2):

    # convert frames to gray scale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # smooth the frames
    frame1 = cv2.GaussianBlur(gray1,(5,5),0)
    frame2 = cv2.GaussianBlur(gray2,(5,5),0)

    # remove the background using threshold
    ret, frame1 = cv2.threshold(frame1,70,255,cv2.THRESH_BINARY)
    ret, frame2 = cv2.threshold(frame2,70,255,cv2.THRESH_BINARY)

    return gray1, gray2, frame1, frame2

def find_circles(frame):
    circles = cv2.HoughCircles(image=frame,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=40,
                               param1=10,
                               param2=5,
                               minRadius=10,
                               maxRadius=20)
    return circles

def find_initial_balls(circles1):
    # list of circles to be tracked
    balls_xy = circles1[0,:,:2]
    for ball_id in range(len(balls_xy)):
        paths['ball_{ball_id}'.format(ball_id=str(ball_id).zfill(2))]  = [list(balls_xy[ball_id].astype(int))]
    return paths

def track_min_dist(paths, ball, circles2):
    """
    Track balls with minimum distance
    """
    # point x and y for one  ball for the frame_i
    pt = paths[ball][-1]
    # all point x and y for one ball for the frame_i+1
    A = circles2[0,:,:2]
    # calculates the distance and index
    distance, index = spatial.KDTree(A).query(pt)
    # print (distance, index)

    if distance > 30:
        # print (distance)
        # pass
        xy_ball = pt # pega o ponto anterior
    else:
        xy_ball = list(A[index].astype(int))

    return xy_ball

def make_gif():
    string = "ffmpeg -framerate 10 -i %*.png output.mp4"
    os.system(string)
    return


if __name__ == '__main__':

    # ------------------------------------------------------------------------ #
    # Dados de entrada

    # filename_video = 'T100_010300_CAM1.avi'

    track_balls = True
    plot_balls_paths = True
    save_balls_paths = False # save dict in csv and pck file
    read_balls_paths = False # read in pickle

    # dicionario com nome do video e tempo inicial e final em que as
    # bolinhas estao dentro da tela e ja se separaram (quando em cluster)
    dict_videos = {
                   # 'T100_010100.CAM1.avi': ['00:04','01:52'],
                   # 'T100_010100.CAM1_ISOPOR.avi': ['00:11','01:35'],
                   # 'T100_010100_CAM1.avi': ['00:12','01:58'],
                   # 'T100_010200_CAM1.avi': ['00:12','01:56'],
                   # 'T100_010300_CAM1.avi': ['00:15','01:37'], # ok
                   # 'T100_020100_CAM1.avi': ['00:00','01:30'],
                   # 'T100_020200_CAM1.avi': ['00:00','01:42'],
                   # 'T100_020201_CAM1.avi': ['00:08','02:30'],
                   # 'T100_020300_CAM1.avi': ['00:00','02:10'],
                   # 'T100_030100_CAM1.avi': ['00:12','01:50'],
                   # 'T100_030200_CAM1.avi': ['00:06','02:25'],
                   # 'T100_030300_CAM1.avi': ['00:07','01:46'],
                   # 'T100_040100_CAM1.avi': ['00:04','03:35'],
                   # 'T100_040200_CAM1.avi': ['00:00','03:30'],
                   'T100_040300_CAM1.avi': ['00:03','02:38'],
                   # 'T100_050100_CAM1.avi': ['00:03','01:50'],
                   # 'T100_050200_CAM1.avi': ['00:05','01:26'],
                   # 'T100_050300_CAM1.avi': ['00:04','01:52'],
                   }

    # pathname_video = '/media/lioc/GODA/wavescatter_videos/DERIVA_RANDOMICA/VIDEO/CAM1/T100/'
    pathname_video = os.environ['HOME'] + '/Documents/wavescatter_data/'

    for filename_video in dict_videos.keys():

        # pathname_fig_output = os.environ['HOME'] + '/Documents/wavescatter_results/{}/'.format(filename_video[:-4])

        # create directory for fig outuput
        # os.system('mkdir {}'.format(pathname_fig_output))

        # ------------------------------------------------------------------------ #
        # Start program

        if track_balls == True:

            cap, fps = read_video(pathname=pathname_video, filename=filename_video)
            nframei, nframef = find_first_and_last_frames(dict_videos, filename_video, fps)

            cont_frame = 0
            paths = {}
            for ff in range(nframei, nframef, 1):

                print ('frame {ff} de {nframef}'.format(ff=ff, nframef=nframef))

                # contador de bolas
                cont_frame += 1

                # read two consecutives frames
                frame1 = read_frame(cap, ff)
                frame2 = read_frame(cap, ff+1)

                # save a copy of original frames
                output1 = frame1.copy()
                output2 = frame2.copy()

                # preprocessing the frames to detect circles
                gray1, gray2, frame1, frame2 = frames_preproc(frame1, frame2)

                # cv2.imwrite('img/img_raw.png',output1)
                # cv2.imwrite('img/img_gray.png',gray1)
                # cv2.imwrite('img/img_proc.png',frame1)

                # list of circles (balls)
                circles1 = find_circles(frame1)
                circles2 = find_circles(frame2)

                    # print (len(circles1))

                    # only the circles identified in the first frame will be tracked along the video
                if cont_frame == 1:
                    paths = find_initial_balls(circles1)

                # condicao para parar quando nao tiver mais bola na imagem
                if circles2 is not None:

                    # plot figure to be saved
                    if plot_balls_paths == True:
                        plt.figure(figsize=(10,10))
                        plt.imshow(output1)
                        plt.tight_layout()

                    # loop to track each ball
                    for ball in paths.keys():
                        # xy position of each ball
                        xy_ball = track_min_dist(paths, ball, circles2)
                        # create a list with paths for each ball
                        paths[ball].append(xy_ball)

                        # plot figure with paths (plot for each frame all balls)
                        if plot_balls_paths == True:
                            a = np.array(paths[ball])
                            plt.plot(a[:,0], a[:,1],'-', linewidth=2.5)
                            plt.text(a[:,0][-1], a[:,1][-1], ball[-2:], color='w')

                    # loop to convert list of xy to array (to save csv with dataframe)
                    paths_xy = {}
                    for path_key in paths.keys():
                        paths_xy[path_key] = np.vstack(paths[path_key])

                    # save the figure in png
                    if plot_balls_paths == True:
                        plt.savefig(os.environ['HOME'] + '/Documents/wavescatter_data/frames/frame_{cont_frame}'.format(cont_frame=str(cont_frame).zfill(4)),
                        bbox_inches='tight')
                        plt.close('all')

            # realease the video obj
            cap.release()

        # save path with xy for each ball
        if save_balls_paths:
            df = pd.DataFrame(paths)
            df.index.name = 'nframe'
            df.to_csv('data/paths_%s.csv' %filename_video[:-4])
            df.to_pickle('data/paths_%s.pkl' %filename_video[:-4])

        if read_balls_paths:
            d = pd.read_pickle('data/paths_%s.pkl' %filename_video[:-4])

        #if make_video_png:
        #    string = "ffmpeg -framerate 10 -i %*.png output.mp4"
        #    os.system(string)
