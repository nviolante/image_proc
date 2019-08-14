# Processamento dos dados do WaveScatter

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.close('all')

def calculate_paths_dists_vels(xy, ball, numframes, pxmm, fps):
    """
    Calculate paths, distances (mm) and velocities (mm/s) in a 2D dimension
    xy - position xy in time
    ball - string of ball name
    numframes - total number of frames
    distances in milimeters
    position in milimeters
    vels in milimters/second
    """

    # take a sample with initial and final frame number
    xy = xy[numframes[0]:numframes[1]]

    # position x and y for each ball
    x, y = np.array(xy[ball].tolist()).T

    # calculate distance between two consecutive frames
    dists_xy = [np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2) for i in range(len(x)-1)]
    dists_x = [x[i+1] - x[i] for i in range(len(x)-1)]
    dists_y = [y[i+1] - y[i] for i in range(len(y)-1)]

    # distance relative to x=x0
    dists_xy_t0 = [np.sqrt((x[i] - x[0])**2 + (y[i] - y[0])**2) for i in range(len(x)-1)]
    dists_x_t0 = [x[i] - x[0] for i in range(len(x)-1)]
    dists_y_t0 = [y[i] - y[0] for i in range(len(y)-1)]

    # derivada da distancia (velocity) for quality control of paths_xy
    dists_dif = pd.Series(np.diff(np.array([dists_xy[0] + dists_xy]))[0])

    # quality control

    # indicies dos arquivos qualificados
    index_qc = np.where((dists_dif<105) & (dists_dif>-105) & (pd.Series(dists_xy)>=0.0) & (pd.Series(dists_xy)<1000))[0]

    # distancias qualificadas
    dists_qc = dists_dif.loc[index_qc]

    # dictionary with distances and paths xy for qualified time series of paths and convert to mm
    paths_xy = np.array(xy[ball].tolist())[dists_qc.index] * pxmm

    dists_xy = pd.Series(dists_xy)[dists_qc.index] * pxmm
    dists_xy_t0 = pd.Series(dists_xy_t0)[dists_qc.index] * pxmm

    dists_x = pd.Series(dists_x)[dists_qc.index] * pxmm
    dists_x_t0 = pd.Series(dists_x_t0)[dists_qc.index] * pxmm

    dists_y = pd.Series(dists_y)[dists_qc.index] * pxmm
    dists_y_t0 = pd.Series(dists_y_t0)[dists_qc.index] * pxmm

    vels_xy = dists_xy / (1./fps)
    vels_x = dists_x / (1./fps)
    vels_y = dists_y / (1./fps)

    return paths_xy, dists_xy, dists_xy_t0, vels_xy, dists_x, dists_x_t0, vels_x, dists_y, dists_y_t0, vels_y

def calculate_mean_path_xy(paths_xy, fps):
    """
    Calculate the mean position x y - position x and y
    """

    x = []
    y = []
    for ball in paths_xy.keys():
        x.append(paths_xy[ball][:,0])
        y.append(paths_xy[ball][:,1])
    x = np.vstack(x).mean(axis=0)
    y = np.vstack(y).mean(axis=0)
    mean_path_xy = pd.DataFrame(np.array([x, y]).T)
    mean_path_xy.index = mean_path_xy.index / fps

    return mean_path_xy

def calculate_relative_dispersion(paths_xy, mean_path_xy):
    """
    Calculate the relative dispersion
    - For each ball
    - Mean of all balls
    """

    # teste
    # mean_path_xy[0] = np.arange(len(mean_path_xy))
    # mean_path_xy[1] = 1.0
    # paths_xy['ball_00'][:,0] = np.arange(len(mean_path_xy))    
    # paths_xy['ball_00'][:,1] = 4.0

    # mean square distance for each ball
    rel_disp = {}
    for ball in paths_xy.keys():
        print (ball)
        x, y = paths_xy[ball][:,[0,1]].T
        rel_disp[ball] = (np.sqrt((x - mean_path_xy.iloc[:,0])**2 + (y - mean_path_xy.iloc[:,1])**2))**2

    # mean distance for all balls
    mean_rel_disp = pd.DataFrame(np.vstack(rel_disp.values()).mean(axis=0), index=mean_path_xy.index)
    rel_disp = pd.DataFrame(rel_disp)

    return rel_disp, mean_rel_disp

def calculate_velocity_statistics(vels_xy):
    """
    Calculate mean velocity, max and standard deviation for all frames for all buoys
    input: velocities for all buoys during the video
    """

    # media da serie temporal de todas as bolas
    mean_vel_balls_xy = {}
    for ball in vels_xy.keys():
        mean_vel_balls_xy[ball] = vels_xy[ball].mean()
        print ('Ball {} - Mean: {}'.format(ball, mean_vel_balls_xy[ball]))

    # media das bolas
    mean_values = np.array([mean_vel_balls_xy[key] for key in mean_vel_balls_xy.keys()])

    # statistical values for balls velocities
    mean_vel_xy = np.mean(mean_values)
    std_vel_xy = np.std(mean_values)
    min_vel_xy = np.min(mean_values)
    max_vel_xy = np.max(mean_values)

    return mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy

def exponential_func(t, x):

    return t ** x

def adjust_fit_rel_disp(xdata, ydata):
    """
    Adjust an exponential fit
    """
    popt, pcov = curve_fit(exponential_func, xdata, ydata)
    yy = exponential_func(xdata, *popt)

    return popt[0], pcov, yy

def plot_paths_vels(filename, paths_xy, mean_path_xy, vels_xy):
    """
    Plot paths for each ball
    """

    fig1 = plt.figure(figsize=(6,3))
    ax1 = fig1.add_subplot(111)
    ax1.grid()
    ax1.set_xlabel('Position X (mm)')
    ax1.set_ylabel('Position Y (mm) \n Wavemaker')
    # ax1.set_title(filename)
    for ball in paths_xy.keys():
        ax1.plot(paths_xy[ball][:,0], paths_xy[ball][:,1]*-1,'b-', linewidth=0.4)
    ax1.plot(mean_path_xy.iloc[:,0], mean_path_xy.iloc[:,1]*-1, 'r')
    # fig1.savefig('img/paths{}.png'.format(filename[-17:]), bbox_inches='tight')

    return

def plot_distances(filename, dists_xy_t0, dists_x_t0, dists_y_t0):

    fig1 = plt.figure(figsize=(12,6))
    fig1.suptitle('Distance from t=t0 - {}'.format(filename))

    ax1 = fig1.add_subplot(131)
    ax1.grid()
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title('XY')
    ax1.plot(dists_xy_t0)
    ax1.set_ylim(0,2500)

    ax2 = fig1.add_subplot(132)
    ax2.grid()
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('X')
    ax2.plot(dists_x_t0)
    ax2.set_ylim(0,2500)

    ax3 = fig1.add_subplot(133)
    ax3.grid()
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Y')
    ax3.plot(dists_y_t0)
    ax3.set_ylim(0,2500)
    # fig1.savefig('img/dists{}.png'.format(filename[-17:]))

    return

def plot_distances_log(filename, dists_xy_t0, dists_x_t0, dists_y_t0):

    fig1 = plt.figure(figsize=(12,6))
    fig1.suptitle('Distance from t=t0 - {}'.format(filename))

    ax1 = fig1.add_subplot(131)
    ax1.grid()
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title('XY')
    ax1.loglog(dists_xy_t0)
    ax1.set_ylim(0,2500)

    ax2 = fig1.add_subplot(132)
    ax2.grid()
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('X')
    ax2.loglog(dists_x_t0)
    ax2.set_ylim(0,2500)

    ax3 = fig1.add_subplot(133)
    ax3.grid()
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Y')
    ax3.loglog(dists_y_t0)
    ax3.set_ylim(0,2500)
    # fig1.savefig('img/dists_loglog{}.png'.format(filename[-17:]))

    return

def plot_mean_distances(filename, dists_xy_t0):

    fig1 = plt.figure(figsize=(10,6))
    fig1.suptitle('Distance from t=t0 - {}'.format(filename))

    ax1 = fig1.add_subplot(121)
    # ax1.grid()
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Distance (mm)')
    # ax1.set_title('XY')
    ax1.loglog(dists_xy_t0)
    ax1.set_ylim(0,2500)

    ax2 = fig1.add_subplot(122)
    # ax2.grid()
    ax2.set_xlabel('Time (seconds)')
    # ax2.set_title('X')
    ax2.loglog(dists_x_t0.mean(axis=1))
    ax2.set_ylim(0,2500)
    # fig1.savefig('img/mean_dists{}.png'.format(filename[-17:]))

    return

def plot_rel_disp(filename, rel_disp, mean_rel_disp):

    fig1 = plt.figure(figsize=(10,6))
    fig1.suptitle('Relative Dispersion \n {}'.format(filename))
    ax1 = fig1.add_subplot(121)
    ax1.grid()
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('MSD')
    # ax1.set_title(filename)
    rel_disp.plot(loglog=True, ax=ax1, legend=None)

    ax2 = fig1.add_subplot(122)
    ax2.grid()
    ax2.set_xlabel('Time (seconds)')
    # ax2.set_ylabel('Mean Relative Dispersion')
    # ax2.set_title(filename)
    mean_rel_disp.plot(loglog=True, ax=ax2, legend=None)
    # fig1.savefig('img/rel_disp{}.png'.format(filename[-17:]))

    return

def plot_adjust_rel_disp(filename, mean_rel_disp, xdata, popt):

    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(filename)
    ax1.loglog(mean_rel_disp)
    ax2 = ax1.twinx()
    ax2.loglog(xdata, xdata ** popt, 'r')
    # fig1.savefig('img/fit_rel_disp{}.png'.format(filename[-17:]))
    return

def plot_adjust_dist_t0(filename, mean_dists_xy, xdata, popt):

    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(filename)
    ax1.loglog(mean_dists_xy)
    ax2 = ax1.twinx()
    ax2.loglog(xdata, xdata ** popt, 'r')
    # fig1.savefig('img/fit_dist_t0{}.png'.format(filename[-17:]))

    return



if __name__ == '__main__':

    # list of balls with errors
    balls_tracked_keys = {
                        # 'paths_T100_010300_CAM1': ['ball_02', 'ball_16', 'ball_04', 'ball_19', 'ball_17',
                        #                            'ball_00', 'ball_14', 'ball_18', 'ball_20', 'ball_06',
                        #                            'ball_07', 'ball_13', 'ball_08'],

                        # 'paths_T100_020100_CAM1': ['ball_20', 'ball_06', 'ball_16', 'ball_05', 'ball_01',
                        #                            'ball_03', 'ball_02', 'ball_04', 'ball_17', 'ball_18',
                        #                            'ball_25', 'ball_32', 'ball_07', 'ball_12', 'ball_15'],

                        # 'paths_T100_020201_CAM1': ['ball_04', 'ball_15', 'ball_02', 'ball_00', 'ball_08',
                        #                            'ball_29', 'ball_05', 'ball_14', 'ball_03', 'ball_16'],

                        # 'paths_T100_020300_CAM1': ['ball_14', 'ball_08', 'ball_21', 'ball_02', 'ball_12',
                        #                            'ball_16', 'ball_05', 'ball_09', 'ball_20', 'ball_06',
                        #                            'ball_00'],

                        # 'paths_T100_030100_CAM1': ['ball_27', 'ball_04', 'ball_00', 'ball_24', 'ball_31',
                        #                            'ball_20', 'ball_05', 'ball_10', 'ball_32', 'ball_25',
                        #                            'ball_30', 'ball_06', 'ball_22', 'ball_12', 'ball_17',
                        #                            'ball_16', 'ball_33', 'ball_13', 'ball_26', 'ball_14',
                        #                            'ball_03', 'ball_28', 'ball_07', 'ball_08', 'ball_02',
                        #                            'ball_01', 'ball_19'],

                        # 'paths_T100_030200_CAM1': ['ball_04', 'ball_05', 'ball_20', 'ball_22', 'ball_19',
                        #                            'ball_08', 'ball_32', 'ball_09', 'ball_29', 'ball_26',
                        #                            'ball_12', 'ball_10', 'ball_25', 'ball_23'],

                        # 'paths_T100_030300_CAM1': ['ball_14', 'ball_28', 'ball_22', 'ball_01', 'ball_07',
                        #                            'ball_28', 'ball_00', 'ball_20', 'ball_27', 'ball_29',
                        #                            'ball_16', 'ball_06', 'ball_17', 'ball_35', 'ball_09',
                        #                            'ball_38'],

                        # 'paths_T100_040100_CAM1': ['ball_08', 'ball_01', 'ball_16', 'ball_20', 'ball_10',
                        #                            'ball_21', 'ball_00', 'ball_13', 'ball_12', 'ball_22',
                        #                            'ball_32', 'ball_06', 'ball_28', 'ball_34', 'ball_17',
                        #                            'ball_23', 'ball_29', 'ball_18', 'ball_03', 'ball_07'],

                        'paths_T100_040300_CAM1': ['ball_26', 'ball_00', 'ball_01', 'ball_13',
                                                   'ball_22', 'ball_27', 'ball_31', 'ball_17', 'ball_14',
                                                   'ball_30', 'ball_23', 'ball_33', 'ball_07', 'ball_29',
                                                   'ball_21', 'ball_09', 'ball_05', 'ball_19'],

                        # 'paths_T100_050100_CAM1': ['ball_29', 'ball_04', 'ball_01', 'ball_00', 'ball_05',
                        #                            'ball_15', 'ball_19', 'ball_12', 'ball_10', 'ball_13',
                        #                            'ball_03', 'ball_18', 'ball_32', 'ball_33', 'ball_24'],

                        # 'paths_T100_050200_CAM1': ['ball_34', 'ball_28', 'ball_05', 'ball_00', 'ball_10',
                        #                            'ball_14', 'ball_11', 'ball_13', 'ball_01', 'ball_08',
                        #                            'ball_12', 'ball_33', 'ball_36', 'ball_26', 'ball_16',
                        #                            'ball_22', 'ball_37', 'ball_21', 'ball_29', 'ball_04',
                        #                            'ball_02'],

                        # 'paths_T100_050300_CAM1': ['ball_02', 'ball_00', 'ball_26', 'ball_21''ball_07',
                        #                            'ball_34', 'ball_29', 'ball_35']
                        }

    # initial and final frame (number of frames)
    numframes_keys = {
                      # 'paths_T100_010300_CAM1': [60, 1300],
                      # 'paths_T100_020100_CAM1': [440, 1513],
                      # 'paths_T100_020201_CAM1': [250, 1553],
                      # 'paths_T100_020300_CAM1': [520, 1500],
                      # 'paths_T100_030100_CAM1': [120, 1283],
                      # 'paths_T100_030200_CAM1': [270, 1725],
                      # 'paths_T100_030300_CAM1': [313, 1500],
                      # 'paths_T100_040100_CAM1': [440, 2900],
                      'paths_T100_040300_CAM1': [210, 2400], # [475, 2400]
                      # 'paths_T100_050100_CAM1': [480, 1893],
                      # 'paths_T100_050200_CAM1': [630, 1480],
                      # 'paths_T100_050300_CAM1': [630, 2100]
                      }

    # initial and final time for adjustment (seconds)
    adjust_keys = {
                   # 'paths_T100_010300_CAM1': [7, 16],
                   # 'paths_T100_020100_CAM1': [10, 35],
                   # 'paths_T100_020201_CAM1': [3, 31],
                   # 'paths_T100_020300_CAM1': [10, 22],
                   # 'paths_T100_030100_CAM1': [11, 24],
                   # 'paths_T100_030200_CAM1': [10, 40],
                   # 'paths_T100_030300_CAM1': [10, 33],
                   # 'paths_T100_040100_CAM1': [11, 32],
                   'paths_T100_040300_CAM1': [10, 31],
                   # 'paths_T100_050100_CAM1': [17, 21],
                   # 'paths_T100_050200_CAM1': [2, 10], #X
                   # 'paths_T100_050300_CAM1': [14, 25]
                   }

    # valor de 1 pixel em milimetros (1px = 1.8 mm)
    pxmm = 1.48

    # frames per second
    fps = 30.0

    for filename in list(balls_tracked_keys.keys()):

        print (filename)

        # path and filename
        pathname = os.environ['HOME'] + '/gdrive/coppe/lioc/wavescatter/out/{}.pkl'.format(filename)
        # pathname = 'data/qc/paths_qc_T100_040300_CAM1.pkl'

        # get values from a dict keys
        balls_tracked = balls_tracked_keys[filename]
        numframes = numframes_keys[filename]

        # read pickle with xy position
        xy = pd.read_pickle(pathname)

        # list with balls
        balls = list(xy.keys())

        # variables create inside a function
        paths_xy = {}
        dists_xy = {}
        dists_xy_t0 = {}
        vels_xy = {}
        dists_x = {}
        dists_x_t0 = {}
        vels_x = {}
        dists_y = {}
        dists_y_t0 = {}
        vels_y = {}

        # calculate path for each ball (loop for each ball)
        for ball in balls:
            if ball in balls_tracked:
                # print ('{}..Processed'.format(ball))
                paths_xy[ball], dists_xy[ball], dists_xy_t0[ball], vels_xy[ball], \
                dists_x[ball], dists_x_t0[ball], vels_x[ball], \
                dists_y[ball], dists_y_t0[ball], vels_y[ball] = calculate_paths_dists_vels(xy, ball, numframes, pxmm, fps)
            else:
                # print ('{}..Error'.format(ball))
                pass

        # create dataframes with times as index
        dists_xy = pd.DataFrame(dists_xy)
        dists_x = pd.DataFrame(dists_x)
        dists_y = pd.DataFrame(dists_y)

        dists_xy_t0 = pd.DataFrame(dists_xy_t0)
        dists_x_t0 = pd.DataFrame(dists_x_t0)
        dists_y_t0 = pd.DataFrame(dists_y_t0)

        vels_xy = pd.DataFrame(vels_xy)
        vels_x = pd.DataFrame(vels_x)
        vels_y = pd.DataFrame(vels_y)

        dists_xy.index = dists_xy.index / fps
        dists_xy_t0.index = dists_xy_t0.index / fps
        vels_xy.index = vels_xy.index / fps
        dists_x.index = dists_x.index / fps
        dists_x_t0.index = dists_x_t0.index / fps
        vels_x.index = vels_x.index / fps
        dists_y.index = dists_y.index / fps
        dists_y_t0.index = dists_y_t0.index / fps
        vels_y.index = vels_y.index / fps

        # time vector
        times = np.array(dists_x.index)

        # calculate mean path
        mean_path_xy = calculate_mean_path_xy(paths_xy, fps)

        # calculate relative dispersion for each ball and mean
        rel_disp, mean_rel_disp = calculate_relative_dispersion(paths_xy, mean_path_xy)

        # D(t)
        # adjust fit for mean relative dispersion
        a, b = adjust_keys[filename]
        xdata = mean_rel_disp[a:b].index.values
        ydata = mean_rel_disp[a:b].values[:,0]
        popt_dt, pcov, yy = adjust_fit_rel_disp(xdata, ydata)
        print ('D(t): {:.1f}'.format(popt_dt))

        # M(t)
        a = 0.2
        xdata = dists_xy_t0[a:].index.values
        ydata = dists_xy_t0[a:].mean(axis=1)
        popt_mt, pcov, yy = adjust_fit_rel_disp(xdata, ydata)
        print ('M(t): {:.1f}'.format(popt_mt))

        # calculate statistics from velocity time series of each ball
        mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy = calculate_velocity_statistics(vels_xy)
        print ('Mean: {:.2f}, STD: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(mean_vel_xy, std_vel_xy, min_vel_xy, max_vel_xy))

        plot_adjust_rel_disp(filename, mean_rel_disp, xdata, popt_dt)
        plot_adjust_dist_t0(filename, dists_xy_t0.mean(axis=1), xdata, popt_mt)
        plot_paths_vels(filename, paths_xy, mean_path_xy, vels_xy)
        plot_distances(filename, dists_xy_t0, dists_x_t0, dists_y_t0)
        plot_mean_distances(filename, dists_xy_t0)
        plot_rel_disp(filename, rel_disp, mean_rel_disp)

        # calcula velocidade media
        dist_total = dists_xy_t0.mean(axis=1).iloc[-1]
        time_total = dists_xy_t0.index[-1]
        mean_vel_total = (dist_total / time_total) / 1000.0 # em metros
        print ('Velocidade media total (m/s): {:.3f}'.format(mean_vel_total))

        # create paths_xy qualified
        paths_xy_qc = {}
        for ball in paths_xy.keys():
            paths_xy_qc[ball] = paths_xy[ball].tolist()
        paths_xy_qc = pd.DataFrame(paths_xy_qc)

        # save paths qualified
        # paths_xy_qc.to_csv('data/qc/paths{}.csv'.format(filename[-17:]))
        # paths_xy_qc.to_pickle('data/qc/paths{}.pkl'.format(filename[-17:]))
        # dists_xy.to_csv('data/qc/dists{}.csv'.format(filename[-17:]))
        # dists_xy_t0.to_csv('data/dists_xy_t0{}.csv'.format(filename[-17:]))
        # vels_xy.to_csv('data/qc/vels{}.csv'.format(filename[-17:]))

    plt.show()
