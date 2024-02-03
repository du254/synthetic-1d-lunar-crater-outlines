import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import fft
import os
import shutil
from matplotlib.colors import LightSource
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator


def create_synthetic_outlines(diameter_array):
    # ------------------------------------------------------------------------------------------------------------------
    num_vertices = 5000
    dimension = 1000
    extent_radius_ratio = 3.5
    flag_flat_floor_background = "yes"
    flag_bp_sigma = "yess"
    flag_avg_sigma = "yess"
    colors = [(0.4, 0.4, 0.4), (0.65, 0.65, 0.65)]
    cmap = LinearSegmentedColormap.from_list('GrayGradient', colors, N=256)
    color_feature_array = ['#fc8d59', '#91cf60', '#67a9cf', 'black']
    # id_temp="(f)"
    id_x = 0.72
    id_y = 0.32055
    title_x = 0.5
    title_y = 0.95
    # ------------------------------------------------------------------------------------------------------------------
    coef_sigma_rim_crest_distance = pd.read_csv("coef_sigma/rim_crest_distance.txt", delimiter=',')
    coef_sigma_rim_crest_elevation = pd.read_csv("coef_sigma/rim_crest_elevation.txt", delimiter=',')
    coef_sigma_floor_distance = pd.read_csv("coef_sigma/floor_distance.txt", delimiter=',')
    coef_sigma_floor_elevation = pd.read_csv("coef_sigma/floor_elevation.txt", delimiter=',')
    coef_sigma_rim_flank_distance = pd.read_csv("coef_sigma/rim_flank_distance.txt", delimiter=',')
    coef_sigma_rim_flank_elevation = pd.read_csv("coef_sigma/rim_flank_elevation.txt", delimiter=',')
    std_dev_array = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    breakpoint_diameter_in_fit = 20
    # ------------------------------------------------------------------------------------------------------------------
    header = "arc_angle_radian,radial_distance_km,elevation_km"
    pathout_txt_outline = "outline/"
    if os.path.exists(pathout_txt_outline):
        shutil.rmtree(pathout_txt_outline)
    os.makedirs(pathout_txt_outline)
    pathout_fig_outline = "figure/"
    if os.path.exists(pathout_fig_outline):
        shutil.rmtree(pathout_fig_outline)
    os.makedirs(pathout_fig_outline)
    # ------------------------------------------------------------------------------------------------------------------
    for idx in range(len(diameter_array)):
        diameter_temp = diameter_array[idx]
        # --------------------------------------------------------------------------------------------------------------
        rim_crest_radius_temp = diameter_temp / 2
        rim_crest_elevation_temp = Get_rim_height(diameter_temp, breakpoint_diameter_in_fit, flag_avg_sigma)
        floor_radius_temp = Get_floor_radius(diameter_temp, flag_avg_sigma)
        rim_to_floor_depth_temp = Get_crater_depth(diameter_temp, breakpoint_diameter_in_fit, flag_avg_sigma)
        rim_flank_radius_temp = Get_rim_flank_radius(diameter_temp, flag_avg_sigma)
        # --------------------------------------------------------------------------------------------------------------
        rim_crest_distance_avg = rim_crest_radius_temp
        rim_crest_elevation_avg = rim_crest_elevation_temp
        floor_distance_avg = floor_radius_temp
        floor_elevation_avg = rim_crest_elevation_temp - rim_to_floor_depth_temp
        rim_flank_distance_avg = rim_flank_radius_temp
        rim_flank_elevation_avg = 0
        # --------------------------------------------------------------------------------------------------------------
        control_points = Get_Breakpoints_slope(diameter_temp, breakpoint_diameter_in_fit, coef_sigma_rim_crest_distance,
                                               flag_bp_sigma)
        psd_target = CalculateTargetPSDFromBreakpointSlope(diameter_temp, control_points, num_vertices)
        psd_target_noise = Add_noise_to_array(psd_target, std_dev_array[0])
        x_coord_rim_crest, y_coord_rim_crest, arc_angle, rim_crest_radial_distance, psd_model = Create_outline(
            diameter_temp,
            num_vertices, rim_crest_distance_avg, psd_target_noise)
        # -------------------------------------------------------------------
        control_points = Get_Breakpoints_slope(diameter_temp, breakpoint_diameter_in_fit,
                                               coef_sigma_rim_crest_elevation,
                                               flag_bp_sigma)
        psd_target = CalculateTargetPSDFromBreakpointSlope(diameter_temp, control_points, num_vertices)
        psd_target_noise = Add_noise_to_array(psd_target, std_dev_array[1])
        x_coord_ele, y_coord_ele, arc_angle, rim_crest_elevation, psd_model = Create_outline(diameter_temp,
                                                                                             num_vertices,
                                                                                             rim_crest_elevation_avg,
                                                                                             psd_target_noise)
        WriteOutline(pathout_txt_outline, f"D_{'{:.2f}'.format(diameter_temp)}_km_rim_crest", header, arc_angle,
                     rim_crest_radial_distance, rim_crest_elevation, flag_write_last="yes")
        # --------------------------------------------------------------------------------------------------------------
        control_points = Get_Breakpoints_slope(diameter_temp, breakpoint_diameter_in_fit, coef_sigma_floor_distance,
                                               flag_bp_sigma)
        psd_target = CalculateTargetPSDFromBreakpointSlope(diameter_temp, control_points, num_vertices)
        psd_target_noise = Add_noise_to_array(psd_target, std_dev_array[2])
        x_coord_floor, y_coord_floor, arc_angle, floor_radial_distance, psd_model = Create_outline(diameter_temp,
                                                                                                   num_vertices,
                                                                                                   floor_distance_avg,
                                                                                                   psd_target_noise)
        # -------------------------------------------------------------
        control_points = Get_Breakpoints_slope(diameter_temp, breakpoint_diameter_in_fit, coef_sigma_floor_elevation,
                                               flag_bp_sigma)
        psd_target = CalculateTargetPSDFromBreakpointSlope(diameter_temp, control_points, num_vertices)
        psd_target_noise = Add_noise_to_array(psd_target, std_dev_array[3])
        x_coord_ele, y_coord_ele, arc_angle, floor_elevation, psd_model = Create_outline(diameter_temp,
                                                                                         num_vertices,
                                                                                         floor_elevation_avg,
                                                                                         psd_target_noise)
        if flag_flat_floor_background == "yes":
            floor_elevation = floor_elevation_avg * np.ones(len(floor_elevation))
        WriteOutline(pathout_txt_outline, f"D_{'{:.2f}'.format(diameter_temp)}_km_floor", header,
                     arc_angle, floor_radial_distance, floor_elevation, flag_write_last="yes")
        # --------------------------------------------------------------------------------------------------------------
        control_points = Get_Breakpoints_slope(diameter_temp, breakpoint_diameter_in_fit, coef_sigma_rim_flank_distance,
                                               flag_bp_sigma)
        psd_target = CalculateTargetPSDFromBreakpointSlope(diameter_temp, control_points, num_vertices)
        psd_target_noise = Add_noise_to_array(psd_target, std_dev_array[4])
        x_coord_rim_flank, y_coord_rim_flank, arc_angle, rim_flank_radial_distance, psd_model = Create_outline(
            diameter_temp,
            num_vertices, rim_flank_distance_avg, psd_target_noise)
        # --------------------------------------------------------------
        control_points = Get_Breakpoints_slope(diameter_temp, breakpoint_diameter_in_fit,
                                               coef_sigma_rim_flank_elevation,
                                               flag_bp_sigma)
        psd_target = CalculateTargetPSDFromBreakpointSlope(diameter_temp, control_points, num_vertices)
        psd_target_noise = Add_noise_to_array(psd_target, std_dev_array[5])
        x_coord_ele, y_coord_ele, arc_angle, rim_flank_elevation, psd_model = Create_outline(diameter_temp,
                                                                                             num_vertices,
                                                                                             rim_flank_elevation_avg,
                                                                                             psd_target_noise)
        if flag_flat_floor_background == "yes":
            rim_flank_elevation = rim_flank_elevation_avg * np.ones(len(rim_flank_elevation))
        WriteOutline(pathout_txt_outline, f"D_{'{:.2f}'.format(diameter_temp)}_km_rim_flank", header,
                     arc_angle, rim_flank_radial_distance, rim_flank_elevation, flag_write_last="yes")
        # --------------------------------------------------------------------------------------------------------------
        dem_temp = np.zeros((dimension, dimension))
        resolution = diameter_temp / 2 * extent_radius_ratio / (dimension / 2)
        # print(f"resolution is :{resolution}")
        for jdx in range(dimension):
            # print(jdx / dimension)
            for kdx in range(dimension):
                x = (kdx - (dimension / 2 - 0.5)) * resolution
                y = ((dimension / 2 - 0.5) - jdx) * resolution
                arc_angle_temp = np.arctan2(y, x) + math.pi
                abs_diff = np.abs(arc_angle - arc_angle_temp)
                closest_index_in_arc_angle = np.argmin(abs_diff)
                radial_distance_temp = math.sqrt(x ** 2 + y ** 2)
                # ------------------------------------------------------------------------------------------------------
                floor_radial_distance_at_angle = floor_radial_distance[closest_index_in_arc_angle]
                rim_crest_radial_distance_at_angle = rim_crest_radial_distance[closest_index_in_arc_angle]
                rim_flank_radial_distance_at_angle = rim_flank_radial_distance[closest_index_in_arc_angle]
                if floor_radial_distance_at_angle > rim_crest_radial_distance_at_angle:
                    floor_radial_distance_at_angle = rim_crest_radial_distance_at_angle - 2 * resolution
                if rim_flank_radial_distance_at_angle < rim_crest_radial_distance_at_angle:
                    rim_flank_radial_distance_at_angle = rim_crest_radial_distance_at_angle + 2 * resolution
                # ------------------------------------------------------------------------------------------------------
                floor_elevation_at_angle = floor_elevation[closest_index_in_arc_angle]
                rim_crest_elevation_at_angle = rim_crest_elevation[closest_index_in_arc_angle]
                rim_flank_elevation_at_angle = rim_flank_elevation[closest_index_in_arc_angle]
                # ------------------------------------------------------------------------------------------------------
                if radial_distance_temp <= floor_radial_distance_at_angle:
                    elevation_temp = Get_elevation_on_power_law(0, floor_elevation_avg, floor_radial_distance_at_angle,
                                                                floor_elevation_at_angle, radial_distance_temp,
                                                                x_power=1)
                # ------------------------------------------------------------------------------------------------------
                if radial_distance_temp > floor_radial_distance_at_angle and \
                        radial_distance_temp <= rim_crest_radial_distance_at_angle:
                    elevation_temp = Get_elevation_on_power_law(floor_radial_distance_at_angle,
                                                                floor_elevation_at_angle,
                                                                rim_crest_radial_distance_at_angle,
                                                                rim_crest_elevation_at_angle, radial_distance_temp,
                                                                x_power=3)
                # ------------------------------------------------------------------------------------------------------
                if radial_distance_temp > rim_crest_radial_distance_at_angle and \
                        radial_distance_temp <= rim_flank_radial_distance_at_angle:
                    elevation_temp = Get_elevation_on_power_law(rim_crest_radial_distance_at_angle,
                                                                rim_crest_elevation_at_angle,
                                                                rim_flank_radial_distance_at_angle,
                                                                rim_flank_elevation_at_angle,
                                                                radial_distance_temp, x_power=-3)
                # ------------------------------------------------------------------------------------------------------
                if radial_distance_temp > rim_flank_radial_distance_at_angle:
                    elevation_temp = rim_flank_elevation_avg
                # ------------------------------------------------------------------------------------------------------
                dem_temp[jdx, kdx] = elevation_temp
        # --------------------------------------------------------------------------------------------------------------
        fig = plt.figure()
        plt.figtext(0.5, 0.2, "Range (km)", va='center', ha='center')
        plt.figtext(0.155, 0.6, "Range (km)", rotation=90, va='center', ha='center')
        ax = fig.add_axes([0.2, 0.3, 0.6, 0.6])
        ls = LightSource(azdeg=315, altdeg=25)
        shaded_matrix = ls.shade(dem_temp, cmap=cmap, vert_exag=500, blend_mode='soft')
        ax.imshow(shaded_matrix, cmap=cmap)
        # --------------------------------------------------------------------------------------------------------------
        ax.plot(dimension / 2 - x_coord_rim_crest / resolution - 0.5,
                y_coord_rim_crest / resolution + dimension / 2 - 0.5,
                linewidth=3, color=color_feature_array[0], label="Rim crest")
        ax.plot(dimension / 2 - x_coord_floor / resolution - 0.5, y_coord_floor / resolution + dimension / 2 - 0.5,
                linewidth=3, color=color_feature_array[1], label="Floor")
        ax.plot(dimension / 2 - x_coord_rim_flank / resolution - 0.5,
                y_coord_rim_flank / resolution + dimension / 2 - 0.5,
                linewidth=3, color=color_feature_array[2], label="Rim flank")
        plt.sca(ax)
        plt.xticks([-0.5, dimension / 4 - 0.5, dimension / 2 - 0.5, dimension * 0.75 - 0.5, dimension - 1 + 0.5],
                   [f'{round(-dimension / 2 * resolution, 2)}', f'{round(-dimension / 4 * resolution, 2)}', '0',
                    f'{round(dimension / 4 * resolution, 2)}', f'{round(dimension / 2 * resolution, 2)}'])
        plt.yticks([-0.5, dimension / 4 - 0.5, dimension / 2 - 0.5, dimension * 0.75 - 0.5, dimension - 1 + 0.5],
                   [f'{round(dimension / 2 * resolution, 2)}', f'{round(dimension / 4 * resolution, 2)}', '0',
                    f'{round(-dimension / 4 * resolution, 2)}', f'{round(-dimension / 2 * resolution, 2)}'])
        # --------------------------------------------------------------------------------------------------------------
        x_minor_locator = AutoMinorLocator(5)
        y_minor_locator = AutoMinorLocator(5)
        ax.xaxis.set_minor_locator(x_minor_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)
        legend = ax.legend(loc=2, fontsize=8)
        legend.get_frame().set_alpha(0.3)
        # plt.figtext( id_x,id_x, id_temp, ha='right', va='center')
        plt.figtext(title_x, title_y, f"Synthetic: $D$={'{:.2f}'.format(diameter_temp)} km", ha='center', va='center')
        plt.savefig(f"{pathout_fig_outline}/D_{'{:.2f}'.format(diameter_temp)}_km.pdf", dpi=500)
        if idx==0:print(f"{idx+1} crater  made and saved")
        else:print(f"{idx+1} craters made and saved")
    # ------------------------------------------------------------------------------------------------------------------
    plt.show()




















def Get_elevation_on_power_law(x1,y1,x2,y2,x,x_power):
    a=(y1-y2)/(x1**(x_power)-x2**(x_power))
    b=y1-a*x1**(x_power)
    # print(a,b)
    return a*x**(x_power)+b


def Create_outline(diameter,num_vertices,y_avg,psd_target):
    period_total=psd_target[-1,0]
    x_final = np.linspace(0, period_total, num_vertices, endpoint=False)
    y_final_sum = np.zeros(num_vertices)
    for idx in range(len(psd_target)):
        peroid_temp = psd_target[idx, 0]
        amplitude_temp = math.sqrt(psd_target[idx, 1] * period_total / (num_vertices ** 2))
        phase_temp = np.random.rand() * period_total
        y_final_ind = amplitude_temp * np.sin(2 * math.pi * (1 / peroid_temp) * (x_final + phase_temp))
        y_final_sum = y_final_sum + y_final_ind
    #-------------------------------------------------------------------------------------------------------------------
    x_coord=[]
    y_coord=[]
    for idx in range(len(x_final)):
        x_coord.append(     math.cos(x_final[idx]) * ( y_final_sum[idx]*diameter/2 + y_avg ) )
        y_coord.append(     math.sin(x_final[idx]) * ( y_final_sum[idx]*diameter/2 + y_avg ) )
    x_coord.append(x_coord[0])
    y_coord.append(y_coord[0])
    # ------------------------------------------------------------------------------------------------------------------
    interval=period_total/num_vertices
    psd_model = CalculatePSD(y_final_sum, interval)
    x_final=np.append(x_final,period_total)
    y_final_sum = np.append(y_final_sum, y_final_sum[0])
    return x_coord,y_coord,x_final,y_final_sum*diameter/2 + y_avg,psd_model



def Get_Breakpoints_slope(diameter,breakpoint_diameter_in_fit,coef_sigma,flag_bp_sigma):
    # Do pre-processing
    if diameter < breakpoint_diameter_in_fit:
        index=0
    else:
        index=3

    bp4_y = coef_sigma.loc[index, "Breakpoint_4_y"] * diameter + coef_sigma.loc[index+1, "Breakpoint_4_y"]
    bp3_y = coef_sigma.loc[index, "Breakpoint_3_y"] * diameter + coef_sigma.loc[index+1, "Breakpoint_3_y"]
    bp2_y = coef_sigma.loc[index, "Breakpoint_2_y"] * diameter + coef_sigma.loc[index+1, "Breakpoint_2_y"]
    bp2_x = coef_sigma.loc[index, "Breakpoint_2_x"] * diameter + coef_sigma.loc[index+1, "Breakpoint_2_x"]
    slope_12 = coef_sigma.loc[index, "Slope_12"] * diameter + coef_sigma.loc[index+1, "Slope_12"]
    bp4_y_sigma = coef_sigma.loc[index+2, "Breakpoint_4_y"]
    bp3_y_sigma = coef_sigma.loc[index+2, "Breakpoint_3_y"]
    bp2_y_sigma = coef_sigma.loc[index+2, "Breakpoint_2_y"]
    bp2_x_sigma = coef_sigma.loc[index+2, "Breakpoint_2_x"]
    slope_12_sigma = coef_sigma.loc[index+2, "Slope_12"]
    # ------------------------------------------------------------------------------------------------------------------
    if flag_bp_sigma=="yes":
        bp4_y_final = np.random.normal(bp4_y, bp4_y_sigma)
        bp3_y_final = np.random.normal(bp3_y, bp3_y_sigma)
        bp2_y_final = np.random.normal(bp2_y, bp2_y_sigma)
        bp2_x_final = np.random.normal(bp2_x, bp2_x_sigma)
        slope_12_final = np.random.normal(slope_12, slope_12_sigma)
        control_points_final=[slope_12_final, bp2_x_final, bp2_y_final,bp3_y_final,bp4_y_final]
        print("bp with sigma are: ", control_points_final)
        return control_points_final
    else:
        control_points = [slope_12, bp2_x, bp2_y, bp3_y, bp4_y]
        return control_points



def CalculateTargetPSDFromBreakpointSlope( diameter_temp,control_points,num_vertices):
    slope_12 = control_points[0]
    bp2_x = control_points[1]
    bp2_y = control_points[2]
    bp3_y = control_points[3]
    bp4_y = control_points[4]
    bp4_x = math.log10(   2*math.pi  )
    bp3_x =  math.log10(10 ** bp4_x / 2)
    interval=10**bp4_x/num_vertices
    dfft = fft.rfft(   np.ones(num_vertices)   )
    iend = dfft.size - 1
    freq = fft.fftfreq(num_vertices, interval)
    wavelength = 1 / freq[1:iend]
    psd = [[0 for i in  range(2) ] for i in range(iend-1)  ]
    psd = np.array(psd)
    psd = np.float64(psd)
    psd[:, 0] = wavelength
    psd[:, 1] = np.abs(dfft[1:iend])
# ----------------------------------------------------------------------------------------------------------------------
    for i in range(len(psd)):
        if psd[i, 0] < 10**bp2_x:
            bp2_x_index = i-1
            break
    k_23=(bp3_y-bp2_y)/(bp3_x-bp2_x)
    b_23 = bp3_y-k_23*bp3_x
    k_12=slope_12
    b_12 = bp2_y-k_12*bp2_x
#-----------------------------------------------------------------------------------------------------------------------
    for idx in range(len(psd)):
        if idx==0:
            psd[idx,1]=10**bp4_y
        if idx==1:
            psd[idx,1]=10**bp3_y
        if idx>1 and idx<=  bp2_x_index:
            psd[idx, 1] =10**(k_23*math.log10(psd[idx, 0])+b_23)
        if idx>bp2_x_index:
            psd[idx, 1] =10**(k_12*math.log10(psd[idx, 0])+b_12)
# ---------------------------------------------------------------------------------------------------------------------------------
    psd=np.flipud(psd)
    return psd


def Add_noise_to_array(psd_target,noise):
    psd_target_noise = psd_target
    power_target_log = np.log10(psd_target[:, 1])
    power_target_log_noise = np.random.normal(0, noise, power_target_log.shape) + power_target_log
    power_target_noise = 10 ** power_target_log_noise
    psd_target_noise[:, 1] = power_target_noise
    return psd_target_noise



def Get_rim_height(diameter,breakpoint_diameter_in_fit,flag_avg_sigma):
    a1=0.03201970096980497
    b1=0.14773892714248935
    sigma_1=0.0034836774525572965
    a2=0.2080709807194973
    b2=-0.5750866649276306
    sigma_2=0.0042548300628533495
    if diameter<breakpoint_diameter_in_fit:
        mean=a1*diameter**b1
        if flag_avg_sigma=="yes":
            return np.random.normal(mean, sigma_1, 1)[0]*diameter
        else:
            return mean*diameter
    else:
        mean = a2 * diameter ** b2
        if flag_avg_sigma=="yes":
            return np.random.normal(mean, sigma_2, 1)[0] * diameter
        else:
            return mean*diameter



def Get_crater_depth(diameter,breakpoint_diameter_in_fit,flag_avg_sigma):
    a1=0.1899251405278605
    b1=0.03230251842222747
    sigma_1=0.01792480619222002
    a2=1.0834922771319102
    b2=-0.7065725758658646
    sigma_2=0.012037153338669018
    if diameter<breakpoint_diameter_in_fit:
        mean=a1*diameter**b1
        if flag_avg_sigma=="yes":
            return np.random.normal(mean, sigma_1, 1)[0]*diameter
        else:
            return mean*diameter
    else:
        mean = a2 * diameter ** b2
        if flag_avg_sigma=="yes":
            return np.random.normal(mean, sigma_2, 1)[0] * diameter
        else:
            return mean*diameter




def Get_floor_radius(diameter,flag_avg_sigma):
    a=0.13213106378523629
    b=0.32536543622022224
    sigma=0.04273883353772488
    mean=a*diameter**b
    if flag_avg_sigma == "yes":
        return np.random.normal(mean, sigma, 1)[0]*diameter/2
    else:
        return mean*diameter/2





def Get_rim_flank_radius(diameter,flag_avg_sigma):
    a=2.8740358071397156
    b=-0.12400776345642424
    sigma=0.18025709610312335
    mean = a * diameter ** b
    if flag_avg_sigma == "yes":
        return np.random.normal(mean, sigma, 1)[0]*diameter/2
    else:
        return mean*diameter/2






def  WriteOutline(pathout,crater_name, header, lon_coor, lat_coor,elevation,flag_write_last):
    filename = crater_name + '.txt'
    path_out = os.path.join(pathout, filename)
    with open(path_out, 'w') as f:
        f.write(header + "\n")
        f.close
    if flag_write_last=="yes":
        length=len(lon_coor)
    else:
        length = len(lon_coor)-1
    with open(path_out, 'a') as f:
        for i in range(length):
            if i == length - 1:
                f.write(str(lon_coor[i]) + "," + str(lat_coor[i]) + "," + str(elevation[i]))
            else:
                f.write(str(lon_coor[i]) + "," + str(lat_coor[i]) + "," + str(elevation[i])+ "\n")
        f.close


def CalculatePSD(y_signal, interval):
    wavelength_output, psd_output = profile2psd(y_signal, interval)
    psd = [[0 for i in range(len(wavelength_output))] for i in range(2)]
    psd = np.array(psd)
    psd = np.float64(psd)
    psd[0, :] = wavelength_output
    psd[1, :] = psd_output
    psd = psd.T
    psd = np.flipud(psd)
    return psd


def profile2psd(y, pix):
    """
    Computes a 1D PSD from a 1D elevation profile
    Inputs:
        y  : dependent variable of profile
        pix : the size of each segment of y
    Outputs:
        psd1D : the 1D PSD
        wavelength : numpy array of wavelength values
    """
    dfft = fft.rfft(y)

    n = y.size
    psd1D = (2 * np.abs(dfft)) ** 2 / (pix * n)

    # psd1D = (2 * np.abs(dfft))  #/ ( n)
    if len(y)%2==0:
        index_end=int(len(y)/2)
    else:
        index_end = math.ceil(len(y) / 2)
    freq = fft.fftfreq(n, pix)
    wavelength = 1 / freq[1:index_end]
    temp_freq=freq[1:index_end]
    return wavelength, psd1D[1:index_end]