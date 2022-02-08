import pathlib
from numpy.lib.function_base import diff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import access2thematrix
import pySPM
import spiepy
import cv2
import os
import statistics as st
from pathlib import Path

from skimage.registration import phase_cross_correlation


def load_All_meta(filenames, subfolder_path):

    '''
    Function to load and collate all files (both sxm and mtrx for Dylan)
    '''

    master_parameters = [0 for z in range(len(filenames))]
    master_key_parameters = [0 for z in range(len(filenames))] 
     
    
    for k in range(0,len(filenames)):
        if filenames[k].endswith('.sxm'):
    
            filename =  subfolder_path + filenames[k]
            
            header=[];
            footer=[];
            decodedfooter=[];
            metadata=[];
        
            zcont=[];
            info=[];
            fn = pathlib.Path(filename)
        
            print('Importing metadata for ' + filenames[k] + '...')
            if fn.exists ():
                #print ("File exist")
                with open(fn, 'rb') as fid:
                    fid = fid.readlines()
                    #this loads all metadata in big table
            else:
                print ("File does not exist")
            ##############################################################    
            
                s1 = fid[0]
                if b':NANONIS_VERSION:' in s1:
                    print('File seems to be a Nanonis file\n')
                else:
                    print('File seems not to be a Nanonis file\n')
                    #this is a carryover from the MATLAB code - doesn't do much here but who knows when could be useful to 
                    #be told if nanonis file weird or whatever
        
            Stop = fid.index(b':SCANIT_END:\n')
            Endline = Stop + 2
            Gap = Stop - 1
        
            #########################################################################
            #This section is all about formating meta data into more digestable dataframe 
            
            for i in range(0,Endline):
                if i == Gap:
                    continue
                
                s1 = fid[i]
                s1 = s1.decode('utf-8')
                #header.append(s1)
            
                        
            
                if s1.endswith(':\n') == True:
                
                    #':Z-CONTROLLER:\n':
                        #print ('found a header')
                
                        if s1 == ':Z-CONTROLLER:\n':
                            header.append(s1)
                            zcont.append(fid[(i+1)])
                            zcont.append(fid[(i+2)])
                            Zcontroller = zcont[0] + zcont[1]
                            footer.append(Zcontroller)
                    
                    
                    
                    
                        elif s1 == ':DATA_INFO:\n':
                            header.append(s1)
                            info.append(fid[(i+1)])
                            info.append(fid[(i+2)])
                            info.append(fid[(i+3)])
                            datainfo = info[0] + info[1] + info[2]
                            footer.append(datainfo)
                    
                
                        
                    
                                
                        else:
                            header.append(s1)
                            s2 = fid[(i+1)]
                            footer.append(s2) 
                
                        
                else:
                    None
            
            for j in range (0, len(footer)):
                line = footer[j]
                footer_decod = line.decode('utf-8')
                decodedfooter.append(footer_decod)
            
        
            
    
            df_h = pd.DataFrame(header).T
            df_f = pd.DataFrame(decodedfooter).T
        
            metadata = pd.concat([df_h, df_f], axis=0)
            new_header = metadata.iloc[0] #grab the first row for the header
            metadata = metadata[1:] #take the data less the header row
            metadata.columns = new_header #set the header row as the df header
        
        
            master_parameters[k] = metadata
            
            
        
            '''collating all relevant parameters'''
        
            a = metadata[':SCAN_PIXELS:\n']
            a = a[0]
            a = a.split()
            xpixel = float(a[0])
            ypixel = float(a[1])
            Scan_pixels = np.vstack((xpixel, ypixel))
            Scan_pixels = Scan_pixels.T
            T = Scan_pixels
        
            a = metadata[':SCAN_RANGE:\n']
            a = a[0]
            a = a.split()
            xrange = float(a[0])
            yrange = float(a[1])
            Scan_range = np.vstack((xrange, yrange))
            Scan_range = Scan_range.T
            T = np.append(T, Scan_range)
        
            a = metadata[':SCAN_OFFSET:\n']
            a = a[0]
            a = a.split()
            xcoord = float(a[0])
            ycoord = float(a[1])
            Coordinates = np.vstack((xcoord, ycoord))
            Coordinates = Coordinates.T
            T = np.append(T, Coordinates)
        
        
        
            a = metadata[':BIAS:\n']
            a = a[0]
            Bias = float(a)
            T = np.append(T, Bias)
        
        #Have no intuition as to what is the most efficient/sensible way to extract the metadata bits I'm after
        #for manipulation and handling etc. Currently code gets it all, and then goes from overall table finds
        #variables based on their names, as of now, with STM, I don't need more than pixels, range, coordinates, 
        #bias and tip lift
        
        #the previous section gets the values for all these table values, and makes a new array, for each image,
        #containing only the relevant information, and no matter where the whereabouts of the variables in the 
        #metadata table (now established column number dependent on what you choose to save in nanonis) the new
        #table for each image will always be of this format (for now):
            
            '''Xpixel    Xrange     Xcoordinate     Bias   '''
            '''Ypixel    Yrange     Ycoordinate     Tiplift'''
            
        #Indexing by number is more natural to me anyway
        
        
            a = metadata[':Z-Controller>TipLift (m):\n']
            a = a[0]
            Tiplift = float(a)
            T = np.append(T, Tiplift)
        
            T = np.reshape(T, (-1,2)).T
            
            master_key_parameters[k] = T
            

        if filenames[k].endswith('_mtrx'):
            print('Importing metadata for ' + filenames[k] + '...')
            file = subfolder_path + filenames[k]
            mtrx_data = access2thematrix.MtrxData()
            data_file = r'{}'.format(file)
            traces, message = mtrx_data.open(data_file)
            im, message = mtrx_data.select_image(traces[0])  
            
            '''collating all relevant parameters'''

            #Scan Pixels
            xpixel, ypixel = im.data.shape
            xpixel = float(xpixel)
            ypixel = float(ypixel)
            Scan_pixels = np.vstack((xpixel, ypixel))
            Scan_pixels = Scan_pixels.T
            T = Scan_pixels
            
            #Scan Range
            xrange = float(im.width)
            yrange = float(im.height)
            Scan_range = np.vstack((xrange, yrange))
            Scan_range = Scan_range.T
            T = np.append(T, Scan_range)
        
            #Scan offset
            xcoord = float(im.x_offset)
            ycoord = float(im.y_offset)
            Coordinates = np.vstack((xcoord, ycoord))
            Coordinates = Coordinates.T
            T = np.append(T, Coordinates)
        
            #Bias - Don't know how to get it here so putting zeros to keep with format
            Bias = float(0)
            T = np.append(T, Bias)
            
            #Tiplift - Don't know how to get it here so putting zeros to keep with format
            Tiplift = float(0)
            T = np.append(T, Tiplift)
            
            #Reshaping into the following format
            '''Xpixel(0s)    Xrange     Xcoordinate     Bias(0s)   
               Ypixel(0s)    Yrange     Ycoordinate     Tiplift(0s)'''

            T = np.reshape(T, (-1,2)).T
            
            master_key_parameters[k] = T
            
            master_parameters[k] = filenames[k] + ' does not need this'
            

    return master_parameters, master_key_parameters
            
        #####################################################################

#####################################################################

def get_image_data(filename, input_folder = ''):
    ''' Function to return both the raw and flattened sxm images from the inputs.
    
    filename - name of the file
    input_folder - location of the directory containing the input file
    '''

    input_folder = Path(input_folder)

    filepath = input_folder.joinpath(filename)

    if filename.suffix == '.sxm':
        S = pySPM.SXM(filepath)
        I = S.get_channel('Z').show()
        rawZ= I._A#*1e12                  
        rawZ = np.array(rawZ)
        rawZ = np.flip(rawZ, 0)
        img = rawZ

    if filename.suffix == '.Z_mtrx':
        mtrx_data = access2thematrix.MtrxData()
        data_file = r'{}'.format(filepath)
        traces, message = mtrx_data.open(data_file)
        im, message = mtrx_data.select_image(traces[0])
        img = im.data

    #Flattens the image data
    im_flat, _ = spiepy.flatten_xy(img) 
    mask, _ = spiepy.mask_by_troughs_and_peaks(im_flat)
    im_flat, _ = spiepy.flatten_by_peaks(img, mask)

    #Normalises the image data
    img_flat_norm = cv2.normalize(im_flat, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)

    plt.close()

    return img_flat_norm, im_flat, img

def Get_times(filename):

    ''' Function which returns the start time and total scan time for an sxm file. 

    filename - Input path for sxm file 
    '''

    cwd = os.getcwd()
    
    file_path = pathlib.Path(filename[len(cwd)+1:]) #pathlib.Path(input_folder + filename)
        
    if file_path.exists:
        with open(file_path, 'rb') as fid:
            fid = fid.readlines()
    else:
                print ("File does not exist")
    
    Stop = fid.index(b':SCANIT_END:\n')
    Endline = Stop + 2
    Gap = Stop - 1

    for i in range(0, Endline):
        if i == Gap:
            continue
        
        s1 = fid[i]
        s1 = s1.decode('utf-8')

        if s1.endswith(':\n') == True:
            if s1 == ':REC_TIME:\n':
                Scan_start = fid[i+1].decode('utf-8')
            elif s1 == ':ACQ_TIME:\n':
                Scan_time = fid[i+1].decode('utf-8')

    return Scan_start, Scan_time

def Get_ranges(filename):

    ''' Function which returns the scan range and scan pixel values for an sxm file. 

    filename - Input path for sxm file 
    '''

    file_path = pathlib.Path(filename)
        
    if file_path.exists:
        with open(file_path, 'rb') as fid:
            fid = fid.readlines()
    else:
                print ("File does not exist")
    
    Stop = fid.index(b':SCANIT_END:\n')
    Endline = Stop + 2
    Gap = Stop - 1

    for i in range(0, Endline):
        if i == Gap:
            continue
        
        s1 = fid[i]
        s1 = s1.decode('utf-8')

        if s1.endswith(':\n') == True:
            if s1 == ':SCAN_PIXELS:\n':
                Scan_pixels = fid[i+1].decode('utf-8')
            elif s1 == ':SCAN_RANGE:\n':
                Scan_range = fid[i+1].decode('utf-8')

    return Scan_pixels, Scan_range

def flatten_by_line(im):
    
    rows, cols = im.shape
    #corrected_row = [0 for value in range(0, cols)]
    flattened_im = [[0 for x in range(0,cols)] for y in range(0, rows)]
    differece_im = [[0 for x in range(0,cols)] for y in range(0, rows)]
    poo = 1

    for row in range(0, rows):
        median = st.median(im[row])
        for col in range(0, cols):
            corrected_element = im[row][col] - median
            flattened_im[row][col] = corrected_element
            differece_im[row][col] = median

    return np.array(flattened_im), np.array(differece_im)

def flatten_by_line_2(im):
    
    rows, cols = im.shape
    #corrected_row = [0 for value in range(0, cols)]
    flattened_im = [[0 for x in range(0,cols)] for y in range(0, rows)]
    differece_im = [[0 for x in range(0,cols)] for y in range(0, rows)]
    poo = 1

    for row in range(0, rows):
        median = st.median(im[row])
        for col in range(0, cols):
            corrected_element = im[row][col] - median
            flattened_im[row][col] = corrected_element
            differece_im[row][col] = median

    return np.array(flattened_im), np.array(differece_im)

def Find_Drift(filenames, input_folder):

    ''' Function which applied phase cross correlation between two images to calculate the drift between the two. Returns the drift in pixels as shift and the error in overlap between the two images in error.

    filenames - list of two filenames which should be the same but with some drift between. 
    '''

    flattened_images = {}
    flattened_normalised_images = {}
    raw_images = {}
    img_flat = {}

    for filename in filenames:
        #Import image data
        try:
            flattened_normalised_images['{}'.format(filename)], flattened_images['{}'.format(filename)], raw_images['{}'.format(filename)] = get_image_data(filename, input_folder)
            #Using line flattening instead of plane from spiepy
            img_flat['{}'.format(filename)] = flatten_by_line(raw_images[filename]) 
            img_flat[filename] = cv2.normalize(img_flat[filename], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)

            image = img_flat[filenames[0]]
            offset_image = img_flat[filenames[1]]
            
            #calculate x and y pixel drift
            shift, error, diffphase = phase_cross_correlation(image, offset_image)
            shift = -shift          #For some reason it results in the opposite shift?? This might be something which I have two change when not using the simulator - it isn't.

            #calculate z drift
            initial_avg_z = np.average(img_flat[filenames[0]])
            final_avg_z = np.average(img_flat[filenames[1]])

            z_shift = final_avg_z -initial_avg_z

        except np.linalg.LinAlgError:
            shift = ['None', 'None']
            z_shift = 'None'
            error = 'None'
            continue
        except KeyError:
            shift = ['None', 'None']
            z_shift = 'None'
            error = 'None'
            continue
    
    return shift, z_shift, error

def Get_ranges(filename, input_folder = ''):

    ''' Function which returns the scan range and scan pixel values for an sxm file. 

    filename - Input path for sxm file 
    '''
    filepath = pathlib.Path(filename)

    #filepath = input_folder.joinpath(filename)

    if filepath.exists:
        with open(filepath, 'rb') as fid:
            fid = fid.readlines()
    else:
                print ("File does not exist")
    
    Stop = fid.index(b':SCANIT_END:\n')
    Endline = Stop + 2
    Gap = Stop - 1

    for i in range(0, Endline):
        if i == Gap:
            continue
        
        s1 = fid[i]
        s1 = s1.decode('utf-8')

        if s1.endswith(':\n') == True:
            if s1 == ':SCAN_PIXELS:\n':
                Scan_pixels = fid[i+1].decode('utf-8')
            elif s1 == ':SCAN_RANGE:\n':
                Scan_range = fid[i+1].decode('utf-8')

    return Scan_pixels, Scan_range

def detect_tip_change(im, boundary = 2.0e-11):
    ''' Function to detect whether a tip change has occurred in a scan. The input image should not have been flattened prior to detection. 

    Input:
        im: numpy array image
        boundary: Threshold at which a tip change would be detected between successive lines in a scan. If not specified boundary = 2.0e-11. 

    Output: 
        tc_detected: Boolean value True if tip change has been detected and False if not. 

    '''
    tc_detected = False

    img_flat, img_diff = flatten_by_line(im)

    rows, cols = im.shape

    img_diff = np.flip(img_diff, 0)
    line_scan = img_diff[:,100]

    for line in range(len(line_scan)):
        if line < (rows - 2):
            line_diff = abs(line_scan[line] - line_scan[line+1]) 
        if line_diff > boundary:
            tc_detected = True
            continue
    return tc_detected

def tip_change_positions(im, boundary = 2.0e-11) :
    ''' Function to output in which rows a tip change has occurred in a scan. To be used in conjuction with other functions for splitting images. 

    Input: 
        im: numpy array image
        boundary: Threshold at which a tip change would be detected between successive lines in a scan. If not specified boundary = 2.0e-11. 

    Output: 
        tc_position: list containing the line numbers of all tip changes in a scan.

    '''
    
    img_flat, img_diff = flatten_by_line(im)

    rows, cols = im.shape

    img_diff = np.flip(img_diff, 0)
    line_scan = img_diff[:,100]

    tc_positions = []
    removed_tc_positions = []

    for line in range(len(line_scan)):
        if line < (rows - 2):
            line_diff = abs(line_scan[line] - line_scan[line+1]) 
            # line_diff_2 = abs(line_scan[line] - line_scan[line+2])        #Taken out for now but could be used to detect changes over more than one pixel if needed. 
        if line_diff > 2.0e-11:
            tc_positions.append(line)

    for position in range(len(tc_positions)):
        if tc_positions[position] == tc_positions[position-1]+1:
            removed_tc_positions.append(tc_positions[position]) 

    for position in removed_tc_positions:
        if position in tc_positions:
            tc_positions.remove(position)

    return tc_positions

def Split_tc_im(filename, boundary = 2.0e-11, min_lines = None) :
    ''' Function to output in which rows a tip change has occurred in a scan. To be used in conjuction with other functions for splitting images. 

    Input: 
        filename: file path of scan
        boundary: Threshold at which a tip change would be detected between successive lines in a scan. If not specified boundary = 2.0e-11. 
        min_lines: Minimum number of lines for an image to be accepted as worthy of a split. 

    Output: 
        images_split: list containing each image the whole has been split into. Iterate through in for loop to plot. 

    '''


    img_flat_norm, im_flat, im = get_image_data(filename)

    img_flat, img_diff = flatten_by_line(im)

    #If min_lines has not been specified, the number of lines needed to capture 1.5 unit cells is calculated (based on Si(111) 7x7)
    if min_lines == None:
        unit_cell_onepointfive = 2.64e-9 * 1.5
        Scan_pixels, Scan_range = Get_ranges(filename)
        Scan_pixels = float(Scan_pixels.split()[0])
        Scan_range = float(Scan_range.split()[0])

        dist_per_pixel = Scan_range / Scan_pixels

        min_lines = round(unit_cell_onepointfive / dist_per_pixel)
        
    rows, cols = im.shape

    #img_diff = np.flip(img_diff, 0)
    line_scan = img_diff[:,100]

    tc_positions = []
    removed_tc_positions = []
    image_splits = []
    images_split = []

    for line in range(len(line_scan)):
        if line < (rows - 2):
            line_diff = abs(line_scan[line] - line_scan[line+1]) 
            # line_diff_2 = abs(line_scan[line] - line_scan[line+2])        #Taken out for now but could be used to detect changes over more than one pixel if needed. 
        if line_diff > 2.0e-11:
            tc_positions.append(line)

    tc_positions.append(rows)

    for position in range(len(tc_positions)):
        if tc_positions[position] == tc_positions[position-1]+1:
            removed_tc_positions.append(tc_positions[position]) 

    for position in removed_tc_positions:
        if position in tc_positions:
            tc_positions.remove(position)

    for i in range(1, len(tc_positions)):
        current_split = [tc_positions[i-1], tc_positions[i]]
        if len(range(current_split[0], current_split[1])) >= min_lines:
            image_splits.append(current_split)
    
    for split in image_splits:
        image = img_flat[split[0]:split[1],:]
        images_split.append(image)
    
    return images_split