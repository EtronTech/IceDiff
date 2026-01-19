import pickle as pkl
import netCDF4 as nc
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
sys.path.append('/Workspace/IceDiff/IceDiff-FM')
from ConfigHandler import Handler as cfg
from utils import MovingAverage
import copy

import time

import argparse

def Calendar_List(start,end):
    '''
    Generate string of Calendar dates.
    '''
    if start == 1978:
        start_mon = 10
        start_day = 25
    else:
        start_mon = 1
        start_day = 1
    start_date = datetime(start, start_mon, start_day)
    end_date = datetime(end+1, 1, 1) # To include December 31st
    t = (end_date - start_date).days
    return [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(0, (end_date - start_date).days)]

def Daily_SIC_List(config,calendar):
    '''
    Generate SIC file names according to specific dates.
    '''
    file_names = os.listdir(config.DAILY_DATA_DIR)
    sic_file_prefix = [["seaice_conc_daily_nh_" + date, date] for date in calendar]
    sic_daily_files = []
    for f in file_names:
        if len(sic_file_prefix) > 0:
            for pre in sic_file_prefix:
                if f.startswith(pre[0]) and f.endswith(".nc"):
                    sic_daily_files.append([config.DAILY_DATA_DIR + '/' + f,pre[1]])
                    sic_file_prefix.remove(pre)
                    break
        else:
            break
    return sic_daily_files

def ndarrayConversion(fPath,varName='cdr_seaice_conc',varIdx=0,mask=False):
    '''
    Extract SIC data from .nc files.
    '''
    ds = nc.Dataset(fPath,mode='r') # Read only for Dataset
    vs = ds.get_variables_by_attributes(name=varName) 
    vs[varIdx].set_auto_mask(mask) # Retrieve Actual Value 
    return vs[varIdx][...]  


def Generate_Daily_Dataset_v2(config,curr_split,split_start_year,split_end_year,designated_sic_list=None):
    # Acquire Daily NetCDF files
    # Calender List
    calendar = Calendar_List(split_start_year,split_end_year)
    
    # Acquire File List
    if designated_sic_list is None:
        sic_daily_files = Daily_SIC_List(config,calendar)
    else:
        sic_daily_files = designated_sic_list
    
    varNames = ['cdr_seaice_conc']

    # Load file and Parse Data
    print('generate pkl data...')
    save_dir = config.OUTPUT.DIR + '/' + curr_split
    os.makedirs(save_dir, exist_ok=True)
    size = len(sic_daily_files)
    # pkl_data = np.zeros((size,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32) # Fixed NSIDC SIC Shape
    
    input_channels = config.OUTPUT.INPUT_LEN
    pred_channels = config.OUTPUT.OUTPUT_LEADS


    def save_one_slide_window(data,dir,name):
        with open(dir + '/' + name, 'wb') as f:
            pkl.dump([data], f)


    if len(varNames) == 1:
        '''
        (M)ulti (G)ranularity (D)aily, (I)nput (C)hannels, (O)utput (C)hannels.
        '''
        suffix = '_MGD_' 'IC' + str(input_channels) + '_' + 'OC' + str(pred_channels)
        total_slide_windows = size - (input_channels + pred_channels)  # slide step = 1

        pkl_dict = {}

        latest_file_interval = (input_channels + pred_channels)  - 1
        with tqdm(total=total_slide_windows) as bar:
            var = varNames[0]
            pkl_data = {}
            pkl_data['Input'] = np.zeros((input_channels,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32)
            pkl_data['Output'] = np.zeros((pred_channels,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32)
            output_start_file_index = 0
            for i in range(total_slide_windows):
                if i == 0:
                    file_index = 0
                    
                    for c in range(input_channels):
                        file = sic_daily_files[file_index]
                        tempArray = ndarrayConversion(file[0],var)
                        tempArray[tempArray>1] = 0. # Clear Mask Values
                        pkl_data['Input'][c] = tempArray
                        file_index += 1
                    
                    output_start_date = sic_daily_files[file_index][1]
                    output_start_file_index = file_index
                    for c in range(pred_channels):
                        file = sic_daily_files[file_index]
                        tempArray = ndarrayConversion(file[0],var)
                        tempArray[tempArray>1] = 0. # Clear Mask Values
                        pkl_data['Output'][c] = tempArray
                        file_index += 1

                    save_one_slide_window(pkl_data,save_dir,str(i)+suffix)
                    pkl_dict[i] = output_start_date
                    
                    # print(pkl_dict[i])
                    
                else:
                    pkl_data['Input'][0][:,:] = 0.
                    update_input = copy.deepcopy(pkl_data['Output'][0])
                    pkl_data['Output'][0][:,:] = 0.

                    insert_file_index = i + latest_file_interval
                    file = sic_daily_files[insert_file_index]
                    update_output = ndarrayConversion(file[0],var)
                    update_output[update_output>1] = 0. # Clear Mask Values

                    tmp_input = np.zeros((input_channels,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32)
                    tmp_output = np.zeros((pred_channels,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32)

                    tmp_input[:-1] = copy.deepcopy(pkl_data['Input'][1:])
                    tmp_output[:-1] = copy.deepcopy(pkl_data['Output'][1:])

                    pkl_data['Input'][:,:,:] = 0.
                    pkl_data['Output'][:,:,:] = 0.

                    tmp_input[-1] = update_input
                    tmp_output[-1] = update_output

                    pkl_data['Input'] = copy.deepcopy(tmp_input)
                    pkl_data['Output'] = copy.deepcopy(tmp_output)

                    # pkl_dict[i] = file[1]
                    output_start_file_index += 1
                    pkl_dict[i] = sic_daily_files[output_start_file_index][1]
                    save_one_slide_window(pkl_data,save_dir,str(i)+suffix)
                    # print(pkl_dict[i])
                
                # time.sleep(0.3)
                bar.update(1)
        
        with open(save_dir + '/' + 'Multi_Granularity_dict', 'wb') as f:
            pkl.dump([pkl_dict], f)



def Generate_Multi_Granularity_Dataset_v2(config,curr_split,split_start_year,split_end_year,designated_sic_list=None):
    '''
    Optimized for saving memory consumption, Batch
    '''
    # Acquire Daily NetCDF files
    # Calender List
    calendar = Calendar_List(split_start_year,split_end_year)

    # Acquire File List
    if designated_sic_list is None:
        sic_daily_files = Daily_SIC_List(config,calendar)
    else:
        sic_daily_files = designated_sic_list
    
    varNames = ['cdr_seaice_conc']

    # Load file and Parse Data
    print('generate Multi-Granularity pkl data...')
    save_dir = config.OUTPUT.DIR + '/' + curr_split
    os.makedirs(save_dir, exist_ok=True)
    size = len(sic_daily_files)

    input_channels = config.OUTPUT.INPUT_LEN
    pred_channels = config.OUTPUT.OUTPUT_LEADS

    periods = config.OUTPUT.MULTI_GRANULARITY_PERIOD

    def process_one_slide_window_v2(window_index,output_index,pkl_data,input,output):
        # output_start_date = sic_daily_files[window_index][1]
        output_start_index = 0
        # Process First Slide Window  (Sequentially Fill Up Empty Windows)
        if window_index == 0:
            file_index = 0
            for c in range(input_channels):
                # Input
                for p in range(period):
                    file = sic_daily_files[file_index]
                    tempArray = ndarrayConversion(file[0],var)[0]
                    tempArray[tempArray>1] = 0. # Clear Mask Values
                    input.insert_daily_data(tempArray,c,p)
                    file_index += 1
            pkl_data['Input'] = input.average

            # Output
            output_start_date = sic_daily_files[file_index][1]
            output_start_index = file_index
            for c in range(pred_channels):
                for p in range(period):
                    file = sic_daily_files[file_index]
                    tempArray = ndarrayConversion(file[0],var)[0]
                    tempArray[tempArray>1] = 0. # Clear Mask Values
                    output.insert_daily_data(tempArray,c,p)
                    file_index += 1
            pkl_data['Output'] = output.average
            
        else:
            # Data Slides in from the Output Moving Average Window
            # Process Output Channels
            insert_file_index = window_index + latest_file_interval
            file = sic_daily_files[insert_file_index]
            tempArray = ndarrayConversion(file[0],var)[0]
            tempArray[tempArray>1] = 0. # Clear Mask Values
            update = output.slide(tempArray)
            pkl_data['Output'] = output.average

            # Process Input Channels
            _ = input.slide(update)
            pkl_data['Input'] = input.average
            
            output_start_index = output_index + 1
            output_start_date = sic_daily_files[output_start_index][1]

        return output_start_date, output_start_index
    
    def save_one_slide_window(data,dir,name):
        with open(dir + '/' + name, 'wb') as f:
            pkl.dump([data], f)


    if len(varNames) == 1:
        for period in periods:
            '''
            (M)ulti (G)ranularity (P)eriod, (I)nput (C)hannels, (O)utput (C)hannels.
            '''
            suffix = '_MGP' + str(period) + '_' 'IC' + str(input_channels) + '_' + 'OC' + str(pred_channels)
            total_slide_windows = size - (input_channels + pred_channels) * period # slide step = 1

            pkl_dict = {}

            input_moving_average = MovingAverage(num_average_window=input_channels, period=period)
            output_moving_average = MovingAverage(num_average_window=pred_channels, period=period)
            input_moving_average.init_moving_average_iter()
            output_moving_average.init_moving_average_iter()

            latest_file_interval = (input_channels + pred_channels) * period - 1
            output_start_index = 0
            
            with tqdm(total=total_slide_windows) as bar:
                var = varNames[0]
                for i in range(total_slide_windows):
                    pkl_data = {}
                    pkl_data['Input'] = np.zeros((input_channels,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32)
                    pkl_data['Output'] = np.zeros((pred_channels,config.SIC_SIZE[0],config.SIC_SIZE[1]),dtype=np.float32)
                    pkl_dict[i], output_start_index = process_one_slide_window_v2(i,output_start_index, pkl_data, input_moving_average, output_moving_average)
                    save_one_slide_window(pkl_data,save_dir,str(i)+suffix)
                    # print(pkl_dict[i])
                    bar.update(1)
        
        with open(save_dir + '/' + 'Multi_Granularity_dict', 'wb') as f:
            pkl.dump([pkl_dict], f)

    
    return


def generate_masks(config):
    # Load NSIDC Ancilliary Data
    Path = config.MASK_DATA_DIR +'/G02202-cdr-ancillary-nh.nc'
    ds_nc = nc.Dataset(Path,mode='r') # read only
    vs_nc = ds_nc.get_variables_by_attributes(name='landmask')
    vs_data = vs_nc[0][...]

    mask_Ocean = vs_data == 0
    mask_Lake  = vs_data == 2
    mask_Coast = vs_data == 253
    mask_Land  = vs_data == 254

    mask_data = {}
    mask_data['mask_Ocean'] = mask_Ocean
    mask_data['mask_Lake'] = mask_Lake
    mask_data['mask_Coast'] = mask_Coast
    mask_data['mask_Land'] = mask_Land

    with open(Path + '/nsidc_masks','wb') as f:
        pkl.dump([mask_data],f)

    data = []
    with open(Path + '/nsidc_masks','rb') as f:
        data.extend(pkl.load(f))


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--gen_mask', type=int,
                    default=0, help='Generate Ocean masks.')
    
    parser.add_argument('--gen_daily', type=int,
                    default=0, help='Generate Ocean masks.')
    
    parser.add_argument('--conf_pth', type=str,
                    default='None', help='Configuration file.')        


    args = parser.parse_args()
    
    gen_mask = args.gen_mask

    gen_daily = args.gen_daily

    config_file = args.conf_pth

    config = cfg(config_file)
    config = config.get_data.DATA_SETS.SIC

    if gen_mask > 0:
        generate_masks(config)

    if config.OUTPUT.MULTI_GRANULARITY:
        if gen_daily == 0:
            # Generate Dataset for Training
            Generate_Multi_Granularity_Dataset_v2(config,'TRAIN',config.TRAIN_SPLIT[0],config.TRAIN_SPLIT[1])
            Generate_Multi_Granularity_Dataset_v2(config,'VAL',config.VAL_SPLIT[0],config.VAL_SPLIT[1])
            Generate_Multi_Granularity_Dataset_v2(config,'TEST',config.TEST_SPLIT[0],config.TEST_SPLIT[1])
        else:
            Generate_Daily_Dataset_v2(config,'TRAIN',config.TRAIN_SPLIT[0],config.TRAIN_SPLIT[1])
            Generate_Daily_Dataset_v2(config,'VAL',config.VAL_SPLIT[0],config.VAL_SPLIT[1])
            Generate_Daily_Dataset_v2(config,'TEST',config.TEST_SPLIT[0],config.TEST_SPLIT[1])
        
        
