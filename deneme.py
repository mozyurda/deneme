# code authors: Marta Krawczyk, Gizem DINCER

#use for 300um Full data: python3 HexaBatch.py --MGS NO --numberOfChannels 199 --SensorThickness 300 --campaignType Production --cms NO --deliveryMonth April --deliveryYear 24
#use for 200um LD Full data: python3 HexaBatch.py --MGS NO --numberOfChannels 199 --SensorThickness 200 --campaignType Production --cms NO --deliveryMonth March --deliveryYear 24
#use for 200um HD Full data: python3 HexaBatch.py --MGS NO --numberOfChannels 445 --SensorThickness 200 --campaignType Production --cms NO --deliveryMonth March --deliveryYear 24
#use for 120um Full data: python3 HexaBatch.py --MGS NO --numberOfChannels 445 --SensorThickness 120 --campaignType Production --cms NO --deliveryMonth September --deliveryYear 24
# use for 120um MGS data; add cuttype: python3 HexaBatch.py --MGS YES --cuttype B --numberOfChannels 474 --SensorThickness 120 --campaignType Production --cms NO --deliveryMonth September --deliveryYear 24
# use for 200um MGS data; add cuttype: python3 HexaBatch.py --MGS YES --cuttype C --numberOfChannels 218 --SensorThickness 200 --campaignType Production --cms NO --deliveryMonth September --deliveryYear 24
# use for 300um MGS data; add cuttype: python3 HexaBatch.py --MGS YES --cuttype C --numberOfChannels 218 --SensorThickness 300 --campaignType Production --cms NO --deliveryMonth September --deliveryYear 24

# adjust the paths
# **ToDo**: Modify the fits for partial sensors depletion voltage.  

import glob, os
import sys
import subprocess
import argparse
import shutil
import datetime
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pplt  # Rename plt to pplt
import matplotlib.colors as mcolors  # Correctly import the colors module
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import ROOT
import re 
from matplotlib.lines import Line2D
from pdf2image import convert_from_path
import matplotlib.lines as mlines
import Grading_csv_generator
import cppyy
from matplotlib.ticker import ScalarFormatter
import ctypes

parser = argparse.ArgumentParser()
parser.add_argument('--MGS', type=str, choices=['YES', 'NO'], required=True, help='Include MGS (YES or NO)')
parser.add_argument("--cuttype", type=str, help="Specify cut type (e.g., A, B)", default="A")
parser.add_argument('--numberOfChannels', type=int, help='number of channels for the sensor', default=199, required=True)
parser.add_argument('--SensorThickness', type=int, help='thickness for the sensor', default=300, required=True)
parser.add_argument('--campaignType', type=str, help='Production', default="Production", required=False)
parser.add_argument('--cms', type=str, choices=['YES', 'NO'], required=True, help='Include CMS (YES or NO)')
parser.add_argument('--deliveryMonth', type=str, required=True, help='Delivery Month')
parser.add_argument('--deliveryYear', type=int, required=True, help='Delivery Year')


args = parser.parse_args()

# Custom logic for handling cuttype if not provided
if args.cuttype is None:
    print("No cuttype specified; applying default behavior.")
    # Define the behavior when cuttype is missing
    # e.g., set a default value or skip certain actions
    args.cuttype = "default_value_or_logic"  # Replace as needed


numberOfChannels = args.numberOfChannels
SensorThickness = args.SensorThickness
campaignType = args.campaignType
deliveryMonth = args.deliveryMonth
deliveryYear = args.deliveryYear
campaign = args.campaignType

pathToCode= os.getcwd() + "/"
pathToCampaign= pathToCode + campaignType

print(f"SensorThickness: {args.SensorThickness}")

def month_name_to_number(month_name):
    try:
        # Convert the month name to a month number (e.g., "March" -> "03")
        month_number = datetime.datetime.strptime(month_name, '%B').strftime('%m')
        return month_number
    except ValueError:
        print(f"Invalid month name: {month_name}")
        return None
# Check if MGS argument is 'NO'
if args.MGS == 'NO':    
    # Check if CMS argument is 'NO'
    if args.cms == 'NO':
        # path to the blanck analysis input data of CMS
        data_IV_CMS= pathToCode + 'input_CMS/hgsensor_iv/'
        data_CV_CMS= pathToCode + 'input_CMS/hgsensor_cv/'
        dataCMS_IV = data_IV_CMS + 'temperature_scaled'
        dataCMS_CV = data_CV_CMS + 'open_corrected'
        # Create the directory if it doesn't exist
        if not os.path.exists(data_IV_CMS):
            os.makedirs(data_IV_CMS)
            print(f"Directory '{data_IV_CMS}' created.")
        else:
            print(f"Directory '{data_IV_CMS}' already exists.")
        if not os.path.exists(data_CV_CMS):
            os.makedirs(data_CV_CMS)
            print(f"Directory '{data_CV_CMS}' created.")
        else:
            print(f"Directory '{data_CV_CMS}' already exists.")
        if not os.path.exists(dataCMS_IV):
            os.makedirs(dataCMS_IV)
            print(f"Directory '{dataCMS_IV}' created.")
        else:
            print(f"Directory '{dataCMS_IV}' already exists.")
        if not os.path.exists(dataCMS_CV):
            os.makedirs(dataCMS_CV)
            print(f"Directory '{dataCMS_CV}' created.")
        else:
            print(f"Directory '{dataCMS_CV}' already exists.")


    if args.cms == 'YES':
        # path to the blanck analysis input data of CMS
        if numberOfChannels == 199:
            data_IV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_iv/Production_LD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
 
            data_CV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_cv/Production_LD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
        if numberOfChannels == 445:
            data_IV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_iv/Production_HD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
 
            data_CV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_cv/Production_HD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
        

    dataHPK_IV = os.path.join(
           pathToCampaign, 
           f"hgsensor_iv/"
        )
    delivery_month_number = month_name_to_number(deliveryMonth)
    print(" delivery_month_number     ", delivery_month_number)
    if SensorThickness == 300:
        dataHPK_CV = os.path.join(
            pathToCampaign, 
            f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/Full/S15591-01/{args.deliveryYear}{delivery_month_number}"
        )
        directory_pattern_HPK_IV = f"Hamamatsu_Production_{deliveryMonth}_20{deliveryYear}_OBA"
        pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")

    if SensorThickness == 200:
        if  numberOfChannels == 199:
            dataHPK_CV = os.path.join(
                pathToCampaign, 
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/Full/S15591-02/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_{deliveryMonth}_20{deliveryYear}_200um_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
        if  numberOfChannels == 445:
            dataHPK_CV = os.path.join(
                pathToCampaign, 
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/Full/S15591-04/{args.deliveryYear}{delivery_month_number}"
            )         
            directory_pattern_HPK_IV = f"Hamamatsu_Production_{deliveryMonth}_20{deliveryYear}_HD_200um_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
         
    if SensorThickness == 120:
        dataHPK_CV = os.path.join(
            pathToCampaign, 
            f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/Full/S15591-03/{args.deliveryYear}{delivery_month_number}"
        )
        directory_pattern_HPK_IV = f"Hamamatsu_Production_{deliveryMonth}_20{deliveryYear}_120um_OBA"
        pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")

# Check if MGS argument is 'YES'
if args.MGS == 'YES':    
    # Check if CMS argument is 'NO'
    if args.cms == 'NO':
        # path to the blanck analysis input data of CMS
        data_IV_CMS= pathToCode + 'input_CMS/hgsensor_iv/'
        data_CV_CMS= pathToCode + 'input_CMS/hgsensor_cv/'
        dataCMS_IV = data_IV_CMS + 'temperature_scaled'
        dataCMS_CV = data_CV_CMS + 'open_corrected'
        # Create the directory if it doesn't exist
        if not os.path.exists(data_IV_CMS):
            os.makedirs(data_IV_CMS)
            print(f"Directory '{data_IV_CMS}' created.")
        else:
            print(f"Directory '{data_IV_CMS}' already exists.")
        if not os.path.exists(data_CV_CMS):
            os.makedirs(data_CV_CMS)
            print(f"Directory '{data_CV_CMS}' created.")
        else:
            print(f"Directory '{data_CV_CMS}' already exists.")
        if not os.path.exists(dataCMS_IV):
            os.makedirs(dataCMS_IV)
            print(f"Directory '{dataCMS_IV}' created.")
        else:
            print(f"Directory '{dataCMS_IV}' already exists.")
        if not os.path.exists(dataCMS_CV):
            os.makedirs(dataCMS_CV)
            print(f"Directory '{dataCMS_CV}' created.")
        else:
            print(f"Directory '{dataCMS_CV}' already exists.")


    if args.cms == 'YES':
        # path to the blanck analysis input data of CMS
        if numberOfChannels == 199:
            data_IV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_iv/Production_LD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
 
            data_CV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_cv/Production_LD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
        if numberOfChannels == 445:
            data_IV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_iv/Production_HD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
 
            data_CV_CMS = os.path.join(
                pathToCampaign, 
                f"hgsensor_cv/Production_HD_{args.SensorThickness}_{args.deliveryMonth}_20{args.deliveryYear}"
            )
        
    ############  now HPK paths ######
    ############  now HPK paths ######
    dataHPK_IV = os.path.join(
        pathToCampaign, 
        "hgsensor_iv/"
    )
    delivery_month_number = month_name_to_number(deliveryMonth)
    print("delivery_month_number", delivery_month_number)
                
# Start of the block for SensorThickness == 300
    if SensorThickness == 300:
    
    # # Use the `cuttype` argument to select the path
        if args.cuttype == "A":
            dataHPK_CV = os.path.join(
             pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-01A/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_01A_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        elif args.cuttype == "B":
            dataHPK_CV = os.path.join(
                pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-01B/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_01B_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        elif args.cuttype == "C":
            dataHPK_CV = os.path.join(
                pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-01C/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_01C_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        else:
            print("Invalid cuttype. Please select 'A', 'B', or 'C'.")



    if SensorThickness == 200:
        
        if args.cuttype == "A":
            dataHPK_CV = os.path.join(
             pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-02A/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_02A_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        elif args.cuttype == "B":
            dataHPK_CV = os.path.join(
                pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-02B/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_02B_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        elif args.cuttype == "C":
            dataHPK_CV = os.path.join(
                pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-02C/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_02C_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        else:
            print("Invalid cuttype. Please select 'A', 'B', or 'C'.")
     
           
    if SensorThickness == 120:
        if args.cuttype == "A":
            dataHPK_CV = os.path.join(
             pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-03A/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_03A_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        elif args.cuttype == "B":
            dataHPK_CV = os.path.join(
                pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-03B/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_03B_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        elif args.cuttype == "C":
            dataHPK_CV = os.path.join(
                pathToCampaign,
                f"/eos/user/h/hgsensor/HGCAL_test_results/Results/Hamamatsu/HPK_Upload/MGS/S17278-03C/{args.deliveryYear}{delivery_month_number}"
            )
            directory_pattern_HPK_IV = f"Hamamatsu_Production_MGS_{args.deliveryMonth}_20{args.deliveryYear}_03C_OBA"
            pattern = re.compile(f"{directory_pattern_HPK_IV}[0-9]{{5}}$")
    
        else:
            print("Invalid cuttype. Please select 'A', 'B', or 'C'.")



        
# path to the raw input data of HPK.
pathToIVResults = pathToCode + 'IV_Results/'
pathToCVResults = pathToCode + 'CV_Results/'
pathToIVFailedSensors = pathToCode + 'IV_failed_HPK/'
pathToIVFailedSensors_CMS = pathToCode + 'IV_failed_CMS/'
pathToCVFailedSensors = pathToCode + 'CV_failed_HPK/'
pathToCVFailedSensors_CMS = pathToCode + 'CV_failed_CMS/'
pathToResults = pathToCVResults + 'DepletionVoltageHPK/'
pathToResultsCMS = pathToCVResults + 'DepletionVoltageCMS/'

dataCMS_CV_Vdep = data_CV_CMS + 'Vdep'



def check_and_create_directory(directory):
    """
    Checks if a directory exists, and if it does not, creates it.

    Args:
    directory (str): The path of the directory to check and create.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
         #print(f"Directory '{directory}' was created.")
    else:
        pass

 #check_and_create_directory(data_IV_CMS)
 #check_and_create_directory(data_CV_CMS)


check_and_create_directory(dataCMS_CV_Vdep)

 #check_and_create_directory(dataHPK_IV)
 #check_and_create_directory(dataHPK_CV)
check_and_create_directory(pathToIVResults)
check_and_create_directory(pathToCVResults)
check_and_create_directory(pathToResults)
check_and_create_directory(pathToResultsCMS)
check_and_create_directory(pathToIVFailedSensors)
check_and_create_directory(pathToIVFailedSensors_CMS)
check_and_create_directory(pathToCVFailedSensors)
check_and_create_directory(pathToCVFailedSensors_CMS)

confidenceLevelForChi2=0.05
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)



def calculateChi2(x, y):
    slope, intercept, r_value, p_Value, std_err = stats.linregress(x, y)
    x_reshaped = x.reshape((-1, 1))   
    model = LinearRegression().fit(x_reshaped, y)    
    y_pred = model.intercept_ + model.coef_ * x_reshaped    
    chi2_stat = np.sum((y - np.transpose(y_pred))**2)
    df= len(y)-2
    chi2_quality=chi2_stat/(std_err)/df
    chi2_quality_alternative=np.sum((y - np.transpose(y_pred))**2/(y_pred))/df
    
    return [slope, intercept, chi2_quality, chi2_quality_alternative]


# ============================================================================
    # """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.plot(xLeftSide, slopeLeft*xLeftSide + interceptLeft)
    ax.plot(xRightSide, slopeRight*xRightSide + interceptRight)
    dataForSingleChannel.plot(kind='scatter',x='Voltage',y='C_serial_open_corrected', c='g', ax=ax)
    plt.text(100, 0.00002, 'estimated depletion voltage: ' + str(round(Vdep,1)))
    plt.title('Channel '+ str(channelNumber) +', '+ sensorName)
    plt.xlabel('Effective bias voltage [V]')
    plt.ylabel('Open-corrected C2 serial [1/pF2]')
    pathToSingleDepletionVoltagesPlosts = pathToResultsCMS
    existsFolder = os.path.exists(pathToSingleDepletionVoltagesPlosts)
    # if not existsFolder:
    #     os. makedirs(pathToSingleDepletionVoltagesPlosts)
    plt.savefig(pathToSingleDepletionVoltagesPlosts  + sensorName + '_Channel' + str(channelNumber) + '.png', dpi=300)
    plt.close()
    # """

# =============================================================================
    return Vdep

def findDepletionVoltageHPK(CVDataHPK, sensor_name, pathToResults, foundry):
    """
    Find depletion voltage for the given data and sensor name.
    
    Args:
        CVDataHPK (DataFrame): Data for single channel.
        sensor_name (str): Name of the sensor.
        path_to_results (str): Path to save results.
        
    Returns:
        tuple: Depletion voltage and its uncertainty.
    """
  
    thickness, quality, coverage = getSensorDetails(sensor_name)
    thickness = args.SensorThickness
    if thickness is None:
        raise ValueError("Invalid sensor name or unknown sensor details")

    try:
        CVDataHPK = CVDataHPK.astype(float).abs()
        CVDataHPK['Capacitance'] = 1 / (CVDataHPK['Capacitance'])**2
        CVDataHPK = CVDataHPK.sort_values(by='Voltage')
    except Exception as e:
        print("Error processing data:", e)
        return None, None
    
    print("          thickness ---------------_____>", thickness)
    # Adapt fit range to sensor thickness
    if thickness == 300:
        lin_low, lin_high, const_low, const_high = 160, 250, 325, 400
    elif thickness == 200:
        lin_low, lin_high, const_low, const_high = 60, 110, 200, 400
    elif thickness == 120:
        lin_low, lin_high, const_low, const_high = 0, 20, 30, 60
    else:
        raise Exception('Invalid thickness')

    f_lin = ROOT.TF1('Lin', 'pol1', lin_low, lin_high)
    f_lin.SetLineColor(ROOT.kBlue)
    f_const = ROOT.TF1('Const', 'pol0', const_low, const_high)
    f_const.SetLineColor(ROOT.kRed)

     #print("f_lin parameter 0  in findDepletionVoltageHPK:", f_lin.GetParameter(0))
     #print("f_lin parameter 1  in findDepletionVoltageHPK:", f_lin.GetParameter(1))
     #print("f_const parameter 0  in findDepletionVoltageHPK:", f_const.GetParameter(0))


    # Initialize the parameters for each fit before it is used
    if thickness == 300:
        f_const.SetRange(250, 350) # Set the range for constant fit
        f_lin.SetRange(160, 210)      # Set the range for linear fit
        #f_const.SetRange(220, 350) # Set the range for constant fit
        #f_lin.SetRange(50, 100)      # Set the range for linear fit
 
        f_lin.SetParameters(0.0, 0.001)  # Initialize linear fit parameters
        f_const.SetParameter(0, 0.116)   # Initialize constant fit parameter
    elif thickness == 200:
        f_const.SetRange(220, 350) # Set the range for constant fit
        f_lin.SetRange(50, 100)      # Set the range for linear fit
 
        f_lin.SetParameters(0.0, 0.001)  # Initialize linear fit parameters
        f_const.SetParameter(0, 0.116)   # Initialize constant fit parameter
    elif thickness == 120:
        f_const.SetRange(30, 60) # Set the range for constant fit
        f_lin.SetRange(0, 20)      # Set the range for linear fit
 
        f_lin.SetParameters(0.0, 0.001)  # Initialize linear fit parameters
        f_const.SetParameter(0, 0.116)   # Initialize constant fit parameter

  
    # Call plot_and_fit function to plot and fit the graph
    Vdep, eVdep = plot_and_fit(CVDataHPK, f_lin, f_const, sensor_name, pathToResults,foundry)
    print("  Vdep      in findDepletionVoltageHPK  ", Vdep)
    print("  eVdep      in findDepletionVoltageHPK  ", eVdep)
    
    return Vdep, eVdep

def save_pdf(canvas, file_path):
    try:
        # Save canvas as a PDF
        canvas.SaveAs(file_path)

        return True
    except Exception as e:
        print("Error saving PDF file:", e)
        return False
    



def readDataFile(datafile):
    dataFrame = pd.read_csv(datafile, delim_whitespace=True, engine = 'python')
    return dataFrame

def readDataFileHPK(datafile):
    dataFrame = pd.read_csv(datafile, delim_whitespace=True, engine = 'python')
     #dataFrame.drop(dataFrame.columns[6:], axis=1, inplace=True)
     #dataFrame.drop(dataFrame.columns[2], axis=1, inplace=True)
     #dataFrame.drop(dataFrame.columns[3], axis=1, inplace=True)
    return dataFrame


def readDataFileHPK_CV(datafile):
    dataFrame = pd.read_csv(datafile, header=None, sep = '\t', engine = 'python', encoding='latin1')
    header = dataFrame.iloc[:18]
    thickness = header.iloc[4][1][:3]
    dataFrame = dataFrame.iloc[18:]
    dataFrame.drop(dataFrame.tail(2).index,inplace=True)
    dataFrame.columns = ["Voltage", "Channel", "Capacitance"]
    return [dataFrame, thickness]

def getSensornameIV(fileName):
    sensorName = fileName[0:6]
    return sensorName

def getSensornameIVHPK(fileName):
    sensorName = fileName[0:6]
    return sensorName


def getSensornameCVHPK(fileName):
         sensorName = fileName[23:29]
         return sensorName


def getSensornameCVCMS(fileName):
    sensorName = fileName[0:6]
    return sensorName

def output(deliveryMonth, deliveryYear):
    return {
        "pdf": f"IVCV_{deliveryMonth}_{deliveryYear}.pdf",
        "elog": f"elog_IVCV_{deliveryMonth}_{deliveryYear}.txt"
    }


def custom_sorting_key(file_name):
    if file_name.endswith("_std_uncorrected_0.png"):
        return 1
    elif file_name.endswith("_allchannels_IV_0.png"):
        return 2
    elif file_name.endswith("_total_current_IV_0.png"):
        return 3
    else:
        return 0  # Default case

def create_pdf_presentation(tex_file_name, pdf_file_name):
        thickness_OBA = SensorThickness
        listOfCellCurrentFiles=os.listdir(pathToIVResults)
        listOfCellCVFiles=os.listdir(pathToCVResults)

        main_tex_tex = open(pathToCode+'main.tex', "r").read()
        main_tex_tex = main_tex_tex.replace("<BATCHNUMBER>", str(amountOfOBA))  # Convert to string
        main_tex_tex = main_tex_tex.replace("<SENSORNUMBER>", str(amountOfTotalSensor))  # Convert to string
        amountOfTotalSensor_CMS = 0
        escaped_campaignType = campaignType.replace("_", "\\_")
        escaped_deliveryMonth = deliveryMonth.replace("_", "\\_")
        escaped_deliveryYear = str(deliveryYear).replace("_", "\\_")
        main_tex_tex = main_tex_tex.replace("<CAMPAIGN>", escaped_campaignType + ", " + escaped_deliveryMonth + escaped_deliveryYear)

        if numberOfChannels < 200:
            main_tex_tex = main_tex_tex.replace("<MEDIUM>", "Low")
        else:
            main_tex_tex = main_tex_tex.replace("<MEDIUM>", "High")           
        sorted_listOfCellCurrentFiles = sorted(listOfCellCurrentFiles, key=lambda x: int(re.search(r"OBA(\d+)", x).group(1)) if re.search(r"OBA(\d+)", x) is not None else float('inf'))
        sorted_listOfCellCVFiles = sorted(listOfCellCVFiles, key=lambda x: int(re.search(r"OBA(\d+)", x).group(1)) if re.search(r"OBA(\d+)", x) is not None else float('inf'))
        cell_current_tex = ""
        cell_cv_tex = ""
        table_tex = ""
        failed_tex = ""
        failed_tex_CMS = ""
        failed_tex_CV = ""
        failed_tex_CMS_CV = ""

        grading_tex = ""
        grading_tex_CMS = ""
        table_rows_left = ""
        table_rows_right = ""
        
        counter = 0
        amountOfFailedlSensor=0
        amountOfFailedlSensor_CMS=0
        amountOfFailedlSensor_CV=0
        amountOfFailedlSensor_CMS_CV=0
        oba_numbers = []
        num_sensors = []
        amountOfSensorsAtCMS=0
        png_files_CMS = []
        sensor_files_CMS = {}
        passedGradingCMS  = []
        failedGradingCMS  = []
        tot_hot_channel_count = 0


        hot_cell_currents_600V = []
        hot_cell_currents_800V = []
        hot_cell_currents_bad_600V = []
        hot_cell_currents_bad_800V = []

        hot_cell_counts = {}
        hot_cell_counts_800V =  {}
        # Initialize the dictionaries with 0 for all sensors
        # Check if channel_NonZero_DataFrame is empty before accessing its columns
        if 'Sensor name' in channel_NonZero_DataFrame.columns:
            sensor_hot_cell_nonzero_counts = {sensor: 0 for sensor in channel_NonZero_DataFrame['Sensor name'].unique()}
        else:
            sensor_hot_cell_nonzero_counts = {}

            # Continue with your code using sensor_hot_cell_nonzero_counts
            # ...
        # Filter the DataFrame for rows where 'foundry' is 'Hamamatsu'
 

        sensor_hot_cell_counts = {sensor: 0 for sensor in channel_DataFrame['Sensor name_600V'].unique()}
        sensor_hot_cell_counts_800V = {sensor: 0 for sensor in channel_DataFrame['Sensor name_600V'].unique()}
         # # #sensor_hot_cell_counts = {}
         # # #sensor_hot_cell_counts_800V = {}

        # Create an empty DataFrame to store selected channels
        selected_channels_df = pd.DataFrame(columns=['Sensor name_600V', 'Channel', 'Cell current [nA]_600V', 'Cell current [nA]_800V', 'Currents_ratio'])

        # loop over the list of OBA's in the directory of IV_Results
        for file in sorted_listOfCellCurrentFiles:
            if file == ".DS_Store":
                continue

            if os.path.isdir(pathToIVResults+file):
                continue
            counter += 1
            OBA_number = 'OBA' + file.split("OBA")[1][:5]
               #grading_OBA =  grading_DataFrame_AllOBA[grading_DataFrame_AllOBA['OBA'] == OBA_number]
            grading_OBA = grading_DataFrame_AllOBA[(grading_DataFrame_AllOBA['OBA'] == OBA_number) & 
                                       (grading_DataFrame_AllOBA['Foundry'] == 'Hamamatsu')]
            grading_OBA_CMS = grading_DataFrame_AllOBA[(grading_DataFrame_AllOBA['OBA'] == OBA_number) & 
                                       (grading_DataFrame_AllOBA['Foundry'] == 'CMS')]
            
            grading_OBA_CV = grading_DataFrame_AllOBA_CV[(grading_DataFrame_AllOBA_CV['OBA'] == OBA_number) & 
                                       (grading_DataFrame_AllOBA_CV['Foundry'] == 'Hamamatsu')]
            grading_OBA_CMS_CV = grading_DataFrame_AllOBA_CV[(grading_DataFrame_AllOBA_CV['OBA'] == OBA_number) & 
                                       (grading_DataFrame_AllOBA_CV['Foundry'] == 'CMS')]

             #print("          grading_OBA_CMS_CV         !!!!!!!!!!!!!!!!!!")
             #print(grading_OBA_CMS_CV)
            
            thickness_OBA = SensorThickness
            # thickness_OBA = int(grading_DataFrame_AllOBA['thickness'].iloc[0])
            amountOfSensorsAtHamamatsu = len(grading_OBA)
            amountOfSensorsAtCMS = len(grading_OBA_CMS)
            print("       amountOfSensorsAtCMS ----------->", amountOfSensorsAtCMS)
            if amountOfSensorsAtCMS > 0:
                passedGradingCMS = grading_OBA_CMS[grading_OBA_CMS['Overall grading'] == 'Passed']
                passedCMS = len(passedGradingCMS)
                failedGradingCMS = grading_OBA_CMS[grading_OBA_CMS['Overall grading'] == 'Failed']
                failedCMS = len(failedGradingCMS)
                amountOfTotalSensor_CMS = amountOfTotalSensor_CMS + amountOfSensorsAtCMS

                passedGradingCMS_CV = grading_OBA_CMS_CV[grading_OBA_CMS_CV['Overall grading'] == 'Passed']
                passedCMS_CV = len(passedGradingCMS_CV)
                failedGradingCMS_CV = grading_OBA_CMS_CV[grading_OBA_CMS_CV['Overall grading'] == 'Failed']
                failedCMS_CV = len(failedGradingCMS_CV)
                 #print("         (failedGradingCMS_CV     !!!!!!!!!!!!!!!!!!")
                 #print(failedGradingCMS_CV)
                 #print("         (failedCMS_CV    ==============", failedCMS_CV)

            else:
                passedCMS = 0
                failedCMS = 0
                passedCMS_CV = 0
                failedCMS_CV = 0
                 
            passedGradingHPK = grading_OBA[grading_OBA['Overall grading'] == 'Passed']
            failedGradingHPK = grading_OBA[grading_OBA['Overall grading'] == 'Failed']
            failedHPKK = len(failedGradingHPK)

            passedGradingHPK_CV = grading_OBA_CV[grading_OBA_CV['Overall grading'] == 'Passed']
            failedGradingHPK_CV = grading_OBA_CV[grading_OBA_CV['Overall grading'] == 'Failed']
            failedHPKK_CV = len(failedGradingHPK_CV)

            
            
            amountOfFailedlSensor = amountOfFailedlSensor + int(failedHPKK)
            amountOfFailedlSensor_CV = amountOfFailedlSensor_CV + int(failedHPKK_CV)
            
            amountOfFailedlSensor_CMS = amountOfFailedlSensor_CMS + int(failedCMS)
            amountOfFailedlSensor_CMS_CV = amountOfFailedlSensor_CMS + int(failedCMS_CV)
            
            amountPassedHPK = int(amountOfSensorsAtHamamatsu) - int(failedHPKK)

            amountPassedHPK_CV =  int(amountOfSensorsAtHamamatsu) - int(failedHPKK_CV)
            
            num_sensors.append(amountOfSensorsAtHamamatsu)
            half_amountOfOBA = int(amountOfOBA / 2)
 
            if counter <= half_amountOfOBA:
                if amountOfSensorsAtHamamatsu < 20:
                    if failedHPKK > 0:
                        if failedCMS > 0:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/\\textcolor{{red}}{{{failedCMS}}} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/\\textcolor{{red}}{{{failedCMS_CV}}}  \\\\\n"
                        else:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/{failedCMS}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/{failedCMS_CV} \\\\\n"
                    else:
                        if failedCMS > 0:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/\\textcolor{{red}}{{{failedCMS}}} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/\\textcolor{{red}}{{{failedCMS_CV}}}  \\\\\n"
                        else:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/{failedCMS}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/{failedCMS_CV} \\\\\n"

                else:
                    if failedHPKK > 0:
                        if failedCMS > 0:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/\\textcolor{{red}}{{{failedCMS}}}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/\\textcolor{{red}}{{{failedCMS_CV}}}  \\\\\n"
                        else:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/{failedCMS}   & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/{failedCMS_CV}\\\\\n"
                    else:
                        if failedCMS > 0:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/\\textcolor{{red}}{{{failedCMS}}} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/\\textcolor{{red}}{{{failedCMS_CV}}} \\\\\n"
                        else:
                            table_rows_left += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/{failedCMS}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/{failedCMS_CV} \\\\\n"
                    
            else:
                if amountOfSensorsAtHamamatsu < 20:
                    if failedHPKK > 0:
                        if failedCMS > 0:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/\\textcolor{{red}}{{{failedCMS}}}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/\\textcolor{{red}}{{{failedCMS_CV}}} \\\\\n"
                        else:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/{failedCMS}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/{failedCMS_CV}\\\\\n"
                    else:
                        if failedCMS > 0:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/\\textcolor{{red}}{{{failedCMS}}} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/\\textcolor{{red}}{{{failedCMS_CV}}}  \\\\\n"
                        else:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize \\textcolor{{blue}}{{{amountOfSensorsAtHamamatsu}}}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/{failedCMS} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/{failedCMS_CV} \\\\\n"
                else:
                    if failedHPKK > 0:
                        if failedCMS > 0:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/\\textcolor{{red}}{{{failedCMS}}}  & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/\\textcolor{{red}}{{{failedCMS_CV}}}  \\\\\n"
                        else:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize \\textcolor{{red}}{{{failedHPKK}}}/{failedCMS} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize \\textcolor{{red}}{{{failedHPKK_CV}}}/{failedCMS_CV}  \\\\\n"
                    else:
                        if failedCMS > 0:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/\\textcolor{{red}}{{{failedCMS}}} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/\\textcolor{{red}}{{{failedCMS_CV}}}  \\\\\n"
                        else:
                            table_rows_right += f"\\scriptsize {OBA_number} & \\scriptsize {amountOfSensorsAtHamamatsu}/\\textcolor{{blue}}{{{amountOfSensorsAtCMS}}} & \\scriptsize {amountPassedHPK}/{passedCMS} & \\scriptsize {failedHPKK}/{failedCMS} & \\scriptsize {amountPassedHPK_CV}/{passedCMS_CV} & \\scriptsize {failedHPKK_CV}/{failedCMS_CV}  \\\\\n"
          



                        
            
            cell_current_tex += f"\\begin{{frame}}\n"
            cell_current_tex += f"\\frametitle{{IV: HPK / CMS; {thickness_OBA}$\\mu$m {OBA_number}}}\n"
            cell_current_tex += f"\\begin{{figure}}[!ht]\n"
            cell_current_tex += f"\\includegraphics[width=.9\\textwidth]{{{pathToIVResults}{file}}}\n"
            cell_current_tex += f"\\end{{figure}}\n"

               #print(" !!!!!!!!!!! channel_DataFrame !!!!!!!!!!")
            
               #print(channel_DataFrame)
            
            # # # # # # # I600 > 1000 nA 
            hot_channels_OBA_data = channel_DataFrame[channel_DataFrame['OBA_number'] == OBA_number]

            hot_channels_OBA_NonZero_data = channel_NonZero_DataFrame[channel_NonZero_DataFrame['OBA_number'] == OBA_number]
            
            selected_channels_label_for_hot_channels_OBA = ", ".join(map(str, hot_channels_OBA_data['Channel']))
            selected_sensors_dict = {}
            current_ratios = 0.0

            ############
            # select hot cells
            for index, row in hot_channels_OBA_NonZero_data.iterrows():
                sensor_name = row['Sensor name']
                hot_cell_name = row['Channel']

                if sensor_name in selected_sensors_dict:
                    selected_sensors_dict[sensor_name].append(str(row['Channel']))
                else:
                    selected_sensors_dict[sensor_name] = [str(row['Channel'])]
                    
             #  Loop over channels
            hot_cell_current_600V = 0.0
            hot_cell_current_800V = 0.0
            # Dictionary to store sets of counted hot cell names for each sensor
            processed_channels = {}
            # Drop any duplicate rows just in case
            hot_channels_OBA_data = hot_channels_OBA_data.drop_duplicates()
           
            for index, row in hot_channels_OBA_data.iterrows():
                sensor_name = row['Sensor name_600V']
                hot_cell_name = row['Channel']
                hot_cell_current_600V = row['Cell current [nA]_600V']
                hot_cell_current_800V = row['Cell current [nA]_800V']
                current_ratios = row['Currents_ratio']

                # Initialize the set for the sensor if it hasn't been processed before
                
                if sensor_name not in processed_channels:
                    processed_channels[sensor_name] = set()

                # 1. criterion I600 > 100 nA
                if hot_cell_current_600V > 100 and hot_cell_name not in processed_channels[sensor_name]:
  
                    sensor_hot_cell_counts[sensor_name] += 1
                    if  sensor_hot_cell_counts[sensor_name] > 1:
                        print(f"   MORE THAN 1  I600 > 100 nA =======================!!!!!!!!!!!! {sensor_hot_cell_counts[sensor_name]}")
                        print(" sensor_name   =======================>", sensor_name)
                        print(" hot_cell_name   =======================>", hot_cell_name)
                        print(" hot_cell_current_600V  =======================>", hot_cell_current_600V)
                        print(" hot_cell_current_800V  =======================>", hot_cell_current_800V)
                        
                    hot_cell_currents_600V.append(hot_cell_current_600V)
                    hot_cell_currents_800V.append(hot_cell_current_800V)
                    # Mark the channel as processed for this sensor
                    processed_channels[sensor_name].add(hot_cell_name)
        
                # 2. criterion# # # ## I600 > 1000 nA OR I800 > 2.5 I600
        

            # Initialize an empty list to store hot cell currents at 600V


                if  ((current_ratios > 2.6) & (hot_cell_current_600V > 10))  | ((hot_cell_current_600V <= 10) & (hot_cell_current_800V > 25)):
                    sensor_hot_cell_counts_800V[sensor_name] += 1
                    hot_cell_currents_bad_600V.append(hot_cell_current_600V)
                    hot_cell_currents_bad_800V.append(hot_cell_current_800V)
                  
                  
                    selected_channels_df = selected_channels_df._append({
                        'Sensor name_600V': sensor_name,
                        'Channel': hot_cell_name,
                        'Cell current [nA]_600V': hot_cell_current_600V,
                        'Cell current [nA]_800V': hot_cell_current_800V,
                        'Currents_ratio': current_ratios
                    }, ignore_index=True)

             
            selected_sensors_info = "\n".join([f"Sensor: {sensor} --> Cell: {', '.join(cells)}" for sensor, cells in selected_sensors_dict.items()])
            cell_current_tex += f"\\vspace{{0.01cm}} % Adjust the space as needed\n"
            cell_current_tex += f"\\begin{{minipage}}[t][2cm][t]{{0.95\\textwidth}} % Adjust the height to your needs\n"
            cell_current_tex += f"\\fontsize{{4}}{{6}}\\selectfont\n"
            cell_current_tex += f"\\raggedright\n"
            cell_current_tex += f"Hot cells at 600 V (HPK): {selected_sensors_info}\n"
            cell_current_tex += f"\\end{{minipage}}\n"
            cell_current_tex += f"\\end{{frame}}\n"
        #
        # END of OBA loop For IV files

        main_tex_tex = main_tex_tex.replace("<SENSORNUMBER_CMS>", str(amountOfTotalSensor_CMS))  # Convert to string
        if len(num_sensors) > 0:
            hist, bins = np.histogram(num_sensors, bins=range(0, max(num_sensors) + 2))
        else:
            print("num_sensors is empty. Skipping histogram creation.")
            hist, bins = [], []

        pplt.figure(figsize=(12, 9))
        pplt.bar(bins[:-1], hist, color='blue', edgecolor='black', alpha=0.7)
    
        pplt.xlabel('Number of Sensors')
        pplt.ylabel('Number of Batches')
        pplt.title('Distribution of Number of Sensors Across Batches')
        pplt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        pplt.grid(axis='y')
        pplt.tight_layout()
        pplt_pdf_filename = "plot_OBA_sensor.pdf"
        pplt.savefig(pathToCode + pplt_pdf_filename)
        pplt.close()  # Close the figure to release resources
               #plt.savefig(pathToCode + 'OBA_number_vs_number_of_sensors_' + campaignType + "_" + str(plot) + '.pdf', format='pdf')

        main_tex_tex = main_tex_tex.replace("<CAMPAIGN>", campaignType)
        main_tex_tex = main_tex_tex.replace("<PLOT_OBA_SENSOR>",  pathToCode + pplt_pdf_filename)

         #plt_pdf_filename = "default_plot.pdf"

    
        # Plot bad sensor
        hot_cell_counts = list(sensor_hot_cell_counts.values())
        hot_cell_counts_800V = list(sensor_hot_cell_counts_800V.values())

        if hot_cell_counts:
            # Prepare data for plotting
            # Create the plot for  1. criterion
            max_bin = 8  # Explicitly set the maximum bin to 8
            bins = np.arange(0, max_bin + 2) 
            hist, bins = np.histogram(hot_cell_counts, bins=bins)

            plt.figure(figsize=(12, 9))
            plt.bar(bins[:-1], hist, color='blue', edgecolor='black', alpha=0.7)

            
            # Adding text labels on top of each bin
            for i in range(len(hist)):
                plt.text(bins[i], hist[i], str(hist[i]), ha='center', va='bottom', fontsize=12, color='black')

            plt.yscale('log')  # Set y-axis to log scale

            # Ensure no zero values for log scale
            if np.any(hist <= 0):
                hist[hist == 0] = 1e-10  # Small value to avoid log(0) issues

            plt.xlabel('Number of Bad Cells', fontsize=16)
            plt.ylabel('Number of Sensors', fontsize=16)
            plt.title('Distribution of Hot Cells per Sensor (I600>100nA)', fontsize=16)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt_pdf_filename = "plot_bad_channel_dist_1.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources
        else:
            # Create an empty plot with the same structure
            max_bin = 8  # Explicitly set the maximum bin to 8
            bins = np.arange(0, max_bin + 2) 
            hist = np.zeros(len(bins) - 1)  # Empty histogram

            plt.figure(figsize=(12, 9))
            plt.bar(bins[:-1], hist, color='blue', edgecolor='black', alpha=0.7)

            # Adding text labels on top of each bin (which will be zero)
            for i in range(len(hist)):
                plt.text(bins[i], hist[i], str(int(hist[i])), ha='center', va='bottom', fontsize=12, color='black')

            plt.yscale('log')  # Set y-axis to log scale
            # Ensure no zero values for log scale
            if np.any(hist <= 0):
                hist[hist == 0] = 1e-10  # Small value to avoid log(0) issues

            plt.xlabel('Number of Bad Cells', fontsize=16)
            plt.ylabel('Number of Sensors', fontsize=16)
            plt.title('Distribution of Hot Cells per Sensor (I600>100nA)', fontsize=16)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt_pdf_filename = "plot_bad_channel_dist_1.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources

        main_tex_tex = main_tex_tex.replace("<PLOT_DIST_1>",  pathToCode + plt_pdf_filename)

        if hot_cell_counts_800V:
            # Create the plot for  1. OR 2. criteria
            #max_count = max(hot_cell_counts_800V)
            #bins = range(0, max_count + 2)  # Adding +2 to include the max value and ensure bin alignment
            max_bin = 8 
            bins = np.arange(0, max_bin + 2) 
            hist, bins = np.histogram(hot_cell_counts_800V, bins=bins)
            plt.figure(figsize=(12, 9))
            plt.bar(bins[:-1], hist, color='blue', edgecolor='black', alpha=0.7)
            # Ensure no zero values for log scale
            if np.any(hist <= 0):
                hist[hist == 0] = 1e-10  # Small value to avoid log(0) issues

            # Adding text labels on top of each bin
            for i in range(len(hist)):
                plt.text(bins[i], hist[i], str(hist[i]), ha='center', va='bottom', fontsize=12, color='black')

            plt.yscale('log')  # Set y-axis to log scale
            plt.xlabel('Number of Bad Cells', fontsize=16)
            plt.ylabel('Number of Sensors', fontsize=16)
            plt.title('Bad Cells per Sensor: (I600 > 10 nA AND I800>2.5xI600)||(I600 ≤ 10 nA AND I800>25 nA)', fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt_pdf_filename = "plot_bad_channel_dist_2.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources
        else:
            # Create an empty plot with the same structure
            max_bin = 8 
            bins = np.arange(0, max_bin + 2) 
            hist = np.zeros(len(bins) - 1)  # Empty histogram
            # Ensure no zero values for log scale
            if np.any(hist <= 0):
                hist[hist == 0] = 1e-10  # Small value to avoid log(0) issues
            
            plt.figure(figsize=(12, 9))
            plt.bar(bins[:-1], hist, color='blue', edgecolor='black', alpha=0.7)
            # Adding text labels on top of each bin
            for i in range(len(hist)):
                plt.text(bins[i], hist[i], str(hist[i]), ha='center', va='bottom', fontsize=12, color='black')

            plt.yscale('log')  # Set y-axis to log scale
            plt.xlabel('Number of Bad Cells', fontsize=16)
            plt.ylabel('Number of Sensors', fontsize=16)
            plt.title('Bad Cells per Sensor: (I600 > 10 nA AND I800>2.5xI600)||(I600 ≤ 10 nA AND I800>25 nA)', fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt_pdf_filename = "plot_bad_channel_dist_2.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources
            
        main_tex_tex = main_tex_tex.replace("<PLOT_DIST_2>",  pathToCode + plt_pdf_filename)

        # Plotting the distribution of hot cell currents I600>100nA  at 600V
        if hot_cell_currents_600V:
            max_count = int(max(hot_cell_currents_600V))  # Convert the max value to an integer
              #bins = range(0, max_count + 2)  # Adding +2 to include the max value and ensure bin alignment
              #hist, bins = np.histogram(hot_cell_currents, bins=range(0, max(num_sensors) + 2))
            bins = np.logspace(np.log10(min(hot_cell_currents_600V)), np.log10(max(hot_cell_currents_600V)), num=80)
            plt.figure(figsize=(12, 9))

            plt.hist(hot_cell_currents_600V, bins=bins, edgecolor='black', log=True)  # log scale on y-axis
            plt.xlabel('Current [nA]', fontsize=16)
            plt.ylabel('Number of Cells (log scale)', fontsize=16)
            plt.title('Hot Cell Currents at 600 V (I600>100nA)', fontsize=16)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # Set x-axis to log scale
            plt.gca().set_xscale('log')

            plt_pdf_filename = "plot_bad_channel_current.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close() # Close the figure to release resources
        else:
            # Create an empty plot with the same structure
            bins = np.logspace(0, 1, num=80)  # Log space bins for empty plot
            hist = np.zeros(len(bins) - 1)  # Empty histogram

            plt.figure(figsize=(12, 9))
            plt.hist([], bins=bins, edgecolor='black', log=True)  # No data, but keep the log scale on y-axis
            plt.xlabel('Current [nA]', fontsize=16)
            plt.ylabel('Number of Cells (log scale)', fontsize=16)
            plt.title('Hot Cell Currents at 600 V (I600>100nA)', fontsize=16)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # Set x-axis to log scale
            plt.gca().set_xscale('log')
            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
            plt.gca().yaxis.get_major_formatter().set_scientific(False)  # Turn off scientific notation
            plt.gca().yaxis.get_major_formatter().set_useOffset(False)   # Do not use offset notation

            plt_pdf_filename = "plot_bad_channel_current.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resource
            print("No hot cell currents found.")
  
        main_tex_tex = main_tex_tex.replace("<PLOT_CURRENT_1>",  pathToCode + plt_pdf_filename)



        ##################
       # Check if both hot_cell_currents and hot_cell_currents_800V are not empty
        if hot_cell_currents_600V and hot_cell_currents_800V and len(hot_cell_currents_600V) == len(hot_cell_currents_800V):
            plt.figure(figsize=(12, 9))
            # Remove zero or negative values to avoid issues with LogNorm
            hot_cell_currents_600V = np.array(hot_cell_currents_600V)
            hot_cell_currents_800V = np.array(hot_cell_currents_800V)
            # Ensure all values are positive for log scale
            hot_cell_currents_600V = hot_cell_currents_600V[hot_cell_currents_600V > 0]
            hot_cell_currents_800V = hot_cell_currents_800V[hot_cell_currents_800V > 0]

            # Create 2D histogram
             #hist, xedges, yedges, im = plt.hist2d(hot_cell_currents_600V, hot_cell_currents_800V, bins=600, norm=mcolors.LogNorm())
            # Create 2D histogram with manual vmin and vmax settings
            hist, xedges, yedges, im = plt.hist2d(
                hot_cell_currents_600V,
                hot_cell_currents_800V,
                bins=600,
                norm=mcolors.LogNorm(vmin=1e-1, vmax=1e3)  # Set vmin and vmax
            )

            plt.colorbar(im, label='Number of Cells')  # Add colorbar
            plt.xlabel('Current at 600V [nA]', fontsize=16)
            plt.ylabel('Current at 800V [nA]', fontsize=16)
            plt.title('2D Histogram of Hot Cell Currents  (I600 > 100 nA)', fontsize=16)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xscale('log')
            plt.yscale('log')
            # Set limits to start from a small positive value close to zero
            plt.xlim(left=1e2, right=max(hot_cell_currents_800V))
            plt.ylim(bottom=1e2, top=max(hot_cell_currents_800V))
            # Ensure no zero values for log scale
            if np.any(hist <= 0):
                hist[hist == 0] = 1e-10  # Small value to avoid log(0) issues

            # Draw the line y = 2.5x in log scale
            x = np.logspace(np.log10(1e2), np.log10(max(hot_cell_currents_800V)), 1000)
            y = 2.5 * x
            plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='y=2.5x')

            plt.legend(fontsize=16)  # Add legend
            plt_pdf_filename = "plot_current_comparison_2D.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources
            main_tex_tex = main_tex_tex.replace("<PLOT_CURRENT_COMPARISON>", pathToCode + plt_pdf_filename)

        else:
            # Create an empty 2D plot with the same structure
            plt.figure(figsize=(12, 9))
            # Manually create empty data for histogram with small positive values
            empty_data_x = np.array([1e-1])  # Small positive value to avoid log(0) issues
            empty_data_y = np.array([1e-1])  # Small positive value to avoid log(0) issues

            # Create empty 2D histogram with fixed limits and manual vmin and vmax
            hist, xedges, yedges, im = plt.hist2d(empty_data_x, empty_data_y, bins=600, norm=mcolors.LogNorm(vmin=1e-1, vmax=1e3))

            # plt.colorbar(im, label='Number of Cells')  # Add colorbar
            plt.xlabel('Current at 600V [nA]', fontsize=16)
            plt.ylabel('Current at 800V [nA]', fontsize=16)
            plt.title('2D Histogram of Hot Cell Currents  (I600 > 100 nA)', fontsize=16)
            plt.grid(True)
            plt.xscale('log')
            plt.yscale('log')

            # Set limits to a small positive value close to zero
            plt.xlim(left=1e-1, right=1e3)
            plt.ylim(bottom=1e-1, top=1e3)

            # Ensure no zero values for log scale
            if np.any(hist <= 0):
                hist[hist == 0] = 1e-10  # Small value to avoid log(0) issues

            plt.legend(fontsize=16)  # Add legend
            plt_pdf_filename = "plot_current_comparison_2D.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources

            print("No hot cell currents found for one or both voltages, or lists are not the same length.")
        main_tex_tex = main_tex_tex.replace("<PLOT_CURRENT_COMPARISON>", pathToCode + plt_pdf_filename)


        # Ensure to replace the placeholder in the main_tex_tex variable

    
        ##################
        # Plotting the distribution of hot cell currents at 600V  I600>100nA OR I800>2.5xI600
        if hot_cell_currents_bad_600V and np.any(np.array(hot_cell_currents_bad_600V) > 0):

            hot_cell_currents_bad_600V = np.array(hot_cell_currents_bad_600V)
            hot_cell_currents_bad_600V = hot_cell_currents_bad_600V[hot_cell_currents_bad_600V > 0]
    
            max_count = int(max(hot_cell_currents_bad_600V))  # Convert the max value to an integer

            # Check if the minimum value is positive for log scaling
            min_value = np.min(hot_cell_currents_bad_600V)
            if min_value <= 0 or min_value >= max_count:
                min_value = max_count / 1000  # Set min_value to a small fraction of max_count

              #bins = np.logspace(np.log10(min_value), np.log10(max_count), num=80)
            # Check if the minimum value is positive for log scaling
            min_value = np.min(hot_cell_currents_bad_600V)
            if min_value <= 0 or min_value >= max_count:
                min_value = max_count / 1000  # Set min_value to a small fraction of max_count

            bins = np.logspace(np.log10(min_value), np.log10(max_count), num=80)
            
            # Ensure bins are strictly increasing
            bins = np.unique(bins)
            if len(bins) < 2:
                bins = np.logspace(0, np.log10(max_count), num=80)

            plt.figure(figsize=(12, 9))
            plt.hist(hot_cell_currents_bad_600V, bins=bins, edgecolor='black', log=True)  # log scale on y-axis
            plt.xlabel('Current [nA]', fontsize=16)
            plt.ylabel('Number of Cells (log scale)', fontsize=16)
            plt.title('Bad Cell Currents at 600 V (I600 > 10 nA AND I800>2.5xI600)||(I600 ≤ 10 nA AND I800>25 nA)', fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # Set x-axis to log scale
            plt.gca().set_xscale('log')
            # Format y-axis to standard notation
              #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
              #plt.gca().yaxis.get_major_formatter().set_scientific(False)  # Turn off scientific notation
              #plt.gca().yaxis.get_major_formatter().set_useOffset(False)   # Do not use offset notation

            # Automatically adjust layout to fit labels and title
            plt.tight_layout()

            plt_pdf_filename = "plot_bad_channel_current_2criteria.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources
        else:
            print("No valid hot cell currents found for log scaling.")
            # Optional: create a basic plot without log scaling if needed
            plt.figure(figsize=(12, 9))
            plt.hist([], bins=80, edgecolor='black')  # Basic histogram with no data
            plt.xlabel('Current [nA]', fontsize=16)
            plt.ylabel('Number of Cells', fontsize=16)
            plt.title('No Bad Cell Currents at 600 V', fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt_pdf_filename = "plot_bad_channel_current_2criteria_empty.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources

        main_tex_tex = main_tex_tex.replace("<PLOT_CURRENT_2>",  pathToCode + plt_pdf_filename)

       ##################

        # Check if both hot_cell_currents and hot_cell_currents_800V are not empty
            # if hot_cell_currents_bad_600V and hot_cell_currents_bad_800V and len(hot_cell_currents_bad_600V) == len(hot_cell_currents_bad_800V):
        # Ensure that the arrays are non-empty and have the same length
        if len(hot_cell_currents_bad_600V) > 0 and len(hot_cell_currents_bad_800V) > 0 and len(hot_cell_currents_bad_600V) == len(hot_cell_currents_bad_800V):

            plt.figure(figsize=(12, 9))

            # Create 2D histogram
            hist, xedges, yedges, im = plt.hist2d(hot_cell_currents_bad_600V, hot_cell_currents_bad_800V, bins=600, norm=mcolors.LogNorm())

            plt.colorbar(im, label='Number of Cells')  # Add colorbar
            plt.xlabel('Current at 600V [nA]', fontsize=16)
            plt.ylabel('Current at 800V [nA]', fontsize=16)
            plt.title('Bad Cell Currents (I600 > 10 nA AND I800>2.5xI600)||(I600 ≤ 10 nA AND I800>25 nA)', fontsize=16)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xscale('log')
            plt.yscale('log')
            # Set limits to start from a small positive value close to zero
            plt.xlim(left=1e-1, right=max(hot_cell_currents_bad_800V))
            plt.ylim(bottom=1e-1, top=max(hot_cell_currents_bad_800V))

            # Draw the line y = 2.5x in log scale
            x = np.logspace(np.log10(1e-1), np.log10(max(hot_cell_currents_bad_800V)), 1000)
            y = 2.5 * x
            plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='y=2.5x')

            plt.legend(fontsize=16)  # Add legend

            plt_pdf_filename = "plot_current_comparison_2D_bad.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources
        else:
            # Create an empty 2D plot with the same structure
            plt.figure(figsize=(12, 9))

            # Manually create empty data for histogram with small positive values
            empty_data_x = np.array([1e-1])  # Small positive value to avoid log(0) issues
            empty_data_y = np.array([1e-1])  # Small positive value to avoid log(0) issues

            # Create empty 2D histogram with fixed limits and manual vmin and vmax
            hist, xedges, yedges, im = plt.hist2d(empty_data_x, empty_data_y, bins=600, norm=mcolors.LogNorm(vmin=1e-1, vmax=1e3))

            # plt.colorbar(im, label='Number of Cells')  # Add colorbar
            plt.xlabel('Current at 600V [nA]', fontsize=16)
            plt.ylabel('Current at 800V [nA]', fontsize=16)
            plt.title('Bad Cell Currents (I600 > 10 nA AND I800>2.5xI600)||(I600 ≤ 10 nA AND I800>25 nA)', fontsize=16)
            plt.grid(True)
            plt.xscale('log')
            plt.yscale('log')

            # Set limits to start from a small positive value close to zero
            plt.xlim(left=1e-1, right=1e3)
            plt.ylim(bottom=1e-1, top=1e3)

            plt_pdf_filename = "plot_current_comparison_2D_bad.pdf"
            plt.savefig(pathToCode + plt_pdf_filename)
            plt.close()  # Close the figure to release resources


            print("No hot cell currents found for one or both voltages, or lists are not the same length.")
        main_tex_tex = main_tex_tex.replace("<PLOT_CURRENT_COMPARISON_BAD>", pathToCode + plt_pdf_filename)

        # Ensure to replace the placeholder in the main_tex_tex variable
 
        ##################

        
        
        # Loop of CV files
        for file in sorted_listOfCellCVFiles:
            if file == ".DS_Store":
                continue  # Skip the .DS_Store file

            if os.path.isdir(pathToCVResults+file):
                continue
            if "OBA" not in file:
                continue  # Skip files that don't contain "OBA" in the name
            split_result = file.split("OBA")
            if len(split_result) < 2:
                continue  # Skip files where "OBA" couldn't be split from the file name
            OBA_number = 'OBA' + split_result[1][:5]
            cell_cv_tex += f"\\begin{{frame}}\n"
            cell_cv_tex += f"\\frametitle{{CV: HPK / CMS; {thickness_OBA}$\\mu$m {OBA_number}}}\n"
            cell_cv_tex += f"\\begin{{figure}}[!ht]\n"
            cell_cv_tex += f"\\includegraphics[width=.95\\textwidth]{{{pathToCVResults}{file}}}\n"   
            cell_cv_tex += f"\\end{{figure}}\n"
            cell_cv_tex += f"\\end{{frame}}\n" 
        # end of IV and CV files sorting
        
        # Convert PDFs to PNGs for HPK
        pathToIVFailedSensors = os.path.join(pathToCode, 'IV_failed_HPK/')
        png_files = os.listdir(pathToIVFailedSensors)  # List all the files in the directory
        sorted_listOfpng_files = sorted(png_files, key=custom_sorting_key)

        pdf_directory = pathToIVFailedSensors
        png_directory = pathToIVFailedSensors
        
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_file_path = os.path.join(pdf_directory, filename)
                images = convert_from_path(pdf_file_path)
                for i, image in enumerate(images):
                    width, height = image.size
                    # Define the coordinates for the box to crop the image
                    left = int(width * 0.3)  # Adjust this value as needed
                    top = int(height * 0.05)
                    right = width
                    bottom = height
                    # Crop the image
                    cropped_image = image.crop((left, top, right, bottom))
                    cropped_image.save(f"{png_directory}/{filename[:-4]}_{i}.png", "PNG", mode="777")
        # End of Convert PDFs to PNGs for HPK

 
        # 2. for file in sorted_listOfpng_files for HPK:
        png_files = []
        sensor_files = {}
        for file in sorted_listOfpng_files:
            if os.path.isdir(pathToIVFailedSensors+file):
                continue
            if not file.startswith('.') and file.endswith('.png'):
                # Append the file to the list of PNG files
                png_files.append(file)
            sensor_number = file.split("_OBA")[0]
            if sensor_number not in sensor_files:
                sensor_files[sensor_number] = []
            sensor_files[sensor_number].append(file) 

        # Generate LaTeX output
        failed_tex = ""
        alternating_sections = []


        
        # 2. printing IV  failed sensor images for HPK:
        for sensor, files in sensor_files.items():
            # Filter out '.DS_Store' files
            png_files_for_sensor = [file for file in files if file.endswith('.png') and file != '.DS_Store']
            if not png_files_for_sensor:
                continue  # Skip if there are no valid PNG files for the sensor
            OBA_number = 'OBA' + png_files_for_sensor[0].split("OBA")[1][:5]
            # Further processing using png_files_for_sensor

               #png_files_for_sensor = [file for file in files if file.endswith('.png')]
               #OBA_number = 'OBA' + files[0].split("OBA")[1][:5]
            failed_tex += f"\\begin{{frame}}\n"
            failed_tex += f"\\frametitle{{IV Failed: HPK; {thickness_OBA} $\\mu$m {OBA_number}, Sensor: {sensor}}}\n"
            failed_tex += f"\\begin{{figure}}[!ht]\n"
            for i, png_file in enumerate(png_files_for_sensor):
                if i % 3 == 0 and i != 0:
                    failed_tex += f"\\end{{figure}}\n"
                    failed_tex += f"\\end{{frame}}\n" 
                    alternating_sections.append(failed_tex)
                    failed_tex += f"\\begin{{frame}}\n"
                    failed_tex += f"\\frametitle{{IV Failed: HPK;  {thickness_OBA}$\\mu$m {OBA_number}, Sensor: {sensor}}}\n"
                    failed_tex += f"\\begin{{figure}}[!ht]\n"
                
                failed_tex += f"\\includegraphics[width=.32\\textwidth]{{{pathToIVFailedSensors}{png_file}}}\n"                
            failed_tex += f"\\end{{figure}}\n"
            failed_tex += f"\\end{{frame}}\n"
            alternating_sections.append(failed_tex)
            
             # Handle grading results
            gradingFileName = os.path.join(pathToIVFailedSensors, f"{sensor}_{OBA_number}_grading_results.tex")
    
    # Read grading results
            if os.path.exists(gradingFileName):
                with open(gradingFileName, "r") as file:
                    grading_tex_content = file.read()

        # Fetching current data for the sensor from DataFrame
                failed_sensor = IVTotalDataFrameCMSHPK_AllOBA[IVTotalDataFrameCMSHPK_AllOBA['Sensor name'] == sensor]
                if not failed_sensor.empty:
                    voltage_600_currents = failed_sensor[failed_sensor['Voltage'] == 600.0]['Total current'].tolist()
                    voltage_800_currents = failed_sensor[failed_sensor['Voltage'] == 800.0]['Total current'].tolist()
                    factor_IV = [voltage_800 / voltage_600 for voltage_800, voltage_600 in zip(voltage_800_currents, voltage_600_currents)]
                    factor_IV_formatted = [f"{value:.2f}" for value in factor_IV]
                    factor_IV_formatted_str = ', '.join(factor_IV_formatted)
            
            # Add the content of grading_results.tex to new_grading_tex_template
                    new_grading_tex_template = re.sub(r"\\frametitle\{.*\}", f"{{IV Grading Sensor: {sensor}}}\n", grading_tex_content)
                    alternating_sections.append(new_grading_tex_template)
            else:
                print(f"Grading results file not found: {gradingFileName}")

            failed_tex = ""

# End of LaTeX document generation

        
        # end of printing failed sensors for HPK

        # Convert pdf files to png files for CMS 
        
 
        # OBA loop for CMS
        pdf_directory_CMS = pathToIVFailedSensors_CMS
        png_directory_CMS = pathToIVFailedSensors_CMS

        for file in sorted_listOfCellCurrentFiles:
            if file == ".DS_Store":
                continue
            if os.path.isdir(pathToIVResults+file):
                continue
            # now check for each OBA 
            OBA_number = 'OBA' + file.split("OBA")[1][:5]
            # Convert PDFs to PNGs for failed sensors CMS
            # Filter the DataFrame to select only the failed sensors
            grading_OBA_CMS = grading_DataFrame_AllOBA[(grading_DataFrame_AllOBA['OBA'] == OBA_number) & (grading_DataFrame_AllOBA['Foundry'] == 'CMS')]
            failed_sensors_CMS = grading_OBA_CMS[grading_OBA_CMS['Overall grading'] == 'Failed']
  
            pdf_directory_CMS = pathToIVFailedSensors_CMS
            png_directory_CMS = pathToIVFailedSensors_CMS
            for filename in os.listdir(pdf_directory_CMS):
                if filename.endswith('.pdf'):
                      #pdf_file_path_CMS = os.path.join(pdf_directory_CMS, filename)
                    # Extract sensor name and OBA number from the filename
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        sensor_name = parts[0]
                        oba_number = parts[1].split('.')[0]  # Remove the file extension

                        # Check if the sensor name and OBA number are valid
                        if sensor_name in failed_sensors_CMS['Sensor name'].astype(str).values and oba_number == OBA_number:
                            pdf_file_path_CMS = os.path.join(pdf_directory_CMS, filename)
                            images = convert_from_path(pdf_file_path_CMS)
                            # Rename the tex file
                            old_filename = f"{sensor_name}_{oba_number}_grading_results.tex"
                            new_filename = f"{sensor_name}_{oba_number}_grading_results_fail.tex"
                            source_path = os.path.join(pdf_directory_CMS, old_filename)
                            destination_path = os.path.join(pdf_directory_CMS, new_filename)
                            shutil.copy(source_path, destination_path)

                            for i, image in enumerate(images):
                                width, height = image.size
                                left = int(width * 0.3)
                                top = int(height * 0.05)
                                right = width
                                bottom = height
                                cropped_image = image.crop((left, top, right, bottom))
                                png_filename = f"{sensor_name}_OBA{OBA_number}_{i}.png"
                                cropped_image.save(os.path.join(png_directory_CMS, f"{filename[:-4]}_{i}.png"), "PNG", mode="777")
                            # Remove the PDF file after converting to PNG
                              #os.remove(pdf_file_path_CMS)
            # Remove PDF files that start with the sensor name

         
        # end of OBA loop for CMS to convert png files
        for filename in os.listdir(pdf_directory_CMS):
            if filename.endswith('.pdf'):
                # Extract sensor name from the filename
                sensor_name = filename.split('_')[0]
        
                # Remove PDF files that start with the sensor name
                if filename.startswith(f"{sensor_name}_"):
                    os.remove(os.path.join(pdf_directory_CMS, filename))

        # Remove tex files that start with the sensor namewhich are not assigned
        # Iterate through tex files
        for filename in os.listdir(pdf_directory_CMS):
            if filename.endswith('.tex'):
                if not filename.endswith('_fail.tex'):
                    os.remove(os.path.join(pdf_directory_CMS, filename))
 
        png_files_CMS = os.listdir(pathToIVFailedSensors_CMS)  # List all the files in the directory
        sorted_listOfpng_files_CMS = sorted(png_files_CMS, key=custom_sorting_key)

        
        # 2. for file in sorted_listOfpng_files for CMS:

        for file in sorted_listOfpng_files_CMS:
            if file.startswith('.DS_Store'):
                continue  # Skip the .DS_Store file

            if os.path.isdir(pathToIVFailedSensors_CMS+file):
                continue
            if not file.startswith('.') and file.endswith('.png'):
                # Append the file to the list of PNG files
                png_files_CMS.append(file)
            sensor_number_CMS = file.split("_OBA")[0]
           
            if sensor_number_CMS not in sensor_files_CMS:
                sensor_files_CMS[sensor_number_CMS] = []
            sensor_files_CMS[sensor_number_CMS].append(file)
          
        index = 0

               
        alternating_sections_CMS = []
 
        # 2. printing failed sensor images for CMS:
 
        for sensor, files in sensor_files_CMS.items():

            # Filter out '.DS_Store' files
            png_files_for_sensor_CMS = [file for file in files if file.endswith('.png') and file != '.DS_Store']
            if not png_files_for_sensor_CMS:
                continue  # Skip if there are no valid PNG files for the sensor

            # Check if the list is not empty before accessing its elements
            if png_files_for_sensor_CMS:
                OBA_number = 'OBA' + png_files_for_sensor_CMS[0].split("OBA")[1][:5]
            else:
                continue  # Skip if there are no PNG files for the sensor

            # Further processing using png_files_for_sensor

               #png_files_for_sensor = [file for file in files if file.endswith('.png')]
               #OBA_number = 'OBA' + files[0].split("OBA")[1][:5]
           
            failed_tex_CMS += f"\\begin{{frame}}\n"
            failed_tex_CMS += f"\\frametitle{{IV Failed: CMS; {thickness_OBA} $\\mu$m {OBA_number}, Sensor: {sensor}}}\n"
            failed_tex_CMS += f"\\begin{{figure}}[!ht]\n"
            for i, png_file_CMS in enumerate(png_files_for_sensor_CMS): 
                if i % 3 == 0 and i != 0:

                    failed_tex_CMS += f"\\end{{figure}}\n"
                    failed_tex_CMS += f"\\end{{frame}}\n" 
                    alternating_sections_CMS.append(failed_tex_CMS)
                    failed_tex_CMS += f"\\begin{{frame}}\n"
                    failed_tex_CMS += f"\\frametitle{{IV Failed: CMS;  {thickness_OBA}$\\mu$m {OBA_number}, Sensor: {sensor}}}\n"
                    failed_tex_CMS += f"\\begin{{figure}}[!ht]\n"
                
                failed_tex_CMS += f"\\includegraphics[width=.32\\textwidth]{{{pathToIVFailedSensors_CMS}{png_file_CMS}}}\n"                
            failed_tex_CMS += f"\\end{{figure}}\n"
            failed_tex_CMS += f"\\end{{frame}}\n"
            alternating_sections_CMS.append(failed_tex_CMS)
            gradingFileName_CMS = pathToIVFailedSensors_CMS + str(sensor) + '_' + OBA_number + '_grading_results_fail.tex'   
            grading_tex_CMS_template = open(gradingFileName_CMS, "r").read()
            alternating_sections_CMS.append(grading_tex_CMS_template)

            failed_tex_CMS = ""
 

        # end of printing failed sensors for CMS

        
     
        main_tex_tex = main_tex_tex.replace("<FRAMES_PLOTS>", "\n".join(alternating_sections))
        main_tex_tex = main_tex_tex.replace("<FRAMES_PLOTS_CMS>", "\n".join(alternating_sections_CMS))
        

        table_tex += "\\begin{frame}\n"
        table_tex += f"\\frametitle{{Table of the Campaign {escaped_campaignType}  (HPK / CMS)}}\n"

        table_tex += "\\begin{minipage}[t]{0.48\\textwidth}\n"
        table_tex += "\\centering\n"
        table_tex += "\\scriptsize\n"
        table_tex += "\\setlength{\\tabcolsep}{1pt}\n"  # Reduce column spacing to 1pt
        table_tex += "\\renewcommand{\\arraystretch}{0.5}\n"  # Reduce row spacing to half
        table_tex += "\\begin{tabular}{cccccc}\n"
           #table_tex += "OBA & N sensor & IV Pas & IV Fail & CV Pas & CV Fail \\\\\n"
        table_tex += "OBA & N\\_sensor & IV\\_Pas & IV\\_Fail & CV\\_Pas & CV\\_Fail \\\\\n"

        table_tex += "\\hline\n"
        table_tex += "<TABLE_LEFT>\n"
        table_tex += "\\end{tabular}\n"
        table_tex += "\\end{minipage}\\hfill\n"

        table_tex += "\\begin{minipage}[t]{0.48\\textwidth}\n"
        table_tex += "\\centering\n"
        table_tex += "\\scriptsize\n"
        table_tex += "\\setlength{\\tabcolsep}{1pt}\n"  # Reduce column spacing to 1pt
        table_tex += "\\renewcommand{\\arraystretch}{0.5}\n"  # Reduce row spacing to half
        table_tex += "\\begin{tabular}{cccccc}\n"

        table_tex += "OBA  & N\\_sensor & IV\\_Pas & IV\\_Fail & CV\\_Pas & CV\\_Fail \\\\\n"
        table_tex += "\\hline\n"
        table_tex += "<TABLE_RIGHT>\n"
        table_tex += "\\end{tabular}\n"
        table_tex += "\\end{minipage}\n"

        table_tex += "\\end{frame}\n"

        
        table_tex = table_tex.replace("<TABLE_LEFT>", table_rows_left)
        table_tex = table_tex.replace("<TABLE_RIGHT>", table_rows_right)
        table_tex = table_tex.replace("<CAMPAIGN>", escaped_campaignType)       
        main_tex_tex = main_tex_tex.replace("<THICKNESS>",  str(thickness_OBA))
        main_tex_tex = main_tex_tex.replace("<SENSORCHANNEL>",  str(numberOfChannels))
        main_tex_tex = main_tex_tex.replace("<DELIVERY_DATE>", escaped_deliveryMonth)
        main_tex_tex = main_tex_tex.replace("<DELIVERY_YEAR>", escaped_deliveryYear)
                
        main_tex_tex = main_tex_tex.replace("<FAILED_SENSORNUMBER>", str(amountOfFailedlSensor))
        main_tex_tex = main_tex_tex.replace("<FAILED_SENSORNUMBER_CMS>", str(amountOfFailedlSensor_CMS))
        main_tex_tex = main_tex_tex.replace("<FAILED_SENSORNUMBER_CV>", str(amountOfFailedlSensor_CV))
        main_tex_tex = main_tex_tex.replace("<FAILED_SENSORNUMBER_CMS_CV>", str(amountOfFailedlSensor_CMS_CV))

        

        main_tex_tex = main_tex_tex.replace("<TABLE_ROWS>", table_tex)
        main_tex_tex = main_tex_tex.replace("<FRAMES_CELLCURRENTS>", cell_current_tex)
        main_tex_tex = main_tex_tex.replace("<FRAMES_CELLCV>", cell_cv_tex)
        
        main_tex_tex = main_tex_tex.replace("<FRAMES_PLOTS>", failed_tex)
        main_tex_tex = main_tex_tex.replace("<FRAMES_PLOTS_CMS>", failed_tex)

        # Call the output function with the deliveryDate and store the result
        output_files = output(deliveryMonth, deliveryYear)

        # Now you can access the filenames using the output_files variable
        tmp_file_path = pathToCode + output_files["pdf"].replace(".pdf", ".tex")
        with open(tmp_file_path, "w") as tmp_file:
            tmp_file.write(main_tex_tex)
        # Make latex file
           # compile without stopping for erors
        cmd = "pdflatex -interaction=nonstopmode -output-directory=%s %s" % (os.path.dirname(pathToCode + output_files["pdf"]), tmp_file_path)

        #cmd = "pdflatex -output-directory=%s %s" % (os.path.dirname(pathToCode + output_files["pdf"]), tmp_file_path)

        for it in range(2):     #need to compile twice to properly see the table of contents and page numbers
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
            out, err = p.communicate()
            code = p.returncode
        #cleanup
        for file_extension in [".aux", ".log", ".nav", ".out", ".snm", ".toc"]:
            os.remove(tmp_file_path.replace(".tex", file_extension))

def copyThePlotsOfFailedSensorsHPK(fileName, directory, parent_dir, sensor, foundry):
      # pathToHexPlot = parent_dir + '/' + directory + '/'+ str(sensor) + '_' + foundry + '/' + fileName
    pathToHexPlot = parent_dir + '/' + directory + '/'+ str(sensor) + '_HPK' + '/' + fileName

    dataHPK_IV = pathToCode + '/IV_failed_HPK/'  # Adjusting to the correct path
    check_and_create_directory(dataHPK_IV)  # This is a function you should implement to check/create the directory

    shutil.copy2(pathToHexPlot, dataHPK_IV)

    pathToIVFailedSensors = pathToCode + '/IV_failed_HPK' +'/'
       #checkIfFolderExistAndCreate(pathToIVFailedSensors)
    new_file_name = pathToIVFailedSensors + str(sensor)  + '_' + OBA_number + '_' + fileName
    old_file_name = dataHPK_IV + '/' + fileName
    os.rename(old_file_name, new_file_name)    
    
def copyThePlotsOfFailedSensors(fileName, directory, parent_dir, sensor, foundry):

    pathToHexPlot = parent_dir + '/' + directory + '/'+ str(sensor) + '/' + fileName
    data_IV_CMS = pathToCode + '/IV_failed_CMS/'  # Adjusting to the correct path
    check_and_create_directory(data_IV_CMS)  # This is a function you should implement to check/create the directory

    shutil.copy2(pathToHexPlot, data_IV_CMS)
    pathToIVFailedSensors = pathToCode + '/IV_failed_CMS' +'/'
    check_and_create_directory(pathToIVFailedSensors)
    new_file_name = pathToIVFailedSensors + str(sensor)  + '_' + OBA_number + '_' + fileName
    old_file_name = data_IV_CMS + '/' + fileName
    os.rename(old_file_name, new_file_name)

def copyThePlotsOfFailedSensors_CV(fileName, directory, parent_dir, sensor, foundry):

    pathToCVFailedSensors_CMS = pathToCode + '/CV_failed_CMS' +'/'
    check_and_create_directory(pathToCVFailedSensors_CMS)
    new_file_name = pathToCVFailedSensors_CMS + str(sensor)  + '_' + OBA_number + '_' + fileName
    old_file_name =  parent_dir  + '/' + str(sensor) +  '/' + fileName
    os.rename(old_file_name, new_file_name)

def determineGradingHPK(current_dir, pathToCode, foundry):
    parent_dir = os.path.dirname(current_dir)
    Grading_csv_generator.createGradingTable(parent_dir, pathToCode, foundry)
    column_names = ["Sensor name", "condition1", "condition2", "condition3", "condition4", "condition5", "condition6", "Overall grading", "I600V", "I800V",  "cfactor"]
    # Read the CSV file with all columns
     
    data_grading_thickness = pd.read_csv(pathToCode + f'/grading_results_{foundry}_IV.csv', names=column_names, delimiter=',')

    graddir = parent_dir + '/grading/'
    failedIndexes = data_grading_thickness.loc[data_grading_thickness['Overall grading'] == 'Failed', 'Sensor name'].values
  
    for sensor in failedIndexes:
        failedSensor = {}
        failedSensor['Sensor name'] = sensor
        failedSensor['Foundry'] = foundry
        failedSensor['OBA number'] = OBA_number
        sensorsFailedIVGrading.append(failedSensor)
        # Determine which function to call based on the foundry
        if foundry == 'CMS':
            copyThePlotsOfFailedSensors('std_uncorrected.pdf', 'hexplots', parent_dir, sensor, foundry)
            copyThePlotsOfFailedSensors('total_current_IV.pdf', 'totalIV', parent_dir, sensor, foundry)
            copyThePlotsOfFailedSensors('allchannels_IV.pdf', 'channelIV', parent_dir, sensor, foundry)
            copyThePlotsOfFailedSensors('grading_results.tex', 'grading', parent_dir, sensor, foundry)
        elif foundry == 'Hamamatsu':
            copyThePlotsOfFailedSensorsHPK('std_uncorrected.pdf', 'hexplots', parent_dir, sensor, foundry)
            copyThePlotsOfFailedSensorsHPK('total_current_IV.pdf', 'totalIV', parent_dir, sensor, foundry)
            copyThePlotsOfFailedSensorsHPK('allchannels_IV.pdf', 'channelIV', parent_dir, sensor, foundry)
            copyThePlotsOfFailedSensorsHPK('grading_results.tex', 'grading', parent_dir, sensor, foundry)
    return data_grading_thickness

def determineGradingCV(current_dir, pathToCode, foundry):
    parent_dir = os.path.dirname(current_dir)
    graddir = parent_dir + '/grading/'
    Grading_csv_generator.createGradingTableCV(parent_dir, pathToCode, foundry)
    column_names = ["Sensor name", "condition1", "condition2", "condition3", "Overall grading"]
    # Read the CSV file with all columns
       #data_grading_thickness = pd.read_csv(parent_dir + f'/grading_results_campaign_IV_{OBA_number}.csv', names=column_names, delimiter=',')
    data_grading_thickness_CV = pd.read_csv(pathToCode + f'/grading_results_{foundry}_CV.csv', names=column_names, delimiter=',')
    graddir = parent_dir + '/grading/'
    failedIndexes = data_grading_thickness_CV.loc[data_grading_thickness_CV['Overall grading'] == 'Failed', 'Sensor name'].values
    
    for sensor in failedIndexes:
        failedSensor = {}
        failedSensor['Sensor name'] = sensor
        failedSensor['Foundry'] = foundry
        failedSensor['OBA number'] = OBA_number
        sensorsFailedCVGrading.append(failedSensor)
        # Determine which function to call based on the foundry

        copyThePlotsOfFailedSensors_CV('grading_results.tex', 'grading', graddir, sensor, foundry)
  
    return data_grading_thickness_CV

def getSensorDetails(sensorName):
    # Ensure sensorName is a string
    sensorName = str(sensorName)
    
    # Now you can safely access the first character
    firstDigit = sensorName[0]

    # Continue with the rest of your function logic...
    # Example logic (replace with your actual logic):
    thickness = None
    quality = None
    coverage = None

    if firstDigit == '1':
        # Logic for sensors starting with '1'
        thickness = 300
        quality = 'Low'
        coverage = 'Full'
    elif firstDigit == '2':
        # Logic for sensors starting with '2'
        thickness = 200
        quality = 'Medium'
        coverage = 'Partial'
    elif firstDigit == '3':
        # Logic for sensors starting with '3'
        thickness = 120
        quality = 'High'
        coverage = 'Full'
    else:
        # Default logic
        thickness = 150
        quality = 'Low'
        coverage = 'Minimal'

    return thickness, quality, coverage
    
def scratchIDFromName(filename):
    # Assuming sensor name is between the last hyphen and the file extension
    sensor_name = os.path.splitext(filename)[0].split('-')[-1]
    return sensor_name
def thicknessFromName(filename):
    # Assuming thickness is between the second and third hyphen
    parts = filename.split('-')
    if len(parts) >= 3:
        thickness_str = parts[2]
        # Extracting digits from the string
        thickness = int(''.join(filter(str.isdigit, thickness_str)))
        return thickness
    else:
        return None

def read(file_path):
    """
    Read data from the given file path.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: Two numpy arrays containing voltages and capacitances.
    """
    voltages = []
    capacitances = []

    with open(file_path, 'r', errors='ignore') as file:
        # Read file line by line
        for line in file:
            line = line.strip()
            # Skip empty lines and lines starting with '#'
            if not line or line.startswith('#'):
                continue
            # Check if the line is the header row
            if line.startswith('voltage'):
                continue
            values = line.split('\t')
            voltages.append(float(values[0]))
            capacitances.append(float(values[2]))

    voltages = np.array(voltages, dtype=np.float32)
    capacitances = np.array(capacitances, dtype=np.float32)

    return voltages, capacitances


def thicknessFromName(fileName):
    # Identify from the file name the sensor thickness in micron
    sFileName = f"{fileName:}"
    id = sFileName[23]  # 24th element is first digit of scratch ID
    if id == '1':
        return 300
    elif id == '2':
        return 200
    elif id == '3':
        return 120
    else:
        return None

def scratchIDFromName(fileName):
    # Get the sensor scratch ID from the file name
    sFileName = f"{fileName:}"

#Creation of DepletionVoltage CV Plots
def plot_and_fit(data, fLin, fConst, sensor_name, pathToResults, foundry):
      #print("  data in  plot_and_fit =================>")
      #print(data)
    print("     fLin   BEFORE plot_and_fit  ================= ",  fLin)
    print("fLin parameter 0 BEFORE:", fLin.GetParameter(0))
    print("fLin parameter 1 BEFORE:", fLin.GetParameter(1))
    print("     fConst  BEFORE plot_and_fit  ================= ",  fConst)
    print("fConst parameter 0 BEFORE:", fConst.GetParameter(0))

    # Initialize ROOT
    ROOT.gROOT.SetBatch(True)
      #ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    canvas = ROOT.TCanvas("Canvas", "Depletion Voltage Plot", 800, 600)
    graph = ROOT.TGraphErrors()

    if foundry == 'Hamamatsu':
         #print("  CV_Graphs.root from Hamamatsu...")    
        outfile = ROOT.TFile(pathToCode + "CV_Graphs.root", "UPDATE")

    
    # Fill the graph with data
    for i, row in data.iterrows():
        graph.SetPoint(i, row['Voltage'], row['Capacitance'])
            
    
    # Additional settings for the graph
    graph.SetLineColor(ROOT.kBlue)
    graph.SetMarkerStyle(20)
    graph.Draw("AP")
    graph.GetXaxis().SetTitle("Effective bias voltage [V]")
    graph.GetYaxis().SetTitle("Open-corrected 1/C^2 serial [1/pF^2]")
    graph.SetTitle(f"Diode at HPK, {sensor_name}")
    if SensorThickness == 120:
        graph.GetXaxis().SetLimits(0, 100)  # Set x-axis range
        graph.SetMaximum(0.15)  # Set y-axis maximum
    else: 
        graph.GetXaxis().SetLimits(0, 400)  # Set x-axis range
        graph.SetMaximum(0.25)  # Set y-axis maximum

    for i in range(graph.GetN()):
        x, y = graph.GetPointX(i), graph.GetPointY(i)  # GetPointX and GetPointY return float
        print(f"Point {i}: x = {x}, y = {y}")

    
    # Fit the right side with a constant function
        #fit_result_const = graph.Fit(fConst, 'S R')
  
    # Fit the left side with a linear function
        #fit_result_lin = graph.Fit(fLin, 'S R+')
    if SensorThickness == 120:
        fLin.SetRange(0, 35)
        fConst.SetRange(55, 95)
    try:
        fit_result_const = graph.Fit(fConst, 'S R')
        if not fit_result_const or not fit_result_const.IsValid():
            raise RuntimeError("Constant fit failed!!!!")

        fit_result_lin = graph.Fit(fLin, 'S R+')
        if not fit_result_lin or not fit_result_lin.IsValid():
            raise RuntimeError("Linear fit failed!!!!")
    except Exception as e:
        print("Fit exception!!!!!!!!!!!:", e)
        return None, None
    
    # Print parameter values after fitting
    print("     fLin   AFTER plot_and_fit  ================= ",  fLin)
    print("fLin parameter 0 AFTER:", fLin.GetParameter(0))
    print("fLin parameter 1 AFTER:", fLin.GetParameter(1))
    print("     fConst  AFTER plot_and_fit  ================= ",  fConst)
    print("fConst parameter 0 AFTER:", fConst.GetParameter(0))

    
    # Check if the fit results are valid
    if not (fit_result_const and fit_result_const.IsValid() and fit_result_lin and fit_result_lin.IsValid()):
        print("Error: Fit failed in plot_and_fit !!!!!!!!!!!!!!!!!!")
        return None, None
    
    # Calculate fit parameters
    C = fConst.GetParameter(0)
    B = fLin.GetParameter(0)
    M = fLin.GetParameter(1)
    print("   fit!!!!        C===========>", C)
    print("   fit!!!!        B===========>", B)
    print("   fit!!!!        M===========>", M)
    # Calculate depletion voltage
    if M != 0:
        Vdep = (C - B) / M
    else:
        print("Error: Division by zero encountered. Cannot calculate Vdep.")
        Vdep = None
    print("   fit!!!!        Vdep===========>", Vdep)
    # Calculate depletion voltage uncertainty
    eC = fConst.GetParError(0)
    eB = fLin.GetParError(0)
    eM = fLin.GetParError(1)
    numerator = C - B
    eNumerator = ROOT.TMath.Sqrt(eC**2 + eB**2)
    if M != 0:
        eVdep = Vdep * ROOT.TMath.Sqrt((eNumerator / numerator)**2 + (eM / M)**2)
    else:
        print("Error: Division by zero encountered while calculating eVdep.")
        eVdep = None

    # Draw vertical line at the depletion voltage
    if Vdep is not None:
        line = ROOT.TLine(Vdep, 0, Vdep, 0.32)  # x1, y1, x2, y2 = graph maximum
        line.SetLineColor(ROOT.kGreen + 2)
        line.SetLineStyle(2)
        line.Draw()
        # Print Vdep results on plot
        t = ROOT.TLatex()
        t.SetTextColor(ROOT.kGreen + 2)
        if SensorThickness ==120:
            t.DrawLatex(42, 0.12, f'V_{{dep}}= ({Vdep:.1f} #pm {eVdep:.1f}) V')
        else:
            t.DrawLatex(42, 0.2, f'V_{{dep}}= ({Vdep:.1f} #pm {eVdep:.1f}) V')
    else:
        print("Error: Vdep calculation failed. Skipping further operations.")
    print("   fit!!!!   2.nd     Vdep===========>", Vdep)
    # New TGraphErrors for Vdep result
    tge = ROOT.TGraphErrors()
    tge.SetMarkerColor(ROOT.kGreen + 2)
    tge.SetLineColor(ROOT.kGreen + 2)
    tge.SetLineStyle(2)
    tge.SetMarkerStyle(20)
    if Vdep is not None:
        tge.SetPoint(0, Vdep, C)
        tge.SetPointError(0, eVdep, 0)
        # Draw Vdep result on the plot
        tge.Draw('PL')
    else:
        print("Error: Vdep calculation failed. Skipping setting point in tge !!!!!!!!!!.")

    #Creating the empty histogram for summary depletion voltage plot
    badflag_histo = ROOT.TH1F(f"{sensor_name}", f"{sensor_name}",10, 0, 10)

    #Change for everu thickness type
    #if Vdep >= 70: #for 120um
    #if Vdep >= 160: #for 200um
    if Vdep >= 370: #for 300um
        badflag_histo.Fill(1)
    
    badflag_histo.Write()

    if pathToResults:
        # Remove the redundant directory name "DepletionVoltageHPK" if it's already included in pathToResults
        if "DepletionVoltageHPK" in pathToResults:
            file_path = os.path.join(pathToResults, f"{sensor_name}.pdf")
        else:
            file_path = os.path.join(pathToResults, "DepletionVoltageHPK", f"{sensor_name}.pdf")

        file_path = file_path.replace("//", "/")  # Replace double slashes if any
        try:
            canvas.SaveAs(file_path)
             #print("PDF file saved successfully at:", file_path)
            graph.Write(f"{sensor_name}")
             #print(f"{sensor_name} has been written to the ROOT file...")
        except Exception as e:
            print("Error saving PDF file:", e)
        finally:
            canvas.Close()
    else:
        print("Error: pathToResults is empty. Cannot save PDF file.")

    outfile.Close()
    return Vdep, eVdep

 


def calculate_CMSSensorCounts(grading_DataFrame_Total, OBA_number):
    # Filter the dataframe for the specific OBA number
    oba_specific_df = grading_DataFrame_Total[grading_DataFrame_Total['OBA'] == OBA_number]

    # Filter CMS sensors with Overall grading = Failed
    failed_count = oba_specific_df[(oba_specific_df['Foundry'] == 'CMS') & 
                                   (oba_specific_df['Overall grading'] == 'Failed')].shape[0]

    # Filter CMS sensors with Overall grading = Passed
    passed_count = oba_specific_df[(oba_specific_df['Foundry'] == 'CMS') & 
                                   (oba_specific_df['Overall grading'] == 'Passed')].shape[0]

    return failed_count, passed_count

def createSingleIVPlot(IVdataAllDataFrame, IVGRDataFrameCMSHPK, IVTotalDataFrameCMSHPK, plot, OBA_number, areSensorsAtCMS, grading_DataFrame_Total, amountOfSensorsAtHamamatsu, amountOfSensorsAtCMS):
    # Filter grading_DataFrame_Total for the specified OBA_number and foundry CMS
    cms_sensors = grading_DataFrame_Total[(grading_DataFrame_Total['OBA'] == OBA_number) & (grading_DataFrame_Total['Foundry'] == 'CMS')]['Sensor name']

    # Call the function to calculate the number of sensors of CMS with Overall grading= Failed and Passed
    failed_count, passed_count = calculate_CMSSensorCounts(grading_DataFrame_Total,OBA_number)

    fig300IV, axs300IV = plt.subplots(2, 2, figsize=(30, 18))
    totalCMSsensor = failed_count +  passed_count  
    dataHPK = IVdataAllDataFrame[IVdataAllDataFrame['Foundry'] == 'Hamamatsu']

     #sensorThickness = getSensorDetails(dataHPK['Sensor name'][0])[0]
    sensorThickness =args.SensorThickness

    medians_HPK = dataHPK.groupby(['Sensor name'])['Cell current [nA]'].median().values
    dataCMS = IVdataAllDataFrame[(IVdataAllDataFrame['Foundry'] == 'CMS')]
 
     # Combine dataHPK and dataCMS dataframes
    # Filter dataHPK to only include sensors that exist in both dataCMS and dataHPK
    # Convert 'Sensor name' column to string type in both dataCMS and dataHPK
     #dataCMS['Sensor name'] = dataCMS['Sensor name'].astype(str)
     #dataHPK['Sensor name'] = dataHPK['Sensor name'].astype(str)

    dataCMS_filtered = dataCMS[dataCMS['Sensor name'].isin(dataHPK['Sensor name'])]

    IVdataCMSHPKDataFrame = pd.concat([dataHPK, dataCMS_filtered])

     
     # medians_CMS = dataCMS.groupby(['Sensor name'])['Cell current [nA]'].median().values
    medians_CMS = dataCMS_filtered.groupby(['Sensor name'])['Cell current [nA]'].median().values

    xmean_HPK = statistics.mean(medians_HPK)


    # Prepare 1000V data
    dataHPK_1000V = IVdata1000VDataFrame[IVdata1000VDataFrame['Foundry'] == 'Hamamatsu']
    dataCMS_filtered_1000V = dataCMS[dataCMS['Sensor name'].isin(dataHPK_1000V['Sensor name'])]
    IVdataCMSHPKDataFrame_1000V = pd.concat([dataHPK_1000V, dataCMS_filtered_1000V])
    medians_HPK_1000V = dataHPK_1000V.groupby(['Sensor name'])['Cell current [nA]'].median().values
    medians_CMS_1000V = dataCMS_filtered_1000V.groupby(['Sensor name'])['Cell current [nA]'].median().values
    xmean_HPK_1000V = statistics.mean(medians_HPK_1000V)

    # Initialize plot
    fig300IV, axs300IV = plt.subplots(2, 2, figsize=(30, 18))

    # Plot 600V data
    sns.boxenplot(x='Sensor name', y="Cell current [nA]", data=dataHPK, hue="Foundry",palette="pastel", ax=axs300IV[0, 0], dodge=True)
    axs300IV[0, 0].axhline(y=xmean_HPK, color='green', linewidth=2, linestyle='--', label=f'Mean HPK 600V = {xmean_HPK:.2f} nA')    

    # Treshold
    axs300IV[0, 0].axhline(y=1e2, color='red', linewidth=2, linestyle='-', label='Treshold ')

    # Plot 1000V data
    sns.boxenplot(x='Sensor name', y="Cell current [nA]", data=dataHPK_1000V, hue="Foundry", 
                  palette="bright", ax=axs300IV[0, 0], dodge=True)
    axs300IV[0, 0].axhline(y=xmean_HPK_1000V, color='blue', linewidth=2, linestyle='--', label=f'Mean HPK 1000V = {xmean_HPK_1000V:.2f} nA')

    # Title and legend
    axs300IV[0, 0].title.set_text(f'Cell Current: {OBA_number}, 600V & 1000V')
    axs300IV[0, 0].legend(loc='lower right', fontsize=12)

    # Set log scale, grid, and labels
    axs300IV[0, 0].set_yscale("log")
    axs300IV[0, 0].set_ylim([1e-3, 1e5])
    axs300IV[0, 0].grid(True, which='major', axis='both')
    axs300IV[0, 0].set_xticklabels(axs300IV[0, 0].get_xticklabels(), rotation=90)
    axs300IV[0, 0].xaxis.label.set_size(30)
    axs300IV[0, 0].yaxis.label.set_size(30)

    
    ################################### 2. plot (0,1
    dataGRHPK = IVGRDataFrameCMSHPK[IVGRDataFrameCMSHPK['Foundry'] == 'Hamamatsu']
    dataGRCMS = IVGRDataFrameCMSHPK[IVGRDataFrameCMSHPK['Foundry'] == 'CMS']
    dataGRCMS_filtered = dataGRCMS[dataGRCMS['Sensor name'].isin(dataGRHPK['Sensor name'])]
  

    IVGRDataFrameCMSHPK = pd.concat([dataGRHPK, dataGRCMS_filtered])



    # Filter the data frame to get rows where 'Total current' is greater than threshold
    treshold=(IVGRDataFrameCMSHPK['GR current'] > 120) & (IVGRDataFrameCMSHPK['Voltage'] > 150) & (IVGRDataFrameCMSHPK['Voltage'] < 900)
    filtered_data = IVGRDataFrameCMSHPK[treshold]
    sensorNamesAboveTreshold=filtered_data.loc[treshold, 'Sensor name'].unique()
    pd.set_option('display.max_rows', None)

    treshold_HPK=(dataGRHPK['GR current'] > 120) & (dataGRHPK['Voltage'] > 150) & (dataGRHPK['Voltage'] < 900)
    filtered_data_HPK = dataGRHPK[treshold_HPK]
    treshold_CMS=(dataGRCMS_filtered['GR current'] > 120) & (dataGRCMS_filtered['Voltage'] > 150) & (dataGRCMS_filtered['Voltage'] < 900)
    filtered_data_CMS = dataGRCMS_filtered[treshold_CMS]

    sensorNamesAboveTreshold_HPK=filtered_data_HPK.loc[treshold_HPK, 'Sensor name'].unique()
    sensorNamesAboveTreshold_CMS=filtered_data_CMS.loc[treshold_CMS, 'Sensor name'].unique()
    
    sns.lineplot(data=IVGRDataFrameCMSHPK, x="Voltage", y="GR current", ax = axs300IV[0,1], hue='Foundry', units='Sensor name', estimator=None)
    
    if sensorNamesAboveTreshold_HPK.size!=0:
        dataGRHPKWithinTreshold=dataGRHPK[~dataGRHPK['Sensor name'].isin(sensorNamesAboveTreshold_HPK)]
        dataGRHPKOutsideTreshold=dataGRHPK[dataGRHPK['Sensor name'].isin(sensorNamesAboveTreshold_HPK)]
        for sensor_name in sensorNamesAboveTreshold_HPK:
            if not dataGRHPKOutsideTreshold.empty:
                    sns.lineplot(data=dataGRHPKOutsideTreshold, x="Voltage", y="GR current", ax=axs300IV[0, 1], hue='Foundry', units='Sensor name', estimator=None, palette=['r'])         
        sns.lineplot(data=dataGRHPKWithinTreshold, x="Voltage", y="GR current", ax = axs300IV[0,1], hue='Foundry', units='Sensor name', estimator=None, color='purple')

    if sensorNamesAboveTreshold_CMS.size!=0:
        dataGRCMS_filteredWithinTreshold=dataGRCMS_filtered[~dataGRCMS_filtered['Sensor name'].isin(sensorNamesAboveTreshold_CMS)]
        dataGRCMS_filteredOutsideTreshold=dataGRCMS_filtered[dataGRCMS_filtered['Sensor name'].isin(sensorNamesAboveTreshold_CMS)]
        for sensor_name in sensorNamesAboveTreshold_CMS:
            if not dataGRCMS_filteredOutsideTreshold.empty:
                    sns.lineplot(data=dataGRCMS_filteredOutsideTreshold, x="Voltage", y="GR current", ax=axs300IV[0, 1], hue='Foundry', units='Sensor name', estimator=None, palette=['r'])
        sns.lineplot(data=dataGRCMS_filteredWithinTreshold, x="Voltage", y="GR current", ax = axs300IV[0,1], hue='Foundry', units='Sensor name', estimator=None, color='purple')


    axs300IV[0,1].set_xlabel('Bias voltage (V)', fontsize=30)
    axs300IV[0,1].set_ylabel('Guard ring current (nA)', fontsize=30)
    axs300IV[0,1].hlines(y=1e2, color = 'r', xmin=0, xmax=600, linewidth = 2)
    axs300IV[0,1].set_yscale("log")
    axs300IV[0,1].set_ylim([1e-1, 1e5])
    axs300IV[0,1].set_xlim([0, 1000])
    axs300IV[0,1].xaxis.grid(True)
    axs300IV[0,1].yaxis.grid(True)

    legend_handles, legend_labels = axs300IV[0, 1].get_legend_handles_labels()
    sensorNamesAboveTreshold_HPK_handle = mlines.Line2D([], [], color='none', label=f'Sensor: {sensorNamesAboveTreshold_HPK}')
    sensorNamesAboveTreshold_CMS_handle = mlines.Line2D([], [], color='none', label=f'Sensor: {sensorNamesAboveTreshold_CMS}')

    # Adjusting legend labels and handles based on conditions
    if sensorNamesAboveTreshold_HPK.size == 0 and areSensorsAtCMS:
        if sensorNamesAboveTreshold_HPK.size == 0:
            legend_labels = ['Hamamatsu', 'CMS']  # List of all labels
        else:
            legend_labels = ['Hamamatsu', 'CMS', f'Sensor = {sensorNamesAboveTreshold_CMS} ']
            legend_handles.append(sensorNamesAboveTreshold_CMS_handle)  # Add new handle to handles list

    elif sensorNamesAboveTreshold_HPK.size != 0 and not areSensorsAtCMS:
        legend_labels = ['Hamamatsu', f'Sensor = {sensorNamesAboveTreshold_HPK} ']
        legend_handles.append(sensorNamesAboveTreshold_HPK_handle)

    elif sensorNamesAboveTreshold_HPK.size == 0 and not areSensorsAtCMS:
        legend_labels = ['Hamamatsu']  # List of all labels

    elif sensorNamesAboveTreshold_HPK.size != 0 and areSensorsAtCMS:
        if sensorNamesAboveTreshold_CMS.size == 0:
            legend_labels = ['Hamamatsu', 'CMS', f'Sensor = {sensorNamesAboveTreshold_HPK} ']
            legend_handles.append(sensorNamesAboveTreshold_HPK_handle)
        else:
            legend_labels = ['Hamamatsu', 'CMS', f'Sensor = {sensorNamesAboveTreshold_HPK} ', f'Sensor = {sensorNamesAboveTreshold_CMS} ']
            legend_handles.extend([sensorNamesAboveTreshold_HPK_handle, sensorNamesAboveTreshold_CMS_handle])
     

    axs300IV[0,1].legend(legend_handles, legend_labels, loc='upper left')

       ################################### 3. plot (1,0)

    dataIVHPK = IVTotalDataFrameCMSHPK[IVTotalDataFrameCMSHPK['Foundry'] == 'Hamamatsu']
    dataIVCMS = IVTotalDataFrameCMSHPK[IVTotalDataFrameCMSHPK['Foundry'] == 'CMS']
    dataIVCMS_filtered = dataIVCMS[dataIVCMS['Sensor name'].isin(dataIVHPK['Sensor name'])]
  
    IVTotalDataFrameCMSHPK = pd.concat([dataIVHPK, dataIVCMS_filtered])
 
       
    sns.lineplot(data=IVTotalDataFrameCMSHPK, x="Voltage", y="Total current", ax = axs300IV[1,0], hue='Foundry', units='Sensor name', estimator=None)

    grading_DataFrame_Total['Sensor name'] = grading_DataFrame_Total['Sensor name'].astype(str)
    gradingHPK = grading_DataFrame_Total[grading_DataFrame_Total['Foundry'] == 'Hamamatsu']
    gradingCMS = grading_DataFrame_Total[grading_DataFrame_Total['Foundry'] == 'CMS']
    gradingCMS_filtered = gradingCMS[gradingCMS['Sensor name'].isin(gradingHPK['Sensor name'])]

    failed_grading_HPK = [str(sensor) for sensor in gradingHPK[gradingHPK['Overall grading'] == 'Failed']['Sensor name'].tolist()]
    failed_grading_CMS = [str(sensor) for sensor in gradingCMS_filtered[gradingCMS_filtered['Overall grading'] == 'Failed']['Sensor name'].tolist()]
    
    
    IVTotalDataFrameCMSHPK['Sensor name'] = IVTotalDataFrameCMSHPK['Sensor name'].astype(str)
    IVTotalDataFrameCMSHPK['Sensor name'] = IVTotalDataFrameCMSHPK['Sensor name'].str.strip()

    # check Failed Sensors 
    failed_sensors_df_HPK = dataIVHPK[dataIVHPK['Sensor name'].isin(failed_grading_HPK)]
    failed_sensors_df_CMS =  dataIVCMS_filtered[ dataIVCMS_filtered['Sensor name'].isin(failed_grading_CMS)]
    
    for sensor in failed_grading_HPK:
        selected_failed_rows = dataIVHPK[dataIVHPK['Sensor name'] == sensor.strip()]
        if not selected_failed_rows.empty and selected_failed_rows.iloc[0]['Foundry'] == 'Hamamatsu':
           sns.lineplot(data=selected_failed_rows, x="Voltage", y="Total current", ax=axs300IV[1, 0], hue='Foundry', units='Sensor name', estimator=None, palette=['r'])
        else:
            sns.lineplot(data=selected_failed_rows, x="Voltage", y="Total current", ax=axs300IV[1, 0], hue='Foundry', units='Sensor name', estimator=None, palette=['g']) 

    for sensor in failed_grading_CMS:
        selected_failed_rows = dataIVCMS_filtered[dataIVCMS_filtered['Sensor name'] == sensor.strip()]
        if not selected_failed_rows.empty and selected_failed_rows.iloc[0]['Foundry'] == 'CMS':
           sns.lineplot(data=selected_failed_rows, x="Voltage", y="Total current", ax=axs300IV[1, 0], hue='Foundry', units='Sensor name', estimator=None, palette=['r'])
        else:
            sns.lineplot(data=selected_failed_rows, x="Voltage", y="Total current", ax=axs300IV[1, 0], hue='Foundry', units='Sensor name', estimator=None, palette=['g']) 

     # axs300IV[1,0].legend(loc='upper left', fontsize='large')
    legend_handles, legend_labels = axs300IV[1, 0].get_legend_handles_labels()
    failed_grading_HPK_handle = mlines.Line2D([], [], color='none', label=f'Sensor: {failed_grading_HPK}')
    failed_grading_CMS_handle = mlines.Line2D([], [], color='none', label=f'Sensor: {failed_grading_CMS}')
         

    # Adjusting legend labels and handles based on conditions
    if len(failed_grading_HPK) == 0 and areSensorsAtCMS:
        if len(failed_grading_CMS) == 0:
            legend_labels = ['Hamamatsu', 'CMS']  # List of all labels
        else:
            legend_labels = ['Hamamatsu', 'CMS', f'Sensor: {failed_grading_CMS}']
            legend_handles.append(failed_grading_CMS_handle)

    elif len(failed_grading_HPK) != 0 and not areSensorsAtCMS:
        legend_labels = ['Hamamatsu', f'Sensor: {failed_grading_HPK}']
        legend_handles.append(failed_grading_HPK_handle)

    elif len(failed_grading_HPK) == 0 and not areSensorsAtCMS:
        legend_labels = ['Hamamatsu']  # List of all labels

    elif len(failed_grading_HPK) != 0 and areSensorsAtCMS:
        if len(failed_grading_CMS) == 0:
            legend_labels = ['Hamamatsu', f'Sensor: {failed_grading_HPK}']
            legend_handles.append(failed_grading_HPK_handle)
        else:
            legend_labels = ['Hamamatsu', 'CMS', f'Sensor: {failed_grading_HPK}', f'Sensor: {failed_grading_CMS}']
            legend_handles.extend([failed_grading_HPK_handle, failed_grading_CMS_handle])
        
    axs300IV[1,0].legend(legend_handles, legend_labels, loc='upper left')
            
    axs300IV[1,0].legend(legend_handles, legend_labels, loc='upper left')
    axs300IV[1,0].set_xlabel('Bias voltage (V)', fontsize=30)
    axs300IV[1,0].set_ylabel('Total current (uA)', fontsize=30)
    axs300IV[1,0].hlines(y=1e2, color = 'r', xmin=0, xmax=600, linewidth = 2)
    axs300IV[1,0].set_yscale("log")
    axs300IV[1,0].set_ylim([1e-2, 1e4])
    axs300IV[1,0].set_xlim([0, 1000])
    axs300IV[1,0].xaxis.grid(True)
    axs300IV[1,0].yaxis.grid(True)
    

    
    sensor_names_in_HPK = gradingHPK['Sensor name'].tolist()
    filtered_gradingCMS = gradingCMS[gradingCMS['Sensor name'].isin(sensor_names_in_HPK)]
    combined_grading = gradingHPK._append(filtered_gradingCMS, ignore_index=True)
    
    gradingCMSSameOBA = gradingCMS[gradingCMS['Sensor name'].isin(gradingHPK['Sensor name'])]
    passedGradingHPK = gradingHPK[gradingHPK['Overall grading'] == 'Passed']
    passedGradingCMS = gradingCMSSameOBA[gradingCMSSameOBA['Overall grading'] == 'Passed']
    amountPassedHPK = passedGradingHPK.count()[0]
    amountPassedCMS = passedGradingCMS.count()[0]
    combined_grading.loc[combined_grading['Overall grading']=='Failed', 'Overall grading'] = 0
    combined_grading.loc[combined_grading['Overall grading']=='Passed', 'Overall grading'] = 1
    grading_DataFrame_Total_sorted = combined_grading.sort_values(['Sensor name', 'Foundry'],ascending=[True, False])


   ################################### 4. plot (1,1)
    sns.scatterplot(data=grading_DataFrame_Total_sorted, x='Sensor name', y='Overall grading',  ax = axs300IV[1,1], hue='Foundry', style='Foundry', s=200, alpha=0.8)  
    y_labels = ['Failed', 'Passed']
    axs300IV[1,1].set_yticks([0, 1])
    axs300IV[1,1].set_yticklabels(y_labels)
    axs300IV[1,1].set_xlabel('Sensor name', fontsize=30)
    axs300IV[1,1].set_ylabel('IV grading result', fontsize=30)
    axs300IV[1,1].tick_params(axis='x', rotation=90)
    axs300IV[1,1].set_ylim(-0.5, 1.5)
    axs300IV[1,1].axhspan(axs300IV[1,1].get_ylim()[0], (axs300IV[1,1].get_ylim()[1] - axs300IV[1,1].get_ylim()[0])/2 + axs300IV[1,1].get_ylim()[0], color='red', alpha=0.1)
    axs300IV[1,1].axhspan((axs300IV[1,1].get_ylim()[1] - axs300IV[1,1].get_ylim()[0])/2 + axs300IV[1,1].get_ylim()[0], axs300IV[1,1].get_ylim()[1], color='green', alpha=0.1)

    

    if not dataCMS_filtered.empty:
        plt.text(4, 0.4, f'Passed sensors at Hamamatsu = {amountPassedHPK:.0f} / {amountOfSensorsAtHamamatsu:.0f}', fontsize=16, color='black')
        plt.text(4, 0.6, f'Passed sensors at CMS = {passed_count:.0f} / {totalCMSsensor:.0f}', fontsize=16, color='blue')
    else:
        plt.text(4, 0.5, f'Passed sensors at Hamamatsu = {amountPassedHPK:.0f} / {amountOfSensorsAtHamamatsu:.0f}', fontsize=16, color='black')
    if amountOfSensorsAtHamamatsu < 20:
        plt.text(4, 1.3, f'Amount of sensors = {amountOfSensorsAtHamamatsu:.0f}', fontsize=22, color='red')
    plt.legend(loc='center right', markerscale=2)

    
    # plt.legend(loc="center right", markerscale=2)

    plt.tight_layout()
    pathToIVResults = pathToCode + '/IV_Results/'
    check_and_create_directory(pathToIVResults)
     #plt.savefig(pathToIVResults + 'CellCurrents_' + str(sensorThickness) + '_'+ OBA_number+ "_" + str(plot) +'.png', dpi=100)
    plt.savefig(pathToIVResults + 'CellCurrents_' + str(sensorThickness) + '_'+ OBA_number + "_" + str(plot) + '.pdf', format='pdf')
    plt.close()
    
def determineCurrentTreshold(SensorThickness):
    if thickness=="120":
        return 70
    elif thickness=="200":
        return 160
    else:
        return 370

#Reads txt. files
def main():
    # Path to data
    # path_to_data = './path_to_your_data/'
    files = []
    data_dict = {300: pd.DataFrame({'Voltage': [1, 2, 3], 'Capacitance': [0.1, 0.2, 0.3], 'Error': [0.01, 0.02, 0.03]}),
                 200: pd.DataFrame({'Voltage': [1, 2, 3], 'Capacitance': [0.1, 0.2, 0.3], 'Error': [0.01, 0.02, 0.03]})}
        
    # Plot combined results
      #pathToResults = 'Results/DepletionVoltageHPK/'
    #plot_and_save_combined_results(data_dict, pathToResults)
    
    i=0 # iterator needed for summary graph x values
    thicknessLimit=0 
    
    for fileName in files:
        voltages, capacitances = read(fileName)

        dataForSingleChannel = pd.DataFrame({'Voltage': voltages, 'Capacitance': capacitances})
        sensorName = scratchIDFromName(fileName)
 
        Vdep, eVdep = findDepletionVoltageHPK(dataForSingleChannel, sensorName, pathToResults)
        print("  Vdep      in main  ", Vdep)
        print("  eVdep      in main  ", eVdep)
   
        thickness = thicknessFromName(fileName)
        if thickness is not None:
            dataForSingleChannel['Error'] = eVdep  # Add error column to DataFrame
            data_dict[thickness] = dataForSingleChannel  # Store data in the dictionary  
        
        if i==0:
            thicknessLimit = determineCurrentTreshold()
            
def read_vdep_file(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True)
    return df['Vdep'].tolist()



def singleBatch_oneOBA_forCVPlot(CVdataAllDataFrame, thickness, campaignType, OBA_number, pathToCode, plot_and_fit, findDepletionVoltageHPK, getSensorDetails, grading_DataFrame_Total_CV):
    # Convert sensor names to string for consistent processing
    grading_DataFrame_Total_CV['Sensor name'] = grading_DataFrame_Total_CV['Sensor name'].astype(str)
    
    # Separate the data by foundry
    grading_HPK = grading_DataFrame_Total_CV[grading_DataFrame_Total_CV['Foundry'] == 'Hamamatsu']
    grading_CMS = grading_DataFrame_Total_CV[grading_DataFrame_Total_CV['Foundry'] == 'CMS']
    
    # Filter CMS data to include only sensors that exist in Hamamatsu data
    gradingCMS_filtered = grading_CMS[grading_CMS['Sensor name'].isin(grading_HPK['Sensor name'])]
    
    # Reset and reinitialize the graphs before each OBA loop
    tgeHPK = ROOT.TGraph()
    tgeCMS = ROOT.TGraph()
    tgeVdepCMS = ROOT.TGraphErrors()

    # Identify sensors that failed the grading
    failed_grading_HPK = [str(sensor) for sensor in grading_HPK[grading_HPK['Overall grading'] == 'Failed']['Sensor name'].tolist()]
    failed_grading_CMS = [str(sensor) for sensor in gradingCMS_filtered[gradingCMS_filtered['Overall grading'] == 'Failed']['Sensor name'].tolist()]
    
    # Get the sensor names that exist in both Hamamatsu and CMS
    sensor_names_in_HPK = grading_HPK['Sensor name'].tolist()
    
    # Filter CMS data to include only those sensors which are also in Hamamatsu
    filtered_gradingCMS = grading_CMS[grading_CMS['Sensor name'].isin(sensor_names_in_HPK)]
    
    # Combine the grading data for sorting and plotting
    combined_grading = grading_HPK._append(filtered_gradingCMS, ignore_index=True)

      #print("         combined_grading                   ------- in singleBatch_oneOBA_forCVPlot _--------------_>")
      #print(combined_grading)

    if combined_grading.empty:
        print("combined_grading is empty; cannot proceed.")
        return
    
    # Map 'Failed' to 0 and 'Passed' to 1 for plotting
    combined_grading['Overall grading'] = combined_grading['Overall grading'].map({'Failed': 0, 'Passed': 1}).astype(int)
    
    # Sort the DataFrame by 'Sensor name' and 'Foundry'
    grading_DataFrame_Total_CV_sorted = combined_grading.sort_values(['Sensor name', 'Foundry'], ascending=[True, False])

 

    # Proceed with the rest of your function...

    if isinstance(CVdataAllDataFrame, list):
        CVdataAllDataFrame = pd.DataFrame(CVdataAllDataFrame)
    if CVdataAllDataFrame.empty:
        print("Error: CVdataAllDataFrame is empty.")
        return
    if isinstance(CVdataAllDataFrame.iloc[0, 0], tuple):
        CVdataAllDataFrame = pd.DataFrame([dict(x) for x in CVdataAllDataFrame.values])

    
    CVdataAllDataFrame = CVdataAllDataFrame.sort_values(by='Sensor name')

    dataCVHPK = CVdataAllDataFrame[CVdataAllDataFrame['Foundry'] == 'Hamamatsu']
    dataCVCMS = CVdataAllDataFrame[CVdataAllDataFrame['Foundry'] == 'CMS']
    dataCVCMS_filtered = dataCVCMS[dataCVCMS['Sensor name'].isin(dataCVHPK['Sensor name'])]
     
    dataCVHPK["Estimated depletion voltage (V)"] = dataCVHPK["Estimated depletion voltage (V)"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
    dataCVHPK["Estimated depletion voltage (V)"] = pd.to_numeric(dataCVHPK["Estimated depletion voltage (V)"], errors='coerce')

     #print(" *********** Depletion Voltage dataCVHPK in plot **************")
     #print(dataCVHPK)
    cVdep = ROOT.TCanvas("cVdep", "Depletion Voltage", 1000,700) 
    cVdep.Divide(1, 3)
    
    # 1. Plot for Depletion Voltage
    cVdep.cd(1)
    gPad1 = cVdep.GetPad(1)
    gPad1.SetBottomMargin(0.25)  
    gPad1.SetTopMargin(0.15)     # Reduce top margin
    gPad1.SetGrid()
    
    tgeVdep = ROOT.TGraphErrors()
    tgeVdep.SetMarkerColor(ROOT.kBlue+1)
    tgeVdep.SetLineColor(ROOT.kBlue+1)
    tgeVdep.SetLineStyle(2)
     #tgeVdep.SetMarkerStyle(24)  # circle
    tgeVdep.SetMarkerStyle(2)  # plus
       #tgeVdep.SetMarkerStyle(3)  # plus*
    tgeVdep.SetMarkerSize(2.0)  # Make markers bigger
    tgeVdep.SetLineWidth(2)
    
    sensor_names = []
    depletion_voltages = []
    
    for i, (index, row) in enumerate(dataCVHPK.iterrows()):
        sensor_name = row['Sensor name']
        Vdep = row['Estimated depletion voltage (V)']
        eVdep = row.get('Uncertainty (V)', 0.0)  # Assuming default uncertainty as 0.0 if missing

        sensor_names.append(sensor_name)
        depletion_voltages.append(Vdep)
        tgeVdep.SetPoint(i, i, Vdep)
        #tgeVdep.SetPointError(i, 0, 15)
        tgeVdep.SetPointError(i, 0, eVdep)
       
     # Determine if CMS data exists
    has_CMS_data = not dataCVCMS_filtered.empty

    # Set the title based on the presence of CMS data
    if has_CMS_data:
        tgeVdep.SetTitle(f"#color[4]{{HPK}} /#color[2]{{CMS}}: Depletion voltage: {thickness} um, {campaignType}, {OBA_number}; Sensor Names (IDs); Depletion voltage (V)")
    else:
        tgeVdep.SetTitle(f"#color[4]{{HPK}}: Depletion voltage: {thickness} um, {campaignType}, {OBA_number}; Sensor Names (IDs); Depletion voltage (V)")


    # Set the title without the foundry name
    #tgeVdep.SetTitle(f"Depletion voltage: {thickness} um, {campaignType}, {OBA_number}; Sensor Names (IDs); Depletion voltage (V)")


    #tgeVdep.SetTitle(f"{foundry}, Depletion voltage: {thickness} um, {campaignType}, {OBA_number}; Sensor Names (IDs); Depletion voltage (V)")
    thicknessLimit = 0
    n = len(dataCVHPK)
    tgeVdep.GetXaxis().SetLimits(-0.5, n - 0.5)
    if SensorThickness == 120:
        thicknessLimit = 70
        tgeVdep.GetYaxis().SetRangeUser(1, 100) # for 120um thickness
    elif  SensorThickness == 200:
        thicknessLimit = 160
        tgeVdep.GetYaxis().SetRangeUser(1, 250) # for 200um thickness 
    elif  SensorThickness == 300:
        thicknessLimit = 370
        tgeVdep.GetYaxis().SetRangeUser(1, 450) # for 300um thickness         
    tgeVdep.GetXaxis().SetRangeUser(-0.5, n - 0.5)
    tgeVdep.GetXaxis().SetTitleOffset(3.0)
    tgeVdep.GetXaxis().SetLabelOffset(0.05)
    tgeVdep.GetXaxis().SetLabelSize(0.05)
    tgeVdep.Draw("AP")

    # Customize x-axis
    axis = tgeVdep.GetXaxis()
    axis.SetNdivisions(n, True)
    axis.SetLabelSize(0.03)

    
    for j, sensor_name in enumerate(dataCVHPK['Sensor name']):
        axis.ChangeLabel(j + 1, 75, -0.5, -1, -1, -1, str(sensor_name))

    #Limit of the thickness please change according to the thicknesses

      #thicknessLimit = determineCurrentTreshold(thickness)
    
      #desired_thickness_limit = 370.0  # for 300um thickness
    #desired_thickness_limit = 160.0  # for 200um thickness
    #desired_thickness_limit = 70.0  # for 120um thickness
     #thicknessLimit = desired_thickness_limit
    limit = ROOT.TLine(-0.5, thicknessLimit, n - 0.5, thicknessLimit)
    limit.SetLineColor(ROOT.kRed+1)
    limit.SetLineWidth(2)
    limit.Draw()


    if has_CMS_data:
    # Ensure tgeVdepCMS is freshly initialized for each plot
        tgeVdepCMS = ROOT.TGraphErrors()
        tgeVdepCMS.SetMarkerColor(ROOT.kRed+1)
        tgeVdepCMS.SetLineColor(ROOT.kRed+1)
        tgeVdepCMS.SetLineStyle(3)
      #  tgeVdepCMS.SetMarkerStyle(2)   # marker +
        tgeVdepCMS.SetMarkerStyle(22)   # marker triangle
        tgeVdepCMS.SetMarkerSize(1.0)  # Ensure marker size is set correctly
 
        sensor_data = []
        point_index = 0  # Initialize point_index outside the loop
     # for sensor_name in dataCVHPK['Sensor name']:
        for i, sensor_name in enumerate(dataCVHPK['Sensor name']):
            sensor_data = dataCVCMS_filtered[dataCVCMS_filtered['Sensor name'] == sensor_name]
            if sensor_data.empty:
                continue
            else:
            # i = sensor_names.index(sensor_name)
                for j, (index, row) in enumerate(sensor_data.iterrows()):
                    Vdep = row['Estimated depletion voltage (V)']
                  #print("  Vdep  in   dataCVCMS_filtered================>",  Vdep)
                    eVdep = row.get('Uncertainty (V)', 0.0)  # Assuming default uncertainty as 0.0 if missing
                    tgeVdepCMS.SetPoint(point_index, i, Vdep + j * 0.1)  # Spread the points around each sensor by slightly offsetting the y-coordinate
                #tgeVdepCMS.SetPointError(point_index, 0, 15)
                    tgeVdepCMS.SetPointError(point_index, 0, eVdep)
                    point_index += 1

        tgeVdepCMS.Draw("P same")

    # Legend
    legend = ROOT.TLegend(0.1, 0.7, 0.3, 0.9)
    legend.AddEntry(tgeVdep, "HPK", "p")  # Assuming dataCVHPK corresponds to HPK
    if has_CMS_data:
        legend.AddEntry(tgeVdepCMS, "CMS", "p")
    legend.Draw()

    gPad1.Update()  # Ensure the first plot updates
    cVdep.Update()  # Update the canvas after the first plot

   # # # # # # # # # # # # # # # # # #
    # 2. plot  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
     #cms_data = grading_DataFrame_Total_CV_sorted[grading_DataFrame_Total_CV_sorted['Foundry'] == 'CMS']
    cVdep.cd(2)
    gPad2 = cVdep.GetPad(2)
    gPad2.SetBottomMargin(0.25)
    gPad2.SetTopMargin(0.15)
    gPad2.SetGrid()

    # Create sensor_name_to_index mapping
    sensor_names = sorted(grading_DataFrame_Total_CV_sorted['Sensor name'].unique())
    sensor_name_to_index = {name: idx for idx, name in enumerate(sensor_names)}
    n_sensors = len(sensor_names)

    # Initialize TGraphs and point counters
    tgeHPK = ROOT.TGraphErrors()
    tgeCMS = ROOT.TGraphErrors()
    hpk_point_counter = 0
    cms_point_counter = 0

    # Populate TGraphs
    for idx, row in grading_DataFrame_Total_CV_sorted.iterrows():
        sensor_name = row['Sensor name']
        overall_grading = row['Overall grading']
        foundry = row['Foundry']
        x_index = sensor_name_to_index[sensor_name]

        if foundry == 'Hamamatsu':
            tgeHPK.SetPoint(hpk_point_counter, x_index, overall_grading)
            tgeHPK.SetPointError(hpk_point_counter, 0, 0)
            hpk_point_counter += 1
        elif foundry == 'CMS':
            tgeCMS.SetPoint(cms_point_counter, x_index, overall_grading)
            tgeCMS.SetPointError(cms_point_counter, 0, 0.01)
            cms_point_counter += 1
    # Determine if we have HPK and/or CMS data
    has_HPK_data = tgeHPK.GetN() > 0
    has_CMS_data = tgeCMS.GetN() > 0


    # Set the title based on the presence of HPK and CMS data
    if has_HPK_data and has_CMS_data:
        tgeHPK.SetTitle(f"#color[4]{{HPK}} /#color[2]{{CMS}}: CV Grading: {thickness} um, {campaignType}, {OBA_number}")
    elif has_HPK_data:
        tgeHPK.SetTitle(f"#color[4]{{HPK}}: CV Grading: {thickness} um, {campaignType}, {OBA_number}")
    elif has_CMS_data:
        tgeHPK.SetTitle(f"#color[2]{{CMS}}: CV Grading: {thickness} um, {campaignType}, {OBA_number}")

    # Configure HPK TGraph if available
    if has_HPK_data:
        tgeHPK.SetMarkerStyle(24)
        tgeHPK.SetMarkerColor(ROOT.kBlue)
        tgeHPK.SetLineColor(ROOT.kBlue)
        tgeHPK.GetXaxis().SetTitle("Sensor Names (IDs)")
        tgeHPK.GetYaxis().SetTitle("Overall CV Grading")
        tgeHPK.GetXaxis().SetLimits(-0.5, n_sensors - 0.5)
        tgeHPK.GetYaxis().SetRangeUser(-0.5, 1.5)
        tgeHPK.Draw("AP")
        # Configure CMS TGraph if available
    if has_CMS_data:
        tgeCMS.SetMarkerStyle(23)
        tgeCMS.SetMarkerColor(ROOT.kRed)
        tgeCMS.SetLineColor(ROOT.kRed)
        tgeCMS.Draw("P same")


    # Set X-axis labels
    axis = tgeHPK.GetXaxis()
    axis.SetNdivisions(n_sensors, False)
    axis.SetLabelSize(0.04)
    tgeHPK.GetXaxis().SetTitleOffset(3.0)
    tgeHPK.GetXaxis().SetLabelOffset(0.05)
    for sensor_name, idx in sensor_name_to_index.items():
        axis.SetBinLabel(axis.FindBin(idx), str(sensor_name))
         #axis.ChangeLabel(idx + 1, 75, -0.5, -1, -1, -1, str(sensor_name))

    # Adjust label size and offset to prevent overlap
    axis.SetLabelSize(0.04)  # Adjust the label size
    axis.SetLabelOffset(0.02)  # Adjust the distance from the axis

    for idx in range(len(sensor_names)):
        bin_position = axis.FindBin(idx)  # Find the bin corresponding to the index
        axis.ChangeLabel(bin_position, 45, 0.05, -1, -1, -1, str(sensor_names[idx]))  # Rotate the label by 45 degrees

    # Set Y-axis labels
    tgeHPK.GetYaxis().SetNdivisions(2, False)
    tgeHPK.GetYaxis().ChangeLabel(1, -1, 0.04, -1, -1, -1, "FAIL")
    tgeHPK.GetYaxis().ChangeLabel(2, -1, 0.04, -1, -1, -1, "PASS")

    # Draw shaded regions
    ymin, ymax = -0.5, 1.5
    mid_y = (ymax + ymin) / 2

    red_box = ROOT.TBox(-0.5, ymin, n_sensors - 0.5, mid_y)
    red_box.SetFillColorAlpha(ROOT.kRed, 0.3)
    red_box.Draw("same")

    green_box = ROOT.TBox(-0.5, mid_y, n_sensors - 0.5, ymax)
    green_box.SetFillColorAlpha(ROOT.kGreen, 0.3)
    green_box.Draw("same")


    # Redraw TGraphs to be on top
    if has_HPK_data:
        tgeHPK.Draw("P same")
    if has_CMS_data:
        tgeCMS.Draw("P same")

    # Draw legend
    legend = ROOT.TLegend(0.75, 0.75, 0.95, 0.85)
    if has_HPK_data:
        legend.AddEntry(tgeHPK, "Hamamatsu", "p")
    if has_CMS_data:
        legend.AddEntry(tgeCMS, "CMS", "p")
    legend.Draw()
 

    gPad2.Update()
    gPad2.Modified()  # Force the pad to update its content
    cVdep.Update()
    cVdep.Modified()  # Force the canvas to update its content

    
     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 3. Plot for Capacitance vs Voltage  # # # # # # # # # # # # # # # # # # # # # # # # # #
    input_file = ROOT.TFile.Open(pathToCode + "/CV_Graphs.root", "READ")

    cVdep.cd(3)
    gPad3 = cVdep.GetPad(3)
    gPad3.Clear()  # Clear pad
    gPad3.SetBottomMargin(0.25)
    gPad3.SetTopMargin(0.1)
    gPad3.SetGrid()

    # Check if file is successfully opened
    if not input_file or input_file.IsZombie():
        print("Error opening file.")
        exit(1)
        
    # List all TGraph objects in the file
    keys = input_file.GetListOfKeys()
    first_graph = True
    
    legend = ROOT.TLegend(0.1, 0.009, 0.9, 0.21)
    #legend.SetHeader("TGraph Legends", "C")  # Set the legend header
    legend.SetBorderSize(1)
    legend.SetFillColor(0)
    legend.SetNColumns(6)
    legend.SetTextSize(0.03)
    sensor_index = 0
    # Loop over graphs in the ROOT file
    keys = input_file.GetListOfKeys()
    first_graph = True
    graphs = []

    for key in keys:
        obj = key.ReadObj()

        if isinstance(obj, ROOT.TGraph):
            print(f"Processing graph: {obj.GetName()} with {obj.GetN()} points")  # Debugging output

            # Filter and create a new graph with valid points
            valid_points = []  # To store valid (x, y) pairs
            x, y = ctypes.c_double(0), ctypes.c_double(0)
            
            for i in range(obj.GetN()):
                obj.GetPoint(i, x, y)  # Modify x and y in place
                x_val, y_val = x.value, y.value  # Extract the values from ctypes.c_double()
                print(f"Point {i}: x = {x_val}, y = {y_val}")

                # Check for valid points
                if not (x_val != x_val or y_val != y_val or abs(x_val) > 1e10 or abs(y_val) > 1e10):
                    valid_points.append((x_val, y_val))
                else:
                    print(f"Invalid point detected at index {i}: x = {x_val}, y = {y_val}.")

            if valid_points:
                # Create and configure the new graph
                new_graph = ROOT.TGraph(len(valid_points))
                for idx, (x_val, y_val) in enumerate(valid_points):
                    new_graph.SetPoint(idx, x_val, y_val)

                # Set properties similar to the original graph
                new_graph.SetLineColor(ROOT.kBlue + 2)
                new_graph.SetTitle("#color[4]{HPK}: Capacitance vs Bias Voltage")
                 #legend.AddEntry(new_graph, name, "l")

                # Set axis limits
                if SensorThickness == 120:
                    new_graph.GetXaxis().SetLimits(0, 100)
                    new_graph.GetYaxis().SetRangeUser(0, 0.1)
                elif SensorThickness == 200:
                    new_graph.GetXaxis().SetLimits(0, 400)
                    new_graph.GetYaxis().SetRangeUser(0, 0.2)
                elif SensorThickness == 300:
                    new_graph.GetXaxis().SetLimits(0, 400)
                    new_graph.GetYaxis().SetRangeUser(0, 0.3)

                # Draw the graph
                draw_option = 'AL' if first_graph else 'L SAME'
                print(f"Drawing graph: {new_graph.GetName()} with option: {draw_option}")  # Debugging output
                new_graph.Draw(draw_option)
                if sensor_index < len(sensor_names):
                    legend.AddEntry(new_graph, sensor_names[sensor_index], "l")
                    sensor_index += 1
                first_graph = False

                graphs.append(new_graph)
                
                # Force a draw to check for any issues
                gPad3.Modified()
                gPad3.Update()



    legend.Draw()
    gPad3.Modified()  # Mark the pad as modified
    gPad3.Update()
    cVdep.Modified()   
    cVdep.Update()
    input_file.Close()   
    os.remove(pathToCode + "/CV_Graphs.root")
    
    # Ensure the directory exists for saving results
    pathToCVResults = os.path.join(pathToCode, 'CV_Results/')
    if not os.path.exists(pathToCVResults):
        os.makedirs(pathToCVResults) 
    # Save plot as PDF
    plotname = os.path.join(pathToCVResults, f'Depletionvoltage_{thickness}_{OBA_number}.pdf')
    cVdep.SaveAs(plotname)
    
if __name__ == "__main__":
    main()



# Define a custom function to assign OBA number to CMS sensors
def assign_OBA_to_CMS(row, hamamatsu_sensor_mapping):
    if row['Foundry'] == 'CMS':
        sensor_name = row['Sensor name']
        # Check if the CMS sensor already has an assigned OBA number
        if pd.notna(row['OBA']):
            return row['OBA']  # Return the existing OBA number
        else:
            # Check if the CMS sensor is present in any Hamamatsu OBA
            for oba_number, hamamatsu_sensors in hamamatsu_sensor_mapping.items():
                if sensor_name in hamamatsu_sensors:
                    return oba_number
            return pd.NA  # Return NA if OBA number cannot be assigned for CMS
    elif row['Foundry'] == 'Hamamatsu':
        return row['OBA']  # For Hamamatsu sensors, keep the original OBA number
    return pd.NA  # Return NA for other foundries


### loop over CMS    OBA's  IV:::::::::   
os.chdir(dataCMS_IV)
sensorNames600VForDataFrame=[]
sensorNames800VForDataFrame=[]
sensorNames1000VForDataFrame=[]
sensorNamesForGRForDataFrame=[]
sensorNamesForGRPlot=[]
DataVoltageForDataFrame=[]
DataTotalCurrentForDataFrame=[]
DataGRVoltageForDataFrame = []
DataGRCurrentForDataFrame = []
IVDataFor600VForDataFrame = []
IVDataFor800VForDataFrame = []
IVDataFor1000VForDataFrame = []
amountOfSensorsAtCMS=0
sensorsFailedIVGrading=[]
sensorsFailedCVGrading=[]
sensorsFailedCVFit=[]
ChannelDataFor600VCMSForDataFrame = []
ChannelDataFor800VCMSForDataFrame = []
ChannelDataFor1000VCMSForDataFrame = []
 
for file in sorted(glob.glob("*.txt")):
   amountOfSensorsAtCMS += 1
   
   if file[-5]=="e": continue
   sensorName=getSensornameIV(file)
   sensorNamesForGRPlot.append(sensorName)
   IVData = readDataFile(dataCMS_IV+"/"+file)

   IVData.loc[IVData.Channel != 0,'CurrentTempScale'] = IVData.loc[IVData.Channel != 0,'CurrentTempScale'].abs().mul(1e9)
   IVData.loc[IVData.Channel != 0,'Tot_currTempScale'] = IVData.loc[IVData.Channel != 0,'Tot_currTempScale'].abs().mul(1e6)
   #IVData.loc[:,'Channel'] = IVData['Channel']
   IVDataFor600VSingleSensor = IVData.loc[(IVData['Voltage'].round(0) == -600.0) & (IVData['Channel'] < numberOfChannels) & (IVData['Channel'] > 0)]
   IVDataFor600VSingleChannel = IVData.loc[(IVData['Voltage'].round(0) == -600.0) & (IVData['Channel'] < numberOfChannels) & (IVData['Channel'] > 0)]
   IVDataFor800VSingleSensor = IVData.loc[(IVData['Voltage'].round(0) == -800.0) & (IVData['Channel'] < numberOfChannels) & (IVData['Channel'] > 0)]
   IVDataFor800VSingleChannel = IVData.loc[(IVData['Voltage'].round(0) == -800.0) & (IVData['Channel'] < numberOfChannels) & (IVData['Channel'] > 0)]
   IVDataFor1000VSingleSensor = IVData.loc[(IVData['Voltage'].round(0) == -1000.0) & (IVData['Channel'] < numberOfChannels) & (IVData['Channel'] > 0)]
   IVDataFor1000VSingleChannel = IVData.loc[(IVData['Voltage'].round(0) == -1000.0) & (IVData['Channel'] < numberOfChannels) & (IVData['Channel'] > 0)]

   CurrentFor600VSingleSensor = list(IVDataFor600VSingleSensor['CurrentTempScale'])
   CurrentFor600VSingleChannel = list(IVDataFor600VSingleSensor['Channel'])
   CurrentFor800VSingleSensor = list(IVDataFor800VSingleSensor['CurrentTempScale'])
   CurrentFor800VSingleChannel = list(IVDataFor800VSingleSensor['Channel'])
   CurrentFor1000VSingleSensor = list(IVDataFor1000VSingleSensor['CurrentTempScale'])
   CurrentFor1000VSingleChannel = list(IVDataFor1000VSingleSensor['Channel'])

   IVDataGR300 = IVData.loc[IVData['Channel'] == numberOfChannels]
   ChannelDataFor600VCMSForDataFrame.extend(CurrentFor600VSingleChannel)
   ChannelDataFor800VCMSForDataFrame.extend(CurrentFor800VSingleChannel)
   ChannelDataFor1000VCMSForDataFrame.extend(CurrentFor1000VSingleChannel)
   IVDataGRRampUp200 = IVDataGR300.abs().drop_duplicates(subset=['Voltage'])
   sensorNames600VSingleSensor200 = [sensorName for current in CurrentFor600VSingleSensor]
   sensorNames600VForDataFrame.extend(list(sensorNames600VSingleSensor200))
   sensorNames800VSingleSensor200 = [sensorName for current in CurrentFor800VSingleSensor]
   sensorNames800VForDataFrame.extend(list(sensorNames800VSingleSensor200))
   sensorNames1000VSingleSensor200 = [sensorName for current in CurrentFor1000VSingleSensor]
   sensorNames1000VForDataFrame.extend(list(sensorNames1000VSingleSensor200))

   IVDataFor600VForDataFrame.extend(CurrentFor600VSingleSensor)
   IVDataFor800VForDataFrame.extend(CurrentFor800VSingleSensor)
   IVDataFor1000VForDataFrame.extend(CurrentFor1000VSingleSensor)

   DataGRVoltageForDataFrame.extend(list(IVDataGRRampUp200.abs().loc[:, 'Voltage']))
   DataGRCurrentForDataFrame.extend(list(IVDataGRRampUp200.abs().loc[:, 'CurrentTempScale']))
   sensorNamesForGR = [sensorName for current in list(IVDataGRRampUp200.abs().loc[:, 'Voltage'])]
   sensorNamesForGRForDataFrame.extend(list(sensorNamesForGR))
   IVData = IVData[IVData.Channel <= numberOfChannels-1]
   IVData_ForTotal=IVData.groupby("Voltage").mean()
   DataTotalCurrentForDataFrame.extend(list(IVData_ForTotal.abs().loc[:, 'Tot_currTempScale']))
   DataVoltageForDataFrame.extend(list(np.flip(IVData['Voltage'].abs().unique())))
 # Create the DataFrame data600VCMS with 'Sensor name', 'Cell current [nA]', and 'Channel' columns
data600VCMS = pd.DataFrame(list(zip(sensorNames600VForDataFrame, IVDataFor600VForDataFrame, ChannelDataFor600VCMSForDataFrame)), columns =['Sensor name', 'Cell current [nA]', 'Channel']).assign(Foundry='CMS')
data800VCMS = pd.DataFrame(list(zip(sensorNames800VForDataFrame, IVDataFor800VForDataFrame, ChannelDataFor800VCMSForDataFrame)), columns =['Sensor name', 'Cell current [nA]', 'Channel']).assign(Foundry='CMS')
data1000VCMS = pd.DataFrame(list(zip(sensorNames1000VForDataFrame, IVDataFor1000VForDataFrame, ChannelDataFor1000VCMSForDataFrame)), columns =['Sensor name', 'Cell current [nA]', 'Channel']).assign(Foundry='     CMS')  

IVDataGRDataFrame = pd.DataFrame(list(zip(sensorNamesForGRForDataFrame, DataGRVoltageForDataFrame, DataGRCurrentForDataFrame)), columns =['Sensor name','Voltage', 'GR current'])
IVDataGRDataFrame['Foundry']='CMS'

IVTotalDataFrame = pd.DataFrame(list(zip(sensorNamesForGRForDataFrame, DataVoltageForDataFrame, DataTotalCurrentForDataFrame)), columns =['Sensor name','Voltage', 'Total current'])
IVTotalDataFrame['Foundry']='CMS'

folderName="temperature_scaled"
amountOfTotalSensor=0
amountOfFailedSensor=0
amountOfOBA=0
if 'grading_DataFrame_AllOBA' not in globals():
    grading_DataFrame_AllOBA = pd.DataFrame()  # Create an empty DataFrame before the loop
if 'IVTotalDataFrameCMSHPK_AllOBA' not in globals():
    IVTotalDataFrameCMSHPK_AllOBA = pd.DataFrame()  # Create an empty DataFrame before the loop

if 'grading_DataFrame_AllOBA_CV' not in globals():
    grading_DataFrame_AllOBA_CV = pd.DataFrame()  # Create an empty DataFrame before the loop
    
selected_channels_label_list = []

channel_NonZero_DataFrame = pd.DataFrame(columns=[
    'OBA_number', 
    'Channel', 
    'Sensor name_600V', 
    'Cell current [nA]'
])

channel_DataFrame = pd.DataFrame(columns=[
    'OBA_number', 
    'Channel', 
    'Sensor name_600V', 
    'Cell current [nA]_600V', 
    'Cell current [nA]_800V', 
    'Currents_ratio'
])
# List all directories in the dataHPK_IV directory (non-recursively)
all_dirs = [d for d in os.listdir(dataHPK_IV) if os.path.isdir(os.path.join(dataHPK_IV, d))]

# Filter directories that match the pattern
#relevant_dirs = [d for d in all_dirs if pattern.match(d)]
dirs = [d for d in os.listdir(dataHPK_IV) if os.path.isdir(os.path.join(dataHPK_IV, d))]

#relevant_dirs = [d for d in dirs if d.startswith(directory_pattern_HPK_IV)]

relevant_dirs = [d for d in all_dirs if pattern.match(d)]

folderName="temperature_scaled"
### loop over HPK OBA's  IV:::::::::
for root in relevant_dirs:
# Loop over the relevant directories
 
    # Construct the full path to the OBA directory
    full_OBA_path = os.path.join(dataHPK_IV, root)
    
    # Check if the "temperature_scaled" directory exists in the current OBA directory
    temperature_scaled_path = os.path.join(full_OBA_path, folderName)
    if os.path.isdir(temperature_scaled_path):
        
        # Extract OBA number safely
        if "OBA" in root:
            OBA_number = "OBA" + root.split("OBA", 1)[1][:5]
        else:
            print(f"Warning: 'OBA' not found in the directory name '{root}'. Skipping...")
            continue

        amountOfOBA += 1
        IVDataFor600VHamamatsuForDataFrame = []
        ChannelDataFor600VHamamatsuForDataFrame = []
        sensorNames600VHamamatsuForDataFrame = []
        IVDataFor800VHamamatsuForDataFrame = []
        ChannelDataFor800VHamamatsuForDataFrame = []
        sensorNames800VHamamatsuForDataFrame = []
        IVDataFor1000VHamamatsuForDataFrame = []
        ChannelDataFor1000VHamamatsuForDataFrame = []
        sensorNames1000VHamamatsuForDataFrame = []

        DataGRHPKVForDataFrame = []
        DataGRHPKIForDataFrame = []
        DataTotalHPKIForDataFrame = []
        DataTotalHPKVForDataFrame = []
        sensorNamesForTotalHPKForDataFrame = []
        sensorNamesForGRHPKForDataFrame = []
         #channel_numbers = []
        amountOfSensorsAtHamamatsu=0
        IVdataAllDataFrame  = []
        IVdata800VDataFrame  = []
        IVdata1000VDataFrame  = []

         #os.chdir(os.path.join(dataHPK_IV, folderName))
        os.chdir(temperature_scaled_path)
        # loop over txt file in temperature scaled
          #print(" OBA number in IV loop ----------------------------->", OBA_number)
               
        for fileHPK in sorted(glob.glob("*.txt")):
                channel_numbers = []
                amountOfSensorsAtHamamatsu += 1
                amountOfTotalSensor+= 1
                sensorNameHPK = getSensornameIVHPK(fileHPK)
                IVDataHPK = readDataFileHPK(fileHPK)
                # Strip any leading/trailing whitespace from column names
                IVDataHPK.columns = IVDataHPK.columns.str.strip()


                IVDataHPK.loc[:,'CurrentTempScale'] = IVDataHPK['CurrentTempScale'].abs().mul(1e9)
                IVDataHPK.loc[:,'Tot_currTempScale'] = IVDataHPK['Tot_currTempScale'].abs().mul(1e6)

              
                  #IVDataHPK.loc[IVDataHPK.Channel != 0,'Tot_currTempScale'] = IVDataHPK.loc[IVDataHPK.Channel != 0,'Tot_currTempScale'].abs().mul(1e6)
                  #IVDataHPK.loc[IVDataHPK.Channel != 0,'CurrentTempScale'] = IVDataHPK.loc[IVDataHPK.Channel != 0,'CurrentTempScale'].abs().mul(1e9)
                  #IVData.loc[:,'Channel'] = IVData['Channel']
                IVDataFor600VSingleSensorHPK = IVDataHPK.loc[(IVDataHPK['Voltage'] == 600) & (IVDataHPK['Channel'] < numberOfChannels) & (IVDataHPK['Channel'] > 0)]
                IVDataFor600VSingleChannelHPK = IVDataHPK.loc[(IVDataHPK['Voltage'] == 600) & (IVDataHPK['Channel'] < numberOfChannels) & (IVDataHPK['Channel'] > 0)]
                IVDataFor800VSingleSensorHPK = IVDataHPK.loc[(IVDataHPK['Voltage'] == 800) & (IVDataHPK['Channel'] < numberOfChannels) & (IVDataHPK['Channel'] > 0)]
                IVDataFor800VSingleChannelHPK = IVDataHPK.loc[(IVDataHPK['Voltage'] == 800) & (IVDataHPK['Channel'] < numberOfChannels) & (IVDataHPK['Channel'] > 0)]
                IVDataFor1000VSingleSensorHPK = IVDataHPK.loc[(IVDataHPK['Voltage'] == 1000) & (IVDataHPK['Channel'] < numberOfChannels) & (IVDataHPK['Channel'] > 0)]
                IVDataFor1000VSingleChannelHPK = IVDataHPK.loc[(IVDataHPK['Voltage'] == 1000) & (IVDataHPK['Channel'] < numberOfChannels) & (IVDataHPK['Channel'] > 0)]


                CurrentFor600VSingleSensorHPK = list(IVDataFor600VSingleSensorHPK['CurrentTempScale'])
                CurrentFor600VSingleChannelHPK = list(IVDataFor600VSingleSensorHPK['Channel'])
                IVDataFor600VHamamatsuForDataFrame.extend(CurrentFor600VSingleSensorHPK)
                ChannelDataFor600VHamamatsuForDataFrame.extend(CurrentFor600VSingleChannelHPK)
                CurrentFor800VSingleSensorHPK = list(IVDataFor800VSingleSensorHPK['CurrentTempScale'])
                CurrentFor800VSingleChannelHPK = list(IVDataFor800VSingleSensorHPK['Channel'])
                IVDataFor800VHamamatsuForDataFrame.extend(CurrentFor800VSingleSensorHPK)
                ChannelDataFor800VHamamatsuForDataFrame.extend(CurrentFor800VSingleChannelHPK)
                CurrentFor1000VSingleSensorHPK = list(IVDataFor1000VSingleSensorHPK['CurrentTempScale'])
                CurrentFor1000VSingleChannelHPK = list(IVDataFor1000VSingleSensorHPK['Channel'])
                IVDataFor1000VHamamatsuForDataFrame.extend(CurrentFor1000VSingleSensorHPK)
                ChannelDataFor1000VHamamatsuForDataFrame.extend(CurrentFor1000VSingleChannelHPK)

                sensorNames600VSingleSensorHPK = [sensorNameHPK for current in CurrentFor600VSingleSensorHPK]
                sensorNames600VHamamatsuForDataFrame.extend(list(sensorNames600VSingleSensorHPK))

                sensorNames800VSingleSensorHPK = [sensorNameHPK for current in CurrentFor800VSingleSensorHPK]
                sensorNames800VHamamatsuForDataFrame.extend(list(sensorNames800VSingleSensorHPK))

                sensorNames1000VSingleSensorHPK = [sensorNameHPK for current in CurrentFor1000VSingleSensorHPK]
                sensorNames1000VHamamatsuForDataFrame.extend(list(sensorNames800VSingleSensorHPK))

                IVDataGR = IVDataHPK.loc[IVDataHPK['Channel'] == numberOfChannels]
                IVDataGRList = list(IVDataGR['CurrentTempScale'])  
                sensorNamesForGRHPK = [sensorNameHPK for current in IVDataGRList]
                sensorNamesForGRHPKForDataFrame.extend(list(sensorNamesForGRHPK))
                DataGRHPKVForDataFrame.extend(list(IVDataGR.astype(float).loc[:, 'Voltage']))
                DataGRHPKIForDataFrame.extend(list(IVDataGR.astype(float).loc[:, 'CurrentTempScale']))
                IVDataHPKforTotalMean = IVDataHPK.drop(IVDataHPK[IVDataHPK['Channel'] < 148].index)
                IVDataHPKforTotalMeanGrouped = IVDataHPKforTotalMean.groupby('Voltage').mean()
                IVDataTotalCurrentList = list(IVDataHPKforTotalMeanGrouped)
                sensorNamesForTotalHPK = [sensorNameHPK for current in list(IVDataHPK.Voltage.unique().astype(float))]
                sensorNamesForTotalHPKForDataFrame.extend(list(sensorNamesForTotalHPK))
                DataTotalHPKVForDataFrame.extend(list(IVDataHPK.Voltage.unique().astype(float)))
                DataTotalHPKIForDataFrame.extend(list(IVDataHPKforTotalMeanGrouped.astype(float).loc[:, 'Tot_currTempScale']))
        # end of loop over txt file in temperature scaled
        # Create the DataFrame data600VHamamatsu with 'Sensor name', 'Cell current [nA]', and 'Channel' columns
        data600VHamamatsu = pd.DataFrame(list(zip(sensorNames600VHamamatsuForDataFrame, IVDataFor600VHamamatsuForDataFrame, ChannelDataFor600VHamamatsuForDataFrame)), columns =['Sensor name', 'Cell current [nA]', 'Channel']).assign(Foundry='Hamamatsu')
        data800VHamamatsu = pd.DataFrame(list(zip(sensorNames800VHamamatsuForDataFrame, IVDataFor800VHamamatsuForDataFrame, ChannelDataFor800VHamamatsuForDataFrame)), columns =['Sensor name', 'Cell current [nA]', 'Channel']).assign(Foundry='Hamamatsu')

        data1000VHamamatsu = pd.DataFrame( list(zip(sensorNames1000VHamamatsuForDataFrame, IVDataFor1000VHamamatsuForDataFrame, ChannelDataFor1000VHamamatsuForDataFrame)),columns=['Sensor name', 'Cell current [nA]', 'Channel']).assign(Foundry='Hamamatsu')

  
        IVDataGRHPKDataFrame = pd.DataFrame(list(zip(sensorNamesForGRHPKForDataFrame, DataGRHPKVForDataFrame, DataGRHPKIForDataFrame)), columns =['Sensor name','Voltage', 'GR current'])
        IVDataGRHPKDataFrame['Foundry']='Hamamatsu'
        IVTotalDataFrameHPK = pd.DataFrame(list(zip(sensorNamesForTotalHPKForDataFrame, DataTotalHPKVForDataFrame, DataTotalHPKIForDataFrame)), columns =['Sensor name','Voltage', 'Total current'])
        IVTotalDataFrameHPK['Foundry']='Hamamatsu'

          #AmountOFSensorsOnOnePlot=50
        AmountOFSensorsOnOnePlot=1000
        if amountOfSensorsAtHamamatsu>=amountOfSensorsAtCMS:
            amountOfPlots,AmountOFSensorsOnLAstPlot = divmod(amountOfSensorsAtHamamatsu,AmountOFSensorsOnOnePlot)
        else:
            amountOfPlots,AmountOFSensorsOnLAstPlot = divmod(amountOfSensorsAtCMS,AmountOFSensorsOnOnePlot)
        amountOfChannels=numberOfChannels-1
        amountOfVoltagePoints=11
        current_dir = os.path.abspath(os.getcwd())
        data_grading_thickness_HPK = determineGradingHPK(current_dir, pathToCode, 'Hamamatsu')
        data_grading_thickness_HPK['Foundry']='Hamamatsu'
        data_grading_thickness_HPK['OBA']=OBA_number
 
        
        areSensorsAtCMS = False
        firstSensorIndex=amountOfChannels*AmountOFSensorsOnOnePlot*amountOfPlots
        lastSensorindex=firstSensorIndex+AmountOFSensorsOnLAstPlot*amountOfChannels-1
        firstSensorIndexLinePlots=amountOfVoltagePoints*AmountOFSensorsOnOnePlot*amountOfPlots
        lastSensorIndexLinePlots=firstSensorIndexLinePlots+AmountOFSensorsOnLAstPlot*amountOfVoltagePoints-1
        matching_sensors = list(set(data600VCMS['Sensor name']) & set(data600VHamamatsu['Sensor name']))
        if matching_sensors:
            areSensorsAtCMS = True
            IVdataAllDataFrame = pd.concat([data600VHamamatsu[firstSensorIndex:lastSensorindex], data600VCMS[firstSensorIndex:lastSensorindex]], ignore_index=True)
            IVdata800VDataFrame = pd.concat([data800VHamamatsu[firstSensorIndex:lastSensorindex], data800VCMS[firstSensorIndex:lastSensorindex]], ignore_index=True)
            IVdata1000VDataFrame = pd.concat([data1000VHamamatsu[firstSensorIndex:lastSensorindex], data1000VCMS[firstSensorIndex:lastSensorindex]], ignore_index=True)
            
            IVGRDataFrameCMSHPK= pd.concat([IVDataGRHPKDataFrame[firstSensorIndexLinePlots:lastSensorIndexLinePlots], IVDataGRDataFrame[firstSensorIndexLinePlots:lastSensorIndexLinePlots]], ignore_index=True)
            IVTotalDataFrameCMSHPK= pd.concat([IVTotalDataFrameHPK[firstSensorIndexLinePlots:lastSensorIndexLinePlots], IVTotalDataFrame[firstSensorIndexLinePlots:lastSensorIndexLinePlots]], ignore_index=True)
            data_grading_thickness_CMS = determineGradingHPK(data_IV_CMS, pathToCode, 'CMS')
            data_grading_thickness_CMS['Foundry']='CMS'
            grading_DataFrame_Total = pd.concat([data_grading_thickness_HPK, data_grading_thickness_CMS])
        else:
            IVdataAllDataFrame = pd.concat([data600VHamamatsu[firstSensorIndex:lastSensorindex]])
            IVdata800VDataFrame = pd.concat([data800VHamamatsu[firstSensorIndex:lastSensorindex]])
            IVdata1000VDataFrame = pd.concat([data1000VHamamatsu[firstSensorIndex:lastSensorindex]])
            
            IVGRDataFrameCMSHPK= pd.concat([IVDataGRHPKDataFrame[firstSensorIndexLinePlots:lastSensorIndexLinePlots]])
            IVTotalDataFrameCMSHPK= pd.concat([IVTotalDataFrameHPK[firstSensorIndexLinePlots:lastSensorIndexLinePlots]])
            grading_DataFrame_Total = data_grading_thickness_HPK
            
        #assigning OBA's to CMS for grading_DataFrame_Total
        # Create a mapping of Hamamatsu OBA numbers to their corresponding sensor names dynamically
        hamamatsu_sensor_mapping = {}
        for oba_number, group in grading_DataFrame_Total[grading_DataFrame_Total['Foundry'] == 'Hamamatsu'].groupby('OBA'):
            hamamatsu_sensor_mapping[oba_number] = group['Sensor name'].tolist()

        # Apply the custom function to assign OBA numbers to CMS sensors
        grading_DataFrame_Total['OBA'] = grading_DataFrame_Total.apply(lambda x: assign_OBA_to_CMS(x, hamamatsu_sensor_mapping), axis=1)

        # Initialize a dictionary to store the mapping of CMS sensors to OBA numbers
        cms_sensor_to_oba_mapping = {}
    
        # Loop over each OBA
        for oba_number, hamamatsu_sensors in hamamatsu_sensor_mapping.items():
            # Check if any CMS sensors are present in this Hamamatsu OBA
            cms_sensors_in_oba = grading_DataFrame_Total[(grading_DataFrame_Total['Foundry'] == 'CMS') &
                                                  (grading_DataFrame_Total['Sensor name'].isin(hamamatsu_sensors))]
            # Assign the OBA number to CMS sensors if not already assigned
            cms_sensors_in_oba_without_oba = cms_sensors_in_oba[cms_sensors_in_oba['OBA'].isna()]
            if not cms_sensors_in_oba_without_oba.empty:
                cms_sensor_to_oba_mapping.update(dict.fromkeys(cms_sensors_in_oba_without_oba['Sensor name'], oba_number))

        # Update the OBA numbers for CMS sensors based on the mapping
        grading_DataFrame_Total.loc[grading_DataFrame_Total['Sensor name'].isin(cms_sensor_to_oba_mapping.keys()), 'OBA'] = \
            grading_DataFrame_Total['Sensor name'].map(cms_sensor_to_oba_mapping)
  
        #end......
 

        createSingleIVPlot(IVdataAllDataFrame, IVGRDataFrameCMSHPK, IVTotalDataFrameCMSHPK, "Last", OBA_number, areSensorsAtCMS, grading_DataFrame_Total, amountOfSensorsAtHamamatsu, amountOfSensorsAtCMS)
        grading_DataFrame_AllOBA = grading_DataFrame_AllOBA._append(grading_DataFrame_Total)
        IVTotalDataFrameCMSHPK_AllOBA = pd.concat([IVTotalDataFrameCMSHPK_AllOBA, IVTotalDataFrameCMSHPK])
        IVTotalDataFrameCMSHPK_AllOBA['OBA_number'] = OBA_number
       # Save the data frame to a CSV file
        IVTotalDataFrameCMSHPK_AllOBA.to_csv(pathToCode + 'IVTotalDataFrameCMSHPK_AllOBA.csv', index=False)    
          #print("   !!!!!!!     IVdataAllDataFrame      !!!!!!!!!!!!!!!!!}")
          #print(IVdataAllDataFrame)
         #merged_df = pd.merge(IVdataAllDataFrame, IVdata800VDataFrame, on=['Channel', 'Sensor name'], suffixes=('_600V', '_800V'))
        # Filter both DataFrames to include only rows where 'foundry' is 'Hamamatsu'
        IVdataAllDataFrame_Hamamatsu = IVdataAllDataFrame[IVdataAllDataFrame['Foundry'] == 'Hamamatsu']
        IVdata800VDataFrame_Hamamatsu = IVdata800VDataFrame[IVdata800VDataFrame['Foundry'] == 'Hamamatsu']

        merged_df = pd.merge(IVdataAllDataFrame_Hamamatsu, IVdata800VDataFrame_Hamamatsu, on=['Channel', 'Sensor name'], suffixes=('_600V', '_800V'))
        # Calculate the ratio of currents
        merged_df['Current_Ratio_800V_600V'] = merged_df['Cell current [nA]_800V'] / merged_df['Cell current [nA]_600V']

        filtered_nonzero = IVdataAllDataFrame[
            (IVdataAllDataFrame['Cell current [nA]'] > 100) &  # Condition for Cell current
            (IVdataAllDataFrame['Foundry'] == 'Hamamatsu')     # Condition for foundry
        ]

        
        for index, row in filtered_nonzero.iterrows():
              #print(f"Channel: {row['Channel']}, Sensor name_600V: {row['Sensor name_600V']}, Sensor name_800V: {row['Sensor name_800V']}, "
                    #f"Cell current [nA]_600V: {row['Cell current [nA]_600V']}, Cell current [nA]_800V: {row['Cell current [nA]_800V']}, "
                    #f"Current Ratio: {row['Current_Ratio_800V_600V']}")

            sensor_name = row['Sensor name']
            channel = row['Channel']
            current_value = row['Cell current [nA]']
            channel_nonzero_data = {'OBA_number': [OBA_number], 'Channel': [channel], 'Sensor name': [sensor_name], 'Cell current [nA]': [current_value]}
            channel_NonZero_DataFrame = pd.concat([channel_NonZero_DataFrame, pd.DataFrame(channel_nonzero_data)], ignore_index=True)      

        ### (1)  filtered_data = IVdataAllDataFrame[(IVdataAllDataFrame['Cell current [nA]'] > 100)]
        filtered_data =  merged_df[merged_df['Cell current [nA]_600V'] > 0]
        for index, row in filtered_data.iterrows():
            sensor_name_600V = row['Sensor name']
            channel = row['Channel']
            current_value_600V = row['Cell current [nA]_600V']
            current_value_800V = row['Cell current [nA]_800V']
            current_ratios = row['Current_Ratio_800V_600V']
            channel_data = {
                'OBA_number': OBA_number,
                'Channel': channel,
                'Sensor name_600V': sensor_name_600V,
                'Cell current [nA]_600V': current_value_600V,
                'Cell current [nA]_800V': current_value_800V,
                'Currents_ratio': current_ratios
            }
            channel_DataFrame = pd.concat([channel_DataFrame, pd.DataFrame([channel_data])], ignore_index=True)
        
### end of loop over HPK OBA's  IV::

depletionVoltageCMSForDataFrame = []
sensorNamesCMSForDataFrame = []
amountOfSensorsAtCMS_CV=0
sensor_names  = []
# loop over CMS CV data (opencorrected txt files)
##### new method Vdep_serial
if os.path.exists(dataCMS_CV_Vdep):
    for directory in sorted(os.listdir(dataCMS_CV_Vdep)):
        if directory == '.DS_Store':
            continue  # Skip .DS_Store files
        amountOfSensorsAtCMS_CV+=1
        sensor_dir = os.path.join(dataCMS_CV_Vdep, directory)
        depletion_voltages = []
        if os.path.isdir(sensor_dir):
            vdep_file = os.path.join(sensor_dir, 'Vdep_serial.txt')
            if os.path.exists(vdep_file):
                print(f"Reading file: {vdep_file}")  # Debugging statement
                depletion_voltages = read_vdep_file(vdep_file)
                sensor_name = directory
                sensor_names = [sensor_name] * len(depletion_voltages)
                depletionVoltageCMSForDataFrame.extend(depletion_voltages)
                sensorNamesCMSForDataFrame.extend(sensor_names)
            else:
                print(f"File does not exist: {vdep_file}") 
            
else:
    print(f"CMS Directory does not exist: {dataCMS_CV_Vdep}")
#####


dataCVCMS = pd.DataFrame(list(zip(sensorNamesCMSForDataFrame, depletionVoltageCMSForDataFrame)), columns =['Sensor name', 'Estimated depletion voltage (V)']).assign(Foundry='CMS')
depletionVoltageHPKForDataFrame = []
depletionVoltageHPKErrors = []  # New list for errors
sensorNamesHPKForDataFrame = []
grading_DataFrame_Total_CV  = []
amountOfPlots = 0
# Create an empty DataFrame for HPK data grading
# Initialize the DataFrame to store the grading data for all OBAs
grading_DataFrame_AllOBA_CV = pd.DataFrame(columns=['Sensor name', 'condition1', 'condition2', 'condition3', 'Overall grading', 'Foundry', 'OBA'])

# loop over HPK CV data 
for root, dirs, files in os.walk(dataHPK_CV):
    if root[-2:]=="CV":
        depletionVoltageHPKForDataFrame = []
        sensorNamesHPKForDataFrame = []
        OBA_number = 'OBA' + root.split("OBA",1)[1][:5]
        CVdataAllDataFrame  = []
        amountOfSensorsAtHamamatsu=0
        print(" OBA_number  in CV loop ----------------> ", OBA_number)
        # Initialize a DataFrame for the current OBA
        grading_DataFrame_Total_CV = pd.DataFrame(columns=['Sensor name', 'condition1', 'condition2', 'condition3', 'Overall grading', 'Foundry'])

        for name in files:
            if name.endswith(("CV.txt")):
                amountOfSensorsAtHamamatsu+=1
                sensorNameCVHPK = getSensornameCVHPK(name)
                print(" sensorNameCVHPK   in CV loop ----------------> ", sensorNameCVHPK)
                pathTOCVFile = root + "/" + name
                CVDataHPK, thickness = readDataFileHPK_CV(pathTOCVFile)
                
                # Convert CVDataHPK to a DataFrame if it isn't already
                if not isinstance(CVDataHPK, pd.DataFrame):
                    CVDataHPK = pd.DataFrame(CVDataHPK)
                    
                foundry = 'Hamamatsu'
                CVDataHPK['Foundry'] = foundry    

                
                sensorNamesHPKForDataFrame.append(sensorNameCVHPK)
                   #depletionVoltageHPK = findDepletionVoltageHPK(CVDataHPK, sensorNameCVHPK, pathToResults)
                depletionVoltageHPK = findDepletionVoltageHPK(CVDataHPK.drop(columns=['Foundry']), sensorNameCVHPK, pathToResults, foundry)
                
                #print("  depletionVoltageHPK    in CV loop  ")
                #print(depletionVoltageHPK)
                # Append voltage and error separately
                depletionVoltageHPKForDataFrame.append(depletionVoltageHPK[0])  # Voltage
                depletionVoltageHPKErrors.append(depletionVoltageHPK[1])  # Error

                 #depletionVoltageHPKForDataFrame.append(depletionVoltageHPK)
  
                depletionVoltageHPK_value = depletionVoltageHPK[0]  # Extract the depletion voltage (first value)
                thicknessLimit = determineCurrentTreshold(SensorThickness)
                # Set conditions to "Passed"
                condition1 = "Passed"
                condition2 = "Passed"
                condition3 = "Passed"

                # Now compare depletionVoltageHPK_value with thicknessLimit
                if depletionVoltageHPK_value > thicknessLimit:
                    overall_grading = "Failed"
                else:
                    overall_grading = "Passed"

                # Append to grading DataFrame for the current OBA
                grading_DataFrame_Total_CV = grading_DataFrame_Total_CV.append({
                    'Sensor name': sensorNameCVHPK,
                    'condition1': condition1,
                    'condition2': condition2,
                    'condition3': condition3,
                    'Overall grading': overall_grading,
                    'Foundry': 'Hamamatsu'
                }, ignore_index=True)
        grading_DataFrame_Total_CV['OBA'] = OBA_number

        AmountOFSensorsOnOnePlot=1000

        
        if amountOfSensorsAtHamamatsu>=amountOfSensorsAtCMS_CV:
            amountOfPlots,AmountOFSensorsOnLAstPlot = divmod(amountOfSensorsAtHamamatsu,AmountOFSensorsOnOnePlot)
        else:
            amountOfPlots,AmountOFSensorsOnLAstPlot = divmod(amountOfSensorsAtCMS_CV,AmountOFSensorsOnOnePlot)
        amountOfChannels=numberOfChannels-1
        amountOfVoltagePoints=11
          #dataCVHamamatsu = pd.DataFrame(list(zip(sensorNamesHPKForDataFrame, depletionVoltageHPKForDataFrame)), columns =['Sensor name', 'Estimated depletion voltage (V)']).assign(Foundry='Hamamatsu')

        # Create the DataFrame with an additional column for 'Uncertainty (V)'
        dataCVHamamatsu = pd.DataFrame(
            list(zip(sensorNamesHPKForDataFrame, depletionVoltageHPKForDataFrame, depletionVoltageHPKErrors)), 
            columns=['Sensor name', 'Estimated depletion voltage (V)', 'Uncertainty (V)']
        ).assign(Foundry='Hamamatsu')

          # print("amountOfPlots!!!!!!!!!!!!!!!!!!!!!!!!", amountOfPlots)  = 0 !!!!!!
        for plot in range(0, amountOfPlots):
 
            firstSensorIndex=plot*AmountOFSensorsOnOnePlot
            lastSensorindex=firstSensorIndex+AmountOFSensorsOnOnePlot
            matching_sensors = list(set(dataCVHamamatsu['Sensor name']) & set(dataCVCMS['Sensor name']))
            if matching_sensors:
                CVdataAllDataFrame = pd.concat([dataCVHamamatsu[firstSensorIndex:lastSensorindex], dataCVCMS[firstSensorIndex:lastSensorindex]], ignore_index=True)
            else:
                CVdataAllDataFrame = pd.concat([dataCVHamamatsu[firstSensorIndex:lastSensorindex]])
        matching_sensors = list(set(dataCVHamamatsu['Sensor name']) & set(dataCVCMS['Sensor name']))

        if matching_sensors:
            CVdataAllDataFrame = pd.concat([dataCVHamamatsu[firstSensorIndex:lastSensorindex], dataCVCMS[firstSensorIndex:lastSensorindex]], ignore_index=True)
            data_grading_thickness_CV_CMS = determineGradingCV(data_CV_CMS, pathToCode, 'CMS')
            data_grading_thickness_CV_CMS['Foundry'] = 'CMS'
            data_grading_thickness_CV_CMS['OBA'] = OBA_number
  
            grading_DataFrame_Total_CV = pd.concat([grading_DataFrame_Total_CV, data_grading_thickness_CV_CMS])
        else:
            CVdataAllDataFrame = pd.concat([dataCVHamamatsu[firstSensorIndex:lastSensorindex]])
            grading_DataFrame_Total_CV = grading_DataFrame_Total_CV
            
        # Assigning OBA's to CMS for grading_DataFrame_Total
        hamamatsu_sensor_mapping = {}
        for oba_number, group in grading_DataFrame_Total_CV[grading_DataFrame_Total_CV['Foundry'] == 'Hamamatsu'].groupby('OBA'):
            hamamatsu_sensor_mapping[oba_number] = group['Sensor name'].tolist()

        cms_sensor_to_oba_mapping = {}
        for oba_number, hamamatsu_sensors in hamamatsu_sensor_mapping.items():
            cms_sensors_in_oba = grading_DataFrame_Total_CV[(grading_DataFrame_Total_CV['Foundry'] == 'CMS') &
                                                            (grading_DataFrame_Total_CV['Sensor name'].isin(hamamatsu_sensors))]
            cms_sensors_in_oba_without_oba = cms_sensors_in_oba[cms_sensors_in_oba['OBA'].isna()]
            if not cms_sensors_in_oba_without_oba.empty:
                cms_sensor_to_oba_mapping.update(dict.fromkeys(cms_sensors_in_oba_without_oba['Sensor name'], oba_number))

        grading_DataFrame_Total_CV.loc[grading_DataFrame_Total_CV['Sensor name'].isin(cms_sensor_to_oba_mapping.keys()), 'OBA'] = \
            grading_DataFrame_Total_CV['Sensor name'].map(cms_sensor_to_oba_mapping)

        # Append the current OBA's DataFrame to the overall DataFrame
        grading_DataFrame_AllOBA_CV = grading_DataFrame_AllOBA_CV._append(grading_DataFrame_Total_CV, ignore_index=True)

        # Call the plotting function
        singleBatch_oneOBA_forCVPlot(CVdataAllDataFrame, thickness, campaignType, OBA_number, pathToCode, findDepletionVoltageHPK, plot_and_fit, getSensorDetails, grading_DataFrame_Total_CV)


# call the function to create the PDF file

create_pdf_presentation("my_presentation.tex", "my_presentation.pdf")

