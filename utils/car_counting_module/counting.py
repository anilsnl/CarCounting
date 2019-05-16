
import csv
import pandas
import cv2
import datetime
import os

is_vehicle_detected = [0]
current_frame_number_list = [0]
bottom_position_of_detected_vehicle = [0]


def predict_count(
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position_top,
    roi_position_bottom,
    roi_right,
    roi_left,
    error_foctor_bt
    ):
    speed = 'n.a.'  # means not available, it is just initialization
    direction = 'n.a.'  # means not available, it is just initialization
    scale_constant = 1  # manual scaling because we did not performed camera calibration
    isInROI = True  # is the object that is inside Region Of Interest
    update_csv = False
  
    
    csv_line=[]
    f = open('abc.csv','r')
    reader = csv.reader(f)
    for row in reader:
        csv_line.append(row)
    f.close()
    
    if(len(csv_line)==0): #if a cae pass onece in the video
        if(bottom>roi_position_top and bottom<roi_position_bottom):
          is_vehicle_detected.insert(0,1)
    else:
        last_car_obj=csv_line[len(csv_line)-1]
        last_car_left = float(last_car_obj[1])
        last_car_right = float(last_car_obj[2])
        last_car_bottom = float(last_car_obj[3])
        last_car_top = float(last_car_obj[4])
        #debuging
        #print(last_car_bottom)
        if(last_car_bottom>bottom and (last_car_bottom-bottom)>error_foctor_bt):
            is_vehicle_detected.insert(0,1)
            print('Frame:'+str(current_frame_number)+'R: '+str(right)+' L: '+str(left)+' T: '+str(top))
            CURRENT_DT = datetime.datetime.now()
            savePath = 'detected_vehicles/'+str(CURRENT_DT.year)+'-'+str(CURRENT_DT.month)+'-'+str(CURRENT_DT.day)+'/'+str(CURRENT_DT.hour)+'_'+str(CURRENT_DT.minute)+'_'+str(CURRENT_DT.second)+'.png'
            cv2.imwrite(savePath,crop_img)

    #file operation
    f = open('abc.csv','a')
    #left,right,bottom,top
    f.write(str(current_frame_number)+','+str(left)+','+str(right)+','+str(bottom)+','+str(top) +'\n')
    f.close()
     
    return is_vehicle_detected
