import numpy as np
import os
import json
import glob
import sys
import cv2
from utils.utils import *
from src.reid import Reid_Extrctor
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
# sklearn.metrics.pairwise.cosine_similarity() 
import h5py


fp = h5py.File('ori_images.hdf5', "r")            
#video id
a=sys.argv[1]
i=int(a)
video_name = str(i)
reid_model_name = "resnet50"
reid_model_path = "/scratch/zx1412/AI-City-Anomaly-Detection/models/reid.pth"

#Read static detection results and video information、
root  =  "data/AIC_Track3/ori_images"
# path_file_number = glob.glob(os.path.join(root,video_name,'*.jpg'))
# nums_frames  =  len(path_file_number)
nums_frames  =  len(fp[video_name])
frame_rate = 30
# print('video'+str(i)+'.json')
with open('/scratch/zx1412/AI-City-Anomaly-Detection/detection_results/test_static/video'+str(i)+'.json','r') as info:
        imgs = json.load(info)

# im = cv2.imread(path_file_number[0])
strr=video_name+'/1jpg'
im = np.array(fp[strr])
h,w,c = im.shape
#segmentation map
ignore_mat = np.load("/scratch/zx1412/AI-City-Anomaly-Detection/detection_results/seg_masks/%s.npy"%a)
#Initialization information matrices
count_matrix = np.zeros((h,w))            
no_detect_count_matrix = np.zeros((h,w))  
start_time_matrix = np.zeros((h,w))
end_time_matrix = np.zeros((h,w))
score_matrix = np.zeros((h,w))
state_matrix = np.zeros((h,w))      #State matrix, 0/1 distinguishes suspicious candidate states
start_frame = 1
# nums_frames = len(path_file_number)

#Thresholds
len_time_thre = 60   #Minimum abnormal duration (seconds)
no_detect_thred = 8  #The shortest continuous number of times undetected, used to judge the abnormal termination or cancel the suspicious state
detect_thred = 5     #The shortest continuous number of times detected
score_thred = 0.3    #Threshold of detection score
light_thred = 0.8
anomely_score_thred = 0.7
similarity_thred = 0.95
suspiciou_time_thre = 18

#Read frame by frame detection results
dt_results_fbf = {}
with open("/scratch/zx1412/AI-City-Anomaly-Detection/detection_results/test_framebyframe/video%s.txt"%(a),'r') as f:
    for line in f:
        line = line.rstrip()
        word = line.split(',')
        frame = int(word[0])
        x1 = int(word[2])
        y1 = int(word[3])
        tmp_w = int(word[4])
        tmp_h = int(word[5])
        score = float(word[6])
        if frame not in dt_results_fbf:
            dt_results_fbf[frame]=[]
        if score > score_thred :
            dt_results_fbf[frame].append([x1,y1,x1+tmp_w,y1+tmp_h,score])


# Initialize the reid model
reid_model = Reid_Extrctor(reid_model_name,reid_model_path)

j=4
internal_frame = 4+j*4 #The interval of average image
num_pic = int(nums_frames/internal_frame)
start=0
tmp_start =0

all_results=[]
anomely_tmp =[]
anomely_now ={}
for i in range(1,num_pic):
    name  =  str(i*internal_frame+start_frame)+'.jpg'
    tmp_detect = np.zeros((h,w))
    tmp_score = np.zeros((h,w))
    if name in imgs:
        # print(imgs[name][str(j)])
        # print(count_matrix[148, 366],state_matrix[148, 366],no_detect_count_matrix[148, 366])
        tmp  =  imgs[name][str(j)]
        num_boxes = 0
        max_score  =  0
        for box in tmp:
            score = float(box[1])
            box  =  box[0]
            if score >0.3:
                tmp_score[int(float(box[1])):int(float(box[3])),int(float(box[0])):int(float(box[2]))] = np.maximum(score,tmp_score[int(float(box[1])):int(float(box[3])),int(float(box[0])):int(float(box[2]))])
                tmp_detect[int(float(box[1])):int(float(box[3])),int(float(box[0])):int(float(box[2]))] = 1
        tmp_score = tmp_score*ignore_mat
        tmp_detect = tmp_detect*ignore_mat
    count_matrix = count_matrix + tmp_detect
    score_matrix = score_matrix + tmp_score
    tmp_no_detect = 1 - tmp_detect 

    #Update detection matrices
    no_detect_count_matrix += tmp_no_detect
    no_detect_count_matrix[tmp_detect>0] = 0
    #Update time matrices
    if i==1:
        start_time_matrix[count_matrix==1]=-600
    else:
        start_time_matrix[count_matrix==1]=i*internal_frame+start_frame
    end_time_matrix[count_matrix>0] =i*internal_frame+start_frame
    #Update state matrices
    state_matrix[count_matrix>detect_thred]=1
  
    #Detect anomaly
    time_delay = end_time_matrix - start_time_matrix
    time_delay = time_delay *state_matrix
    index = np.unravel_index(time_delay.argmax(), time_delay.shape)
    # print(index,(i*internal_frame+start_frame)/frame_rate)
    # print(count_matrix[index],state_matrix[index],no_detect_count_matrix[index])
    if np.max(time_delay)/frame_rate>len_time_thre and start == 0: #and score_matrix[index]/count_matrix[index]>0.8:
        
        index = np.unravel_index(time_delay.argmax(), time_delay.shape)
        #backtrack the start time
        time_frame = int(start_time_matrix[index]/5)*5+1
        G = count_matrix.copy()
        G[G<count_matrix[index]-2]=0
        G[G>0]=1

        region = search_region(G,index)

        #vehicle reid
        if 'start_time' in anomely_now and (time_frame/frame_rate - anomely_now['end_time'])<30:
            maxf = str(int(max(1,anomely_now['start_time']*frame_rate)))
            dirPath = video_name+'/'+maxf+'jpg'
            # print(dirPath)
            # feature1 = reid_model.extract("data/AIC_Track3/ori_images/%s/%d.jpg"%(a,max(1,anomely_now['start_time']*frame_rate)),anomely_now['region'])
            feature1 = reid_model.extract(dirPath,anomely_now['region'])
            maxf1 = str(int(max(1,time_frame)))
            dirPath1 = video_name+'/'+maxf1+'jpg'
            # feature2 = reid_model.extract("data/AIC_Track3/ori_images/%s/%d.jpg"%(a,max(1,time_frame)),region)
            feature2 = reid_model.extract(dirPath1,region)
            similarity = cosine_similarity(feature1,feature2)
            # print(similarity)
            if similarity > similarity_thred:
                time_frame = int(anomely_now['start_time']*frame_rate/5)*5+1
            else:
                anomely_now['region'] = region
        else: 
            anomely_now['region'] = region

                
        # print(region)
        max_iou = 1
        count = 1
        start_time=time_frame
        tmp_len =1
        while (max_iou>0.1 or tmp_len<40 or raio>0.6) and time_frame>1 :
            raio = float(count)/float(tmp_len)
            if time_frame in dt_results_fbf:
                max_iou = compute_iou(anomely_now['region'],np.array(dt_results_fbf[time_frame]))
            else:
                max_iou = 0 
            time_frame -=5
            if max_iou>0.3:
                count+=1
                if max_iou>0.5:
                    start_time = time_frame
            tmp_len+=1
        time_frame = start_time
        sst = str(int(time_frame))
        tfdir = video_name+'/'+sst+'jpg'
        # print(tfdir)
        # print(fp[tfdir])
        tmp_im = np.array(fp[tfdir])
        # tmp_im = cv2.imread("data/AIC_Track3/ori_images/%s/%d.jpg"%(a,time_frame))
        while  time_frame>1 and compute_brightness(tmp_im[region[1]:region[3],region[0]:region[2],:])>light_thred :
            start_time = time_frame
            time_frame -= 5
            sst1 = str(time_frame)
            tfdir1 = video_name+'/'+sst1+'jpg'
            # print(tfdir1)
            tmp_im = np.array(fp[tfdir1])
            # tmp_im = cv2.imread("data/AIC_Track3/ori_images/%s/%d.jpg"%(a,time_frame))

        anomely_now['start_time'] = max(0,start_time/frame_rate)
        anomely_now['end_time'] = max(0,end_time_matrix[index]/frame_rate)
        start = 1
    elif np.max(time_delay)/frame_rate>suspiciou_time_thre and tmp_start == 0:
        # print(anomely_now)
        time_frame = int(start_time_matrix[index])
        G = count_matrix.copy()
        G[G<count_matrix[index]-2]=0
        G[G>0]=1
        region = search_region(G,index)

        #vehicle reid
        if 'start_time' in anomely_now and (time_frame/frame_rate - anomely_now['end_time'])<30:
            maxf2 = str(int(max(1,anomely_now['start_time']*frame_rate)))
            dirPath2 = video_name+'/'+maxf2+'jpg'
            feature1 = reid_model.extract(dirPath2,anomely_now['region'])
            # feature1 = reid_model.extract("data/AIC_Track3/ori_images/%s/%d.jpg"%(a,max(1,anomely_now['start_time']*frame_rate)),anomely_now['region'])
            maxf3 = str(int(max(1,time_frame)))
            dirPath3 = video_name+'/'+maxf3+'jpg'
            feature2 = reid_model.extract(dirPath3,region)
            # feature2 = reid_model.extract("data/AIC_Track3/ori_images/%s/%d.jpg"%(a,max(1,time_frame)),region)
            similarity = cosine_similarity(feature1,feature2)
            # print(similarity)
            
            if similarity > similarity_thred:
                # print(similarity)
                time_frame = int(anomely_now['start_time']*frame_rate/5)*5+1
                region = anomely_now['region']
                
        anomely_now['region'] = region
        anomely_now['start_time'] = max(0,time_frame/frame_rate)
        anomely_now['end_time'] = max(0,end_time_matrix[index]/frame_rate)
        # print(anomely_now)

        tmp_start = 1


    if np.max(time_delay)/frame_rate>len_time_thre and start == 1:

        index = np.unravel_index(time_delay.argmax(), time_delay.shape)
        if no_detect_count_matrix[index]>no_detect_thred:
            # print(anomely_now)
            anomely_score = score_matrix[index]/count_matrix[index]
            # print(anomely_score)
            if anomely_score > anomely_score_thred:
                anomely_now['end_time'] = end_time_matrix[index]/frame_rate
                anomely_now['score'] = anomely_score
                # print(anomely_score)
                all_results.append(anomely_now)
                anomely_now = {}
                # print("end time: "+str(end_time_matrix[index]/frame_rate)+" score: "+str(score_matrix[index]/count_matrix[index]))
            start = 0
    elif np.max(time_delay)/frame_rate>suspiciou_time_thre and tmp_start == 1:
        if no_detect_count_matrix[index]>no_detect_thred:

            anomely_score = score_matrix[index]/count_matrix[index]
            if anomely_score > anomely_score_thred:
                anomely_now['end_time'] = end_time_matrix[index]/frame_rate
                anomely_now['score'] = anomely_score
            tmp_start = 0

    #no_detect matrix change state_matrix
    state_matrix[no_detect_count_matrix>no_detect_thred] = 0
    no_detect_count_matrix[no_detect_count_matrix>no_detect_thred] = 0
    #update matrix
    tmp_detect = tmp_detect+state_matrix
    tmp_detect[tmp_detect>1] =1
    count_matrix = count_matrix * tmp_detect
    score_matrix = score_matrix * tmp_detect

if np.max(time_delay)/frame_rate>len_time_thre and start == 1:
    # print("end time: "+str(end_time_matrix[index]/frame_rate)+" score: "+str(score_matrix[index]/count_matrix[index]))
    anomely_score = score_matrix[index]/count_matrix[index]
    if anomely_score > anomely_score_thred:
        anomely_now['end_time'] = end_time_matrix[index]/frame_rate
        anomely_now['score'] = anomely_score
        # print(anomely_score)
        all_results.append(anomely_now)
        anomely_now = {}
        start = 0
if all_results:
    nms_out = anomely_nms(all_results)
    # print(nms_out)
    final_result={'start_time':892,'score':0}
    for i in range(nms_out.shape[0]):
        if nms_out[i,5]<final_result['start_time']:
            final_result['start_time'] = max(0,int(nms_out[i,5]-1))
            final_result['score'] = 1.0 #nms_out[i,4]

    print("%s %d %.1f"%(a,final_result['start_time'],final_result['score']))
fp.close()
