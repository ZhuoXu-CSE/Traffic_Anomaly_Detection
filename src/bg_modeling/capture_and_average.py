import cv2
import os
import sys
import glob
import tqdm
import h5py
import numpy as np
# video_name = sys.argv[1]
# root = "/scratch/zx1412/AI-City-Anomaly-Detection/data/AIC_Track3/videos/"
# dest_dir = "data/AIC_Track3/ori_images/"
# dest_dir = "/ext3/ori_images/"

video_names = [str(i)+'.mp4' for i in range(1,101)]
# print("caputure videos")
"""
f = h5py.File('ori_images.hdf5', "a")
for video_name in tqdm.tqdm(video_names):
    file_name = video_name
    folder_name=file_name.split('.')[0]
    # folder_name = dest_dir+file_name.split('.')[0]
    # os.makedirs(folder_name,exist_ok=True)
    grp = f.create_group(folder_name)
    vc = cv2.VideoCapture(root+video_name)
    print('\n'+root+video_name)
    c = 1
    print('\n')
    print(vc.isOpened())
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    timeF =1   # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if rval == False:
            break
        # pic_path = folder_name+'/'
        if (c % timeF == 0):  # 每隔timeF帧进行存储操作
            num=str(c)
            print(folder_name + num + '.jpg')
            pic_path=num+'jpg'
            # cv2.imwrite(pic_path + str(c) + '.jpg', frame)  # 存储为图像,保存名为 >文件夹名_数字（第几个文件）.jpg
            npframe=np.array(frame)
            dset = grp.create_dataset(pic_path,data=npframe)
        c = c + 1
        # cv2.waitKey(1)

    vc.release()
f.close()
"""
# dest_dir_processed = "data/AIC_Track3/processed_images/"
f = h5py.File('ori_images.hdf5', "r")
fp = h5py.File('processed_images.hdf5', "a")
# dest_dir_processed = "/ext33/processed_images/"
print("average images")
for i in tqdm.tqdm(range(1,101)):
    # video_name = str(i)
    # path_file_number=glob.glob(os.path.join(dest_dir,video_name,'*.jpg')) #获取当前文件夹下个数
    internal_frame = 4
    start_frame = 1
    video_name = str(i)
    nums_frames = len(f[video_name])
    alpha=0.1
#    os.mkdir(dest_dir_processed+video_name,exist_ok=True)
    grp=fp.create_group(video_name)
    # os.makedirs(dest_dir_processed+video_name,exist_ok=True)
    for j in range(4,5):
        internal_frame = 4+j*4
        num_pic = int(nums_frames/internal_frame)
        # former_im = cv2.imread(dest_dir_processed+"%d/1.jpg"%i)
        ordir=video_name+'/1jpg'
        former_im = np.array(f[ordir])
        # img = cv2.imread(os.path.join(root,video_name,str(start_frame)+'.jpg'))
        img = np.array(f[ordir])
        for i in range(num_pic):
            # now_im = cv2.imread(os.path.join(root,video_name,str(i*internal_frame+start_frame)+'.jpg'))
            ndir = video_name+'/'+str(i*internal_frame+start_frame)+'jpg'
            print(ndir)
            now_im = np.array(f[ndir])
            dsdir = str(i*internal_frame+start_frame)+'_'+str(j)+'jpg'
            print(dsdir)
            if np.mean(np.abs(now_im-former_im))>5:
                img = img*(1-alpha)+now_im*alpha
                # cv2.imwrite(dest_dir_processed+video_name+'/'+str(i*internal_frame+start_frame)
                #         +'_'+str(j)+'.jpg',img)
                dset = grp.create_dataset(dsdir,data=img)
            else:
                # cv2.imwrite(dest_dir_processed+video_name+'/'+str(i*internal_frame+start_frame)
                #         +'_'+str(j)+'.jpg',img*0)
                dset = grp.create_dataset(dsdir,data=img*0)
            former_im = now_im
f.close()
fp.close()