from os import listdir
from os.path import isfile, join
import os
from os import walk

currpath = 'data'
count = 0

def listfile(path,count):
    targetpath = os.path.join(os.getcwd(),path)
    path_str = ""
    for i in range(count-1):
        path_str = path_str+"    "
    if(count > 1):
        path_str = path_str +"ᶫ---"+ path
    else:
        path_str = path_str + path

    print(path_str)
    if(os.path.isdir(targetpath)):
        for y in os.listdir(targetpath):
            listfile(os.path.join(targetpath,y),count+1)
    else:
        return None

listfile(currpath,count)
# print(os.path.isdir('C:\\Users\\User\\Desktop\\mapimage\\data\\Cap_image_cam1\\0cam1_.png'))