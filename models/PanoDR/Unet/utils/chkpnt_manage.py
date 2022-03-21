import glob
import time
import os

def getCheckpoint(folder_path):
    
    files_D = glob.glob(folder_path.replace("*.pth", "*D*.pth"))
    files_G = glob.glob(folder_path.replace("*.pth", "*G*.pth"))
    file_times_D = list(map(lambda x: time.ctime(os.path.getctime(x)), files_D))
    file_times_G = list(map(lambda x: time.ctime(os.path.getctime(x)), files_G))
    files_D[sorted(range(len(file_times_D)), key=lambda x: file_times_D[x])[-1]]
    files_G[sorted(range(len(file_times_G)), key=lambda x: file_times_G[x])[-1]]

    return [files_D[0], files_G[0]]