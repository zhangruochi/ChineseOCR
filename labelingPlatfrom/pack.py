import os
from pathlib import Path
import zipfile


name_lists = ["杨哥","韩锋","曲晓薇","刘占柱","徐海峰","李经理","赵爽","李涛","杨茜","刘洋","杨光","马哥","周琼","李鹏飞","高冠钰","张丽影","张健","魏东","张若驰","春霞","白鑫"]


root = Path("/home/ruochi/Documents/share/wanyi_brain/data/tyc_20200814")
zip_path = root.parent/"zipfiles"
zip_path.mkdir(exist_ok = True)

direcs = []

for d in root.glob("*"):
    if d.is_dir():
        direcs.append(d)



num_for_one = len(direcs) // len(name_lists) + 1


start = 0
for name in name_lists:
    (zip_path/name).mkdir(exist_ok=True)
    for d in direcs[start: start+num_for_one]:
        os.system("mv {} {}".format(str(d),str(zip_path/name)))

    start += num_for_one


def adddirfile(root,name): 
    f = zipfile.ZipFile(str(root/name)+".zip",'w',zipfile.ZIP_DEFLATED) 
    startdir = root/name
    for dirpath, dirnames, filenames in os.walk(startdir): 
        for filename in filenames: 
            f.write(os.path.join(dirpath,filename)) 
    f.close() 

for name in name_lists:
    adddirfile(zip_path, name)

    