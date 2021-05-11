import os
from os import listdir

path = './Test/'

count = 1

for i, file in enumerate(listdir(path)):
    old_name = path + file
    print(old_name)
    new_name = path + str(i) + ".jpg"
    os.rename(old_name, new_name)