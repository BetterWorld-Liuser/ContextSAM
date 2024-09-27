# 删除名称中带有_的图片


import os


def listFilewithsuffix(dir, suffix):
    return [
        os.path.join(dir, entry.name)
        for entry in os.scandir(dir)
        if entry.name.lower().endswith(suffix.lower())
    ]


for image in listFilewithsuffix("D:\\Datasets\\LoveDA\\Train\\Rural\\masks_png", "jpg"):
    file = os.path.basename(image)
    if "_" in file:
        os.remove(image)
