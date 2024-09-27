
import os


class ULFile:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.dir = os.path.dirname(file_path)
        self.name_extention = os.path.basename(file_path)
        self.name = os.path.basename(file_path).split(".")[0]
        self.extension = os.path.basename(file_path).split(".")[1]


def listFilewithsuffix(dir, suffix):
    # 使用列表推导，遍历指定目录下的所有文件
    return [
        os.path.join(dir, entry.name)  # 将文件路径与文件名拼接成完整路径
        for entry in os.scandir(dir)  # 遍历指定目录下的所有文件
        if entry.name.lower().endswith(suffix.lower())  # 判断文件名是否以指定后缀结尾
    ]
