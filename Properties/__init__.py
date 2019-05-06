import os
import sys

module_path = os.path.dirname(os.path.abspath(__file__))
files_list = [f for f in os.listdir(module_path)]
__all__ = []
packages = []
for file in files_list:
    if '__' in file or 'test' in file:
        continue
    elif '.py' in file:
        __all__.append(file[:-3])
    else:
        sys.path.append(module_path + '/' + file)
