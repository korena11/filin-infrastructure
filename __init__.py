import os
import sys

module_path = os.path.dirname(os.path.abspath(__file__))
list_directories = [name for name in os.listdir(module_path) if os.path.isdir(os.path.join(module_path, name))]

for directory in list_directories:
    if '.' in directory or 'venv' in directory or 'docs' in directory or '__' in directory:
        continue

    else:
        directory_path = module_path + '/' + directory
    if directory_path not in sys.path:
        sys.path.append(directory_path)

print('Filin infrastructure fully loaded')
