import h5py


def CreateFilename(filename, mode = 'a', **kwargs):
    """
   Checks or creates a file object according to specifications given.
       Default is hdf5.

   .. warning:: Need to be implemented for other formats than hdf5

   :param filename: can be a filename or a path and filename, with or without extension.
   :param mode: is an optional string that specifies the mode in which the file is opened.
            It defaults to 'a' Append; an existing file is opened for reading and writing, and if the file does
    not exist it is created.

   :param extension: 'h5', 'json', 'shp', 'pts', etc...

   :type filename: str
   :type mode: str
   :type extension: str

   :return: a file object (according to the extension) and its extension

   """

    import re

    matched = re.match('(.*)\.([a-z].*)', filename)

    if matched is None:
        # if no extension is in filename, add
        extension = kwargs.get('extension', 'h5')  # if no extension given - default is h5

        filename = filename + '.' + extension
    else:
        # otherwise - use the extension in filename
        extension = matched.group(2)

    if extension == 'h5':
        return (h5py.File(filename, mode), extension)

    else:  # change if needed
        return (open(filename, mode), extension)
