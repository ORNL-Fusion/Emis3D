# Constants.py
#
# File containing directory paths, etc.

from os.path import dirname, join, realpath

FILE_PATH = dirname(realpath(__file__))
EMIS3D_PARENT_DIRECTORY = dirname(FILE_PATH)
EMIS3D_TOKMAK_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "tokamaks")
EMIS3D_INPUTS_DIRECTORY = join(EMIS3D_PARENT_DIRECTORY, "inputs")
SUPPORTED_TOKAMAKS = ["DIII-D", "SPARC"]
