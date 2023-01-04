import numpy as np
import matplotlib

from termcolor import colored

# for using latex
matplotlib.use('Agg')

# print functions
def info(string):
    print(colored("\nINFO:: %s\n"%(string),'green'))

def warn(string):
    print(colored("\nWARNING:: %s\n"%(string),'yellow'))

def abort(string,exception=SystemExit):
    raise exception(colored("\nABORTING:: %s\n"%(string),'red'))
