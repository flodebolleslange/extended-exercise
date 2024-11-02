import os
import numpy as np


root = r".\lab_data"

def analyse_documents(generic_file):
    for file_name in os.listdir(root):
        if file_name[:len(generic_file)] == generic_file:
            data = np.genfromtxt(root + "\\" + file_name, delimiter=',', skip_header=2)

            frequency_list = 


# low amplitude
generic_file = r"test_"

analyse_documents(generic_file)

# high amplitude
generic_file = r"test_la_"

# long shake
generic_file = r"test_long_p"
