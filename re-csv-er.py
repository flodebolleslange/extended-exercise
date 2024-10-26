import numpy as np
import csv

def read_lines(file):
    f = open(file, "r")
    return f.readlines()

def re_csv(file, file_path):
    new_file = []
    lines = read_lines(file)

    for line in lines:
        line = line[:-1]
        split_line = line.split(" ")
        new_line = []
        for number in split_line:
            if number == '':
                a=0
            else:
                new_line.append(number)
        new_file.append(new_line)

    with open(file_path, "x") as file_object:
        write = csv.writer(file_object)
        write.writerows(new_file)

def analyse_documents(generic_file, generic_file_path, name_list):
    for frequency in name_list:
        frequency = str(int(frequency))
        file = generic_file
        file += frequency
        file += ".csv"
        file_path = generic_file_path
        file_path += frequency
        file_path += ".csv"

# low amplitude

generic_file = r"C:\Users\simon\Downloads\lg137\lg137\test_"
generic_file_path = r"C:\Users\simon\Downloads\lg137\re-comma-ified\test_"
test_frequencies = np.linspace(10, 40, 16)

analyse_documents(generic_file, generic_file_path, test_frequencies)

# high amplitude

generic_file = r"C:\Users\simon\Downloads\lg137\lg137\test_la_"
generic_file_path = r"C:\Users\simon\Downloads\lg137\re-comma-ified\test_la_"
test_frequencies = np.linspace(10, 40, 16)

analyse_documents(generic_file, generic_file_path, test_frequencies)

# long shake

generic_file = r"C:\Users\simon\Downloads\lg137\lg137\test_long_p"
generic_file_path = r"C:\Users\simon\Downloads\lg137\re-comma-ified\test_long_p"
test_frequencies = np.linspace(1, 3, 3)

analyse_documents(generic_file, generic_file_path, test_frequencies)