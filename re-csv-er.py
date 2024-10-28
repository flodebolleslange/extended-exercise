import numpy as np
import csv
import os

def read_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return lines

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
