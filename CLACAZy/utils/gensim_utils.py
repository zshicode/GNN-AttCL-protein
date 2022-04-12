# parsing the vector.txt file of glove to enable it to be loaded by gensim
import gensim
import os
import shutil
import hashlib
from sys import platform

# if your vector file has different path and name, please modify the variable, INPUT_FILE
INPUT_FILE = "../data/vectors.txt"
# if you want to change the name and path of output file, please modify OUTPUT_FILE
OUTPUT_FILE = "../data/glove_cazy.txt"
# The dimension of the vectors of token
DIMENSION = 100

def getFileLineNums(filename):
    # get the lines of vector.txt
    handle = open(filename, 'r')
    count = 0
    for _ in handle:
        count += 1
    handle.close()
    return count

def prepend_line(infile, outfile, line):
    with open(infile, 'r') as in_handle:
        with open(outfile, 'w') as out_handle:
            out_handle.write(str(line) + '\n')
            shutil.copyfileobj(in_handle, out_handle)

def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as in_handle:
        with open(outfile, 'w') as out_handle:
            out_handle.write(line + '\n')
            for line in in_handle:
                out_handle.write(line)

def parse_glove(filename, output_file, dimension):
    # parse the file and output the gensim supported file
    num_lines = getFileLineNums(filename)
    gensim_file = output_file
    getsim_first_line = "{} {}".format(num_lines, dimension)
    if platform == "linux" or platform == 'linux2':
        prepend_line(filename, gensim_file, getsim_first_line)
    else:
        prepend_slow(filename, gensim_file, getsim_first_line)
    # Use gensim to load the model like the following if you want,
    # model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    # return model

if __name__ == '__main__':
    parse_glove(INPUT_FILE, OUTPUT_FILE, DIMENSION)


