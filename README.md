# HEKADatConverter
Convert HEKA Patchmaster *.dat to ABF files that can be opened with Clampfit

## How to use:

from HEKADatConverter import convert_dat_to_ABF

input_dir = '../data/'

output_dir = input_dir

file = '240214s1c06.dat'

convert_dat_to_ABF(input_dir=input_dir, file=file, output_dir=output_dir)
