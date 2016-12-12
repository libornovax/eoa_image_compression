#!/usr/bin/env python
"""
Takes a CSV file (statistics.txt), which has data columns separated by space and plots the curves.
The file has the following structure:

epoch best_fitness worst_fitness mean_fitness stddev_fitness
0 5.64289e+09 1.58199e+10 8.9853e+09 1.84628e+09
1 5.64289e+09 1.58649e+10 8.93447e+09 1.84987e+09
2 5.63805e+09 1.58649e+10 8.87005e+09 1.83073e+09
3 5.63805e+09 1.58649e+10 8.77841e+09 1.81308e+09

The first line contains column names. The first column is the epoch number.
"""

__author__ = 'Libor Novak, novakli2@fel.cvut.cz'


from optparse import OptionParser
import os

import matplotlib
matplotlib.use('Agg')  # Prevents from using X interface for plotting
from matplotlib import pyplot as plt


####################################################################################################
#                                             FUNCTIONS                                            #
####################################################################################################

def draw_plot(column_names, data, path_output, title):
	"""
	Draws and saves the fitness evolution plot.

	Input:
		column_names: list of column names
		data:         list of lists of data
		path_output:  path where to save the plot (without an extension)
		title:        title of the graph
	"""
	col_name_dict = {column_names[i]: i for i in range(len(column_names))}

	for col_name in col_name_dict.keys():
		if col_name != 'epoch':
			plt.plot(data[col_name_dict['epoch']], data[col_name_dict[col_name]], label=col_name)

	plt.grid()

	# Create a legend
	plt.legend()
	plt.xlabel('epoch')
	plt.title(title)

	plt.savefig(path_output + '.pdf')


def run_plotting(path_data, path_output, title):
	"""
	Loads data from data file and draws and saves the fitness evolution plot.

	Input:
		path_data:   path to the CSV data file (columns separated by spaces)
		path_output: path where to save the plot (without an extension)
		title:       title of the graph
	"""
	print('-- Reading file: ' + path_data)
	print('-- Output file: ' + path_output + '.pdf')
	print('-- Plot title: ' + title)

	with open(path_data, 'r') as infile:
		lines = infile.readlines()

		# First line are column names
		column_names = lines[0][:-1].split(' ')
		lines[:] = lines[1:]

		print(column_names)

		# Initialize empty data columns
		DATA = [[] for i in range(len(column_names))]

		# Fill the data columns
		for line in lines:
			data = line[:-1].split(' ')
			for i in range(len(data)):
				DATA[i].append(data[i])

		draw_plot(column_names, DATA, path_output, title)

		print('-- Plotting DONE')


####################################################################################################
#                                               MAIN                                               #
####################################################################################################

def parse_options():
	""" Parse input options of the script """
	parser = OptionParser()
	parser.add_option('-f', '--path-data', action='store', dest='path_data', default='',
	                  help='Path to a CSV data file with data columns separated by space')
	parser.add_option('-t', '--title', action='store', dest='title', default='',
	                  help='Title of the plot')
	parser.add_option('-o', '--output', action='store', dest='path_output', default='',
	                  help='Path to the output file (EXCLUDING extension)')

	options, args = parser.parse_args()

	if options.path_data == '':
		print('Missing input filename!')
		parser.print_help()
		exit(1)
	if not os.path.isfile(options.path_data):
		print('Input file "%s" does not exist!'%(options.path_data))
		parser.print_help()
		exit(1)
	if options.path_output == '':
		print('Missing output filename!')
		parser.print_help()
		exit(1)

	return options


def main():
	options = parse_options()
	
	run_plotting(options.path_data, options.path_output, options.title)
	


if __name__ == '__main__':
    main()


