#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 data_filename"
  exit 1
fi

datafile="$1"
outputfile="${datafile%.*}.png"

gnuplot -e "set terminal png; set output '${outputfile}'; plot '${datafile}' using 2:3 with linespoints pt 7 ps 1.5 title '${datafile}'"
echo "Plot saved to ${outputfile}"
