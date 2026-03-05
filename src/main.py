# Copyright 2026 Caleb Whitmer

import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os

import frequencydomain as fd


def getMeanEnergiesOverFileGroup(files, partCount=8, path="", suffix=""):
  """
  Gets the average energies across a group of image files.
  
  :param      files:      The file names
  :type       files:      array of strings
  :param      partCount:  The number of partitions to split each magnitude
                          spectrum into
  :type       partCount:  int
  :param      path:       The path to the file
  :type       path:       str
  :param      suffix:     The suffix of the file
  :type       suffix:     str
  
  :returns:   The average energies over file group.
  :rtype:     array of floats
  """

  # Initialize an output array
  energies_sum = [0] * partCount

  # Iterate through each file
  for file in files:
    # Open each file in gray scale format
    image = cv2.imread(f"{path}{file}{suffix}", cv2.IMREAD_GRAYSCALE)

    # Get the magnitude spectrum of the current image
    ms = np.float32(fd.pixelToFreqDomain(image))

    # Partition the current magnitude spectrum
    parts = fd.partition(ms, partCount)

    # Get the energies of each partition
    energies = [np.mean(part[1]) for part in parts]

    # Add the energies to the rolling sum
    energies_sum = np.add(energies_sum, energies)

  # Return the average of all energies by dividing the rolling sum by the count
  return np.divide(energies_sum, len(files))


def graphDifferenceOfEnergies(aEnergies, bEnergies, barWidth = 0.15, 
                              aColor="r", bColor="b", aLabel="a", 
                              bLabel="b", hLabel="", vLabel="",
                              output=""):
  """
  Make a bar graph showing the difference of two partition energy records
  
  :param      aEnergies:  The first partition energy record
  :type       aEnergies:  array of floats
  :param      bEnergies:  The second partition energy record
  :type       bEnergies:  array of floats
  :param      barWidth:   The bar width
  :type       barWidth:   float
  :param      aColor:     The first group color
  :type       aColor:     str
  :param      bColor:     The second group color
  :type       bColor:     str
  :param      aLabel:     The first group label
  :type       aLabel:     str
  :param      bLabel:     The second group label
  :type       bLabel:     str
  :param      hLabel:     The horizontal axis label
  :type       hLabel:     str
  :param      vLabel:     The vertical axis label
  :type       vLabel:     str
  :param      output:     The path of the output
  :type       output:     str
  """

  # List each set of bars from left to right in ascending order
  br1 = np.arange(len(aEnergies))
  br2 = [x + barWidth for x in br1]

  # Add each set of bars to the plot
  plt.bar(br1, aEnergies, color=aColor, width=barWidth, label=aLabel)
  plt.bar(br2, bEnergies, color=bColor, width=barWidth, label=bLabel)

  # Set labels along x axis
  plt.xticks([r + barWidth/2 for r in range(len(aEnergies))],
             [r+1 for r in range(len(aEnergies))])

  # Add horizontal and vertical labels
  plt.xlabel(hLabel)
  plt.ylabel(vLabel)

  # Create the plot
  plt.legend()

  # If an output path is specified the save the plot
  if (output != ""):
    plt.savefig(output)
  else:
    # Otherwise, show the plot
    plt.show()


def main():
  # Get the directory in which the source file resides
  currdir = os.path.dirname(__file__)

  # Get the local path to the datasets
  datadir = f"{currdir}/../datasets/CIFAKE/train"

  # The range of files to select from each section of the dataset
  range_ = 2000

  # Generate arrays of filenames
  realFiles = [f"{r:04d}" for r in range(range_)]
  fakeFiles = [f"{(r+1000):04d}" for r in range(range_)]

  # Get the mean energies of the real group
  realEnergies = getMeanEnergiesOverFileGroup(realFiles, 
                                              path=f"{datadir}/REAL/", 
                                              suffix=".jpg")

  # Get the mean energies of the fake group
  fakeEnergies = getMeanEnergiesOverFileGroup(fakeFiles, 
                                              path=f"{datadir}/FAKE/", 
                                              suffix=".jpg")

  # Graph the two energies against each other
  graphDifferenceOfEnergies(realEnergies, fakeEnergies, 
                            aLabel = "real", bLabel = "synthetic", 
                            aColor = "#7BAE7F", bColor = "#454851", 
                            hLabel="frequency partitions", vLabel="noise",
                            output=f"{currdir}/../figures/figure_1.png")

  return

if __name__ == '__main__':
  main()
