# Copyright 2026 Caleb Whitmer

import numpy as np
import math

#
# Resources:
#
# https://www.youtube.com/watch?v=OOu5KP3Gvx0
# https://genmind.ch/posts/AI-Image-Detection-More-Robust-Than-You-Think/#3-frequency-domain-fft-features


def lowPassFilter(freqDomain, radius):
  """
  Run a low pass filter on the frequency domain of an image
  
  :param      freqDomain:  The frequency domain
  :type       freqDomain:  np.ndarray
  :param      radius:      The radius
  :type       radius:      float
  
  :returns:   The filtered frequency domain
  :rtype:     np.ndarray
  """

  # Scale the radius by to smallest dimension of the image
  radius = np.clip(np.abs(radius), 0, 1) * min(freqDomain.shape)

  # Get the midpoint of the image
  mid = np.divide(freqDomain.shape, 2)

  # Iterate through each pixel of the image
  for i in range (freqDomain.shape[0]):
    for j in range (freqDomain.shape[1]):
      # If the distance of the pixel to the midpoint is greater than the radius
      # of the filter then color it black
      freqDomain[i][j] *= (math.dist(mid, [i, j]) < radius)

  # Return the frequency domain
  return freqDomain


def highPassFilter(freqDomain, radius):
  """
  Run a high pass filter on the frequency domain of an image
  
  :param      freqDomain:  The frequency domain
  :type       freqDomain:  np.ndarray
  :param      radius:      The radius
  :type       radius:      float
  
  :returns:   The filtered frequency domain
  :rtype:     np.ndarray
  """

  # Scale the radius by to smallest dimension of the image
  radius = np.clip(np.abs(radius), 0, 1) * min(freqDomain.shape)

  # Get the midpoint of the image
  mid = np.divide(freqDomain.shape, 2)

  # Iterate through each pixel of the image
  for i in range (freqDomain.shape[0]):
    for j in range (freqDomain.shape[1]):
      # If the distance of the pixel to the midpoint is less than the radius of
      # the filter then color it black
      freqDomain[i][j] *= (math.dist(mid, [i, j]) > radius)

  # Return the frequency domain
  return freqDomain

def pixelToFreqDomain(pixelDomain):
  """
  Get the frequency domain from a 2d array of gray-scale pixel data

  :param      pixelDomain:  The pixel domain (gray-scale)
  :type       pixelDomain:  np.ndarray

  :returns:   The frequency domain of the image
  :rtype:     { tbd }
  """

  # Multiply by 20 to scale; add 0.01 to prevent divide-by-zero error in
  # logarithm
  return 20 * np.log(np.fft.fftshift(np.fft.fft2(pixelDomain)) + 0.01)

def freqDomainToPixel(freqDomain):
  """
  Get the gray-scale image pixel data from a given frequency domain

  :param      freqDomain:  The frequency domain
  :type       freqDomain:  { tbd }

  :returns:   The image pixel data (gray-scale)
  :rtype:     np.ndarray
  """
  return np.float32(np.fft.ifft2(np.fft.fftshift(-0.01 + np.exp((1/20) * freqDomain))))

def binarySearchParts(parts, key, sindex=0):
  """
  Recursively search an array of partitions to find which partition a given key
  fits in
  
  :param      parts:   The array of partitions
  :type       parts:   array of pairs of keys and arrays
  :param      key:     The key to the partition
  :type       key:     float32
  :param      sindex:  The starting index
  :type       sindex:  int
  
  :returns:   The index of the partition if there is a valid partition, -1
              otherwise
  :rtype:     int
  """


  # Base cases: if the key is outside of the range then return a negative index
  if (key < 0):
    return -1
  if (key > parts[len(parts) - 1][0]):
    return -1

  # If we have found the partition which best fits the key then return the
  # starting index
  if (len(parts) == 1):
    return sindex

  # Get the midpoint of the current partition
  mid = int(len(parts) / 2) # floor of midpoint

  # Split array into left/right parts
  left = parts[0:mid]
  right = parts[mid:len(parts)]

  # If the key fits into the left part then recursively search that part
  if (key < left[-1][0]):
    return binarySearchParts(left, key, sindex)

  # Otherwise recursively search the right part
  return binarySearchParts(right, key, sindex + mid)

def partition(freqDomain, partCount):
  """
  Partition a frequency domain

  :param      freqDomain:  The frequency domain to be partitioned
  :type       freqDomain:  np.ndarray
  :param      partCount:   The number of partitions to make
  :type       partCount:   int

  :returns:   An array of partitions
  :rtype:     array of pairs of partition radii and values found in partition
  """

  # Get the minimium dimension of the frequency domain for scaling the radii
  minDim = min(freqDomain.shape)

  # Get the midpoint of the image
  mid = np.divide(freqDomain.shape, 2)

  # Get the radius of the first ring
  minRadius = (1 / math.sqrt(partCount)) / 2

  # Get the desired area of each ring
  desiredArea = math.pi * (minRadius**2)

  # The array which will hold the radius of each partition
  parts = []

  # Sqrt of part count (cached for use in loop)
  sqrtPartCount = math.sqrt(partCount)

  # Create an array of partitions
  for part in range(partCount):
    # Get the outer radius of each partition
    r1 = (math.sqrt(part + 1) / sqrtPartCount) / 2

    # Append to the array of parts
    parts.append((r1, []))

  # Go through frequency domain
  for i in range (freqDomain.shape[0]):
    for j in range (freqDomain.shape[1]):
      # Get the distance of the current pixel to the center of the image
      distance = math.dist(mid, [i, j])

      # Normalize the distance to be on a scale of 0-1 (corners of image will be
      # > 1 by they are disregarded anyways)
      distance /= minDim

      # Use the distance to find the partition to which the pixel belongs
      partIndex = binarySearchParts(parts, distance)
      # partIndex = 0

      # If the pixel is inside of a valid partition then add it to that
      # partition in the output array
      if (partIndex != -1):
        parts[partIndex][1].append(freqDomain[i][j])

  # Finally return the array of partitions
  return parts