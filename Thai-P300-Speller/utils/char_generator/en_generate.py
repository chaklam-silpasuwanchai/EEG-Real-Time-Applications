# -*- coding: utf-8 -*- 

#------------------------------------ Imports ----------------------------------#
# Import python imaging libs
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Import operating system lib
import os

# Import random generator
from random import randint

import string
	
#------------------------------ Generate Characters ----------------------------#
def GenerateCharacters():
	# For each character do
	for char in characters:
		# For each background color do
		for background_color in background_colors:

			# Create character image : 
			# Grayscale, image size, background color
			char_image = Image.new('L', (image_size, image_size),background_color)

			# Draw character image
			draw = ImageDraw.Draw(char_image)

			# Specify font : Resource file, font size
			font = ImageFont.truetype("Arial.ttf", 512)

			# Get character width and height
			(font_width, font_height) = font.getsize(char)

			# Calculate x position
			x = (image_size - font_width)/2

			# Calculate y position
			y = (image_size - font_height)/2

			# Draw text : Position, String, 
			# Options = Fill color, Font
			draw.text((x, y), char, fill='grey' , font=font)
	
			# Final file name    	
			file_name = out_dir + char + '_grey.png'

			# Save image
			char_image.save(file_name)
	
			# Print character file name
			print(file_name)
			
	return

#---------------------------------- Output ---------------------------#

# Output
out_dir = '../images/letter_grey/'

#------------------------------------ Characters -------------------------------#

# Numbers
#numbers = ['0', '1', '2']

# Small letters
#small_letters = ['a', 'b', 'c']

# Capital letters
#capital_letters = ["A", 'B', 'C']
    	
# Select characters
#characters = numbers + small_letters + capital_letters

characters = list(string.ascii_uppercase) + ['0' , '1' , '2', '3','4', '5' ,'6' ,'7', '8' ,'9']  #generate upper case letters

import numpy as np

# characters = np.array2string(np.arange(10))  #be careful!  this will will also generate [ ] images...


	
#------------------------------------- Colors ----------------------------------#

# Background color
#white_colors = (215, 225, 235, 245)
#black_colors = (0, 10, 20, 30)
#gray_colors = (135, 145, 155)

black_bg = (0,)

background_colors =  black_bg #white_colors # + black_colors + gray_colors
    	
#-------------------------------------- Sizes ----------------------------------#

# Image size
image_size = 800

#-------------------------------------- Main -----------------------------------#

# Generate characters
GenerateCharacters()
