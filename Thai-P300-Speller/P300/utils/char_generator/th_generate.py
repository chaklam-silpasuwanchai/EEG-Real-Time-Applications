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
def GenerateCharacters(characters):
	# For each character do
	for idx,char in enumerate(characters):
		# For each background color do
		for background_color in background_colors:

			# Create character image : 
			# Grayscale, image size, background color
			char_image = Image.new('L', (image_size, image_size),background_color)

			# Draw character image
			draw = ImageDraw.Draw(char_image)

			# Specify font : Resource file, font size
			font = ImageFont.truetype("/home/chanapa/BCI_fork/BCI/P300/utils/char_generator/THSarabunNew/THSarabunNew.ttf", 600) # 800 for alphabet, 600 for sound

			# Get character width and height
			(font_width, font_height) = font.getsize(char)

			# Calculate x position
			x = (image_size - font_width)/2

			# Calculate y position
			y = 50 # -115 for alpha,  50 for sounds

			# Draw text : Position, String, 
			# Options = Fill color, Font
			draw.text((x, y), char, fill='white', font=font)
	
			# Final file name    	
			file_name = out_dir + 'sound' + str(idx) + 'black.png'

			# Save image
			char_image.save(file_name)
	
			# Print character file name
			print(file_name)
			
	return

#---------------------------------- Output ---------------------------#

# Output
out_dir = '/home/chanapa/BCI_fork/BCI/P300/utils/th_images/all_th_letters/'

#------------------------------------ Characters -------------------------------#

# Numbers
#numbers = ['0', '1', '2']

# Small letters
#small_letters = ['a', 'b', 'c']

# Capital letters
#capital_letters = ["A", 'B', 'C']

# Thai letters
characters = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ',
					 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป',
					  'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ' ,
                      'ะ', 'า' ,'–ิ', '–ี' ,'–ึ', '–ื', '–ุ', '–ู', 'เ' , 'แ' , 'โ' , '–ำ' , 'ไ' , 'ใ' , 'ๆ']

sound = [ '–่', '–้', '–๊', '–๋', '–์']
    	
# Select characters
#characters = numbers + small_letters + capital_letters

#characters = list(string.ascii_uppercase)  #generate upper case letters

import numpy as np
	
#------------------------------------- Colors ----------------------------------#
# Background color
white_bg = (215, 225, 235, 245)
#black_colors = (0, 10, 20, 30)
#gray_colors = (135, 145, 155)

black_bg = (0,)

background_colors =  black_bg #white_colors # + black_colors + gray_colors
    	
#-------------------------------------- Sizes ----------------------------------#
# Image size
image_size = 800

#-------------------------------------- Main -----------------------------------#

# Generate characters
GenerateCharacters(sound)
