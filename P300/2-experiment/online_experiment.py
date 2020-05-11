import glob
import os
from collections import deque
from tkinter import Tk, Toplevel, IntVar, W, EW, Text, StringVar, Radiobutton, filedialog
from tkinter.ttk import Label, Button, Entry, Frame
import tkinter.font as tkFont

import numpy as np
from PIL import ImageTk, Image
from pylsl import StreamInfo, StreamOutlet, local_clock, IRREGULAR_RATE, StreamInlet, resolve_byprop

import random
import string
import time


MAX_REPETITION = 10000  # Maximum number of repetition, each repetition runs through 1-36 randomly. Window will stop afterwards

FLASH_CONCURRENT = True
CONCURRENT_ELEMENTS = 3  #if above set to True, else will not take effect

TEST_UI = False  #will not send markers


class P300Window(object):

    def __init__(self, master: Tk):
        self.master = master
        master.title('P300 speller')

        #Parameters
        self.imagesize = 125
        self.images_folder_path = '../utils/images/'  #use utils/char_generator to generate any image you want
        self.flash_image_path = '../utils/images/flash_images/einstein.jpg'
        self.number_of_rows = 6
        self.number_of_columns = 6  #make sure you have 6 x 6 amount of images in the images_folder_path
        self.flash_mode = 2  #single element  #1 for columns and rows; currently is NOT working yet; if I have time, will revisit
        self.flash_duration = 100  #soa
        self.break_duration = 125  #iti

        self.trials = 6 #number of letters
        self.delay = 2500 #interval between trial
        self.letter_idx = 0

        #did not include numbers yet!
        self.random_letter = random.choices(string.ascii_lowercase, k=self.trials)   #randomize [self.trials] number letters
        self.word = ''.join(self.random_letter)

        # Variables
        self.usable_images = []
        self.image_labels = []
        self.flash_sequence = []
        self.flash_image = None
        self.sequence_number = 0
        self.lsl_output = None
        
        self.running = 0  #for pause

        self.image_frame = Frame(self.master)
        self.image_frame.grid(row=0, column=0, rowspan=self.number_of_rows,
                              columnspan=self.number_of_columns)

        self.start_btn_text = StringVar()
        self.start_btn_text.set('Start')
        self.start_btn = Button(self.master, textvariable=self.start_btn_text, command=self.start)
        self.start_btn.grid(row=self.number_of_rows + 3, column=self.number_of_columns - 1)

        self.pause_btn = Button(self.master, text='Pause', command=self.pause)
        self.pause_btn.grid(row=self.number_of_rows + 3, column=self.number_of_columns - 4)  #-4 for center
        self.pause_btn.configure(state='disabled')

        self.close_btn = Button(self.master, text='Close', command=master.quit)
        self.close_btn.grid(row=self.number_of_rows + 3, column=0)

        fontStyle = tkFont.Font(family="Courier", size=40)
        
        self.output = Text(root, height=1, font=fontStyle)
        self.output.tag_configure("red", foreground="red")
        self.output.tag_configure("green", foreground="green")
        self.output.configure(width=10)
        self.output.insert("end", "  ")
        self.output.grid(row=self.number_of_rows + 2, column=self.number_of_columns - 4)

        self.outputlabel = Label(root, text="Output: ", font=fontStyle)
        self.outputlabel.grid(row=self.number_of_rows + 2, column=self.number_of_columns - 5)

        self.targetlabel = Label(root, text="Target: ", font=fontStyle)
        self.targetlabel.grid(row=self.number_of_rows + 1, column=self.number_of_columns - 5)

        self.show_highlight_letter(0)

        # Initialization
        self.show_images()
        self.create_flash_sequence()
        self.lsl_output = self.create_lsl_output()

    def open_images(self):
        self.usable_images = []
        self.highlight_letter_images = []

        letter_images = sorted(glob.glob(os.path.join(self.images_folder_path, 'letter_images/*.png')))

        #currently, still did not flash number yet!
        number_images = sorted(glob.glob(os.path.join(self.images_folder_path, 'number_images/*.png')))
        letter_highlight_images = sorted(glob.glob(os.path.join(self.images_folder_path, 'letter_highlight_images/*.png')))
        number_highlight_images = sorted(glob.glob(os.path.join(self.images_folder_path, 'number_highlight_images/*.png')))

        for number_image in number_images:
            letter_images.append(number_image)
        #print("Paths: ", letter_images)
        min_number_of_images = self.number_of_columns * self.number_of_rows
        if len(letter_images) < min_number_of_images:
            print('To few images in folder: ' + self.images_folder_path)
            return

        # Convert and resize images
        for image_path in letter_images:
            image = Image.open(image_path)
            resized = image.resize((self.imagesize, self.imagesize), Image.BICUBIC)
            Tkimage = ImageTk.PhotoImage(resized)
            self.usable_images.append(Tkimage)

        # Convert and resize images
        for image_path in letter_highlight_images:
            image = Image.open(image_path)
            resized = image.resize((self.imagesize, self.imagesize), Image.BICUBIC)
            Tkimage = ImageTk.PhotoImage(resized)
            self.highlight_letter_images.append(Tkimage)

        flash_img = Image.open(self.flash_image_path)
        flash_img_res = flash_img.resize((self.imagesize, self.imagesize), Image.BICUBIC)
        self.flash_image = ImageTk.PhotoImage(flash_img_res)

    def show_images(self):
        self.open_images()

        if self.usable_images == []:
            print('No images opened')
            return

        num_rows = self.number_of_rows
        num_cols = self.number_of_columns

        # Arrange images
        for r in range(0, num_rows):
            for c in range(0, num_cols):
                current_image = self.usable_images[r * num_cols + c]
                label = Label(self.image_frame, image=current_image)
                label.image = current_image
                label.grid(row=r, column=c)
                self.image_labels.append(label)

    def create_lsl_output(self):

        """Creates an LSL Stream outlet"""
        info = StreamInfo(name='LetterMarkerStream', type='LetterFlashMarkers',
                  channel_count=1, channel_format='int8', nominal_srate=IRREGULAR_RATE,
                  source_id='lettermarker_stream', handle=None)

        return StreamOutlet(info)  #for sending the predicted classes

    def create_flash_sequence(self):
        self.flash_sequence = []
        num_rows = self.number_of_rows
        num_cols = self.number_of_columns
        maximum_number = num_rows * num_cols

        flash_sequence = []

        for i in range(MAX_REPETITION):
            seq = list(range(maximum_number))  #generate 0 to maximum_number
            random.shuffle(seq)  #shuffle
            flash_sequence.extend(seq)

        self.flash_sequence = flash_sequence

    def start(self):
        if not (TEST_UI):
            self.read_lsl_marker()
        self.running = 1
        letter = self.word[0]
        image_index = string.ascii_lowercase.index(letter)
        self.highlight_image(image_index)
        self.start_btn.configure(state='disabled')
        self.pause_btn.configure(state='normal')

    def pause(self):
        self.running = 0
        self.start_btn_text.set('Resume')
        self.start_btn.configure(state='normal')
        self.pause_btn.configure(state='disabled')

    def check_pause(self):
        if self.running == 0:
            print('Flashing paused at sequence number ' + str(self.sequence_number))
            return

    def check_sequence_end(self):
        if self.sequence_number == len(self.flash_sequence):  #stop flashing if all generated sequence number runs out
            print('All elements had flashed - run out of juice')
            self.running = 0
            self.sequence_number = 0
            return

    def get_marker_result(self):
        result = self.marker_result()
        if(result):
            print("Marker received: ", result[0][0])
            receive = result[0][0]
        else:
            receive = 0 

        return receive


    def output_letter(self, receive):
        if not(receive):
            self.master.after(self.break_duration, self.start_concurrent_flashing)
        else:
            if((image_index + 1) == receive):
                self.output.insert("end", self.pos_to_char(receive), "green")
            else:
                self.output.insert("end", self.pos_to_char(receive), "red")

            self.letter_idx += 1
            if(self.letter_idx == len(self.word)):
                return
            letter = self.word[self.letter_idx]
            image_index = string.ascii_lowercase.index(letter)
            self.master.after(self.break_duration, self.highlight_target, image_index)

    def start_concurrent_flashing(self):
        
        self.check_sequence_end()
        self.check_pause()
        receive = self.get_marker_result()

        element_to_flash = self.flash_sequence[self.sequence_number:self.sequence_number+CONCURRENT_ELEMENTS]
        letter = self.word[self.letter_idx]
        image_index = string.ascii_lowercase.index(letter)

        #pushed markers to LSL stream

        print("Letter: ", image_index, " Element flash: ", [x + 1 for x in element_to_flash])
        for e in element_to_flash:
            self.lsl_output.push_sample([e + 1])  # add 1 to prevent 0 in markers

        self.flash_multiple_elements(element_to_flash)
        self.output_letter(receive)

        self.sequence_number = self.sequence_number + CONCURRENT_ELEMENTS  #change flash position

    def start_flashing(self):
        self.check_sequence_end()
        self.check_pause()
        receive = self.get_marker_result()

        element_to_flash = self.flash_sequence[self.sequence_number]
        letter = self.word[self.letter_idx]
        image_index = string.ascii_lowercase.index(letter)

        #pushed markers to LSL stream

        print("Letter: ", image_index, " Element flash: ", [element_to_flash + 1])
        self.lsl_output.push_sample([element_to_flash + 1])  # add 1 to prevent 0 in markers
        self.flash_single_element(element_to_flash)

        self.output_letter(receive)       

        self.sequence_number = self.sequence_number + 1  #change flash position

    def pos_to_char(self, pos):
        return chr(pos -1 + 97)

    def highlight_target(self, image_index):
        self.show_highlight_letter(self.letter_idx)
        self.highlight_image(image_index)

    def change_image(self, label, img):
        label.configure(image=img)
        label.image = img

    def highlight_image(self, element_no):
        self.change_image(self.image_labels[element_no], self.highlight_letter_images[element_no])
        self.master.after(self.delay, self.unhighlight_image, element_no)

    def unhighlight_image(self, element_no):
        self.change_image(self.image_labels[element_no], self.usable_images[element_no])

        if(FLASH_CONCURRENT):
            self.master.after(self.flash_duration, self.start_concurrent_flashing)
        else:
            self.master.after(self.flash_duration, self.start_flashing)

    def show_highlight_letter(self, pos):

        fontStyle = tkFont.Font(family="Courier", size=40)
        fontStyleBold = tkFont.Font(family="Courier bold", size=40)

        text = Text(root, height=1, font=fontStyle)
        text.tag_configure("bold", font=fontStyleBold)
        text.tag_configure("center", justify='center')

        for i in range(0, len(self.word)):
            if(i != pos):
                text.insert("end", self.word[i])
            else:
                text.insert("end", self.word[i], "bold")

        text.configure(state="disabled", width=10)
        text.tag_add("center", "1.0", "end")

        text.grid(row=self.number_of_rows + 1, column=self.number_of_columns - 4)

    def flash_row_or_col(self, rc_number):
        num_rows = self.number_of_rows
        num_cols = self.number_of_columns

        if rc_number < num_rows:
            for c in range(0, num_cols):  #flash row
                cur_idx = rc_number * num_cols + c
                self.change_image(self.image_labels[cur_idx], self.flash_image)
        else:
            current_column = rc_number - num_rows
            for r in range(0, num_rows):  #flash column
                cur_idx = current_column + r * num_cols
                self.change_image(self.image_labels[cur_idx], self.flash_image)

        self.master.after(self.flash_duration, self.unflash_row_or_col, rc_number)

    def unflash_row_or_col(self, rc_number):
        num_rows = self.number_of_rows
        num_cols = self.number_of_columns
        if rc_number < num_rows:
            for c in range(0, num_cols):   #flash row
                cur_idx = rc_number * num_cols + c
                self.change_image(self.image_labels[cur_idx], self.usable_images[cur_idx])
        else:
            current_column = rc_number - num_rows
            for r in range(0, num_rows):   #flash column
                cur_idx = current_column + r * num_cols
                self.change_image(self.image_labels[cur_idx], self.usable_images[cur_idx])

    def flash_multiple_elements(self, element_array):
        for element_no in element_array:
            self.change_image(self.image_labels[element_no], self.flash_image)
        
        self.master.after(self.flash_duration, self.unflash_multiple_elements, element_array)

    def unflash_multiple_elements(self, element_array):
        for element_no in element_array:
            self.change_image(self.image_labels[element_no], self.usable_images[element_no])

    def flash_single_element(self, element_no):
        self.change_image(self.image_labels[element_no], self.flash_image)
        self.master.after(self.flash_duration, self.unflash_single_element, element_no)

    def unflash_single_element(self, element_no):
        self.change_image(self.image_labels[element_no], self.usable_images[element_no])

    def marker_result(self):
        if not (TEST_UI):
            marker, timestamp = self.inlet_marker.pull_chunk()
            return marker
        else:
            return 0

    def read_lsl_marker(self):
        print("looking for a Markers stream...")
        marker_streams = resolve_byprop('name', 'ResultMarkerStream')
        if marker_streams:
            self.inlet_marker = StreamInlet(marker_streams[0])
            marker_time_correction = self.inlet_marker.time_correction()
            print("Found Markers stream")

root = Tk()
main_window = P300Window(root)
root.mainloop()