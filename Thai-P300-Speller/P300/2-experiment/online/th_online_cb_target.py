import glob
import os
from tkinter import Tk, W, Text, StringVar
from tkinter.ttk import Label, Button, Frame
import tkinter.font as tkFont

import numpy as np
from PIL import ImageTk, Image
from pylsl import StreamInfo, StreamOutlet, local_clock, IRREGULAR_RATE, StreamInlet, resolve_byprop

import random
import time

MAX_FLASHES = 10000  # Maximum number of flashed images. Window will stop afterwards

class P300Window(object):
    def __init__(self, master: Tk, workQueue):
        self.master = master
        master.configure(background = 'black')
        master.title('P300 speller')
        
        self.workQueue = workQueue
        
        self.imagesize = 110
        self.images_folder_path = '../../utils/images/th_images/'  #use utils/char_generator to generate any image you want
        self.flash_image_path = '../../utils/images/flash_images/chaky-ConvertImage.jpg'
        
        self.number_of_rows = 8
        self.number_of_columns = 9  # make sure you have 8 x 9 amount of images in the images_folder_path
        
        self.lsl_streamname = 'P300_stream'
        self.flash_mode = 'cb'  # cb = Checkerboard
        
        self.flash_duration = 62  # soa
        self.break_duration = 62  # iti
        self.delay = 2500 # interval between trial

        self.trials = 6 # number of letters in the target word
        self.epochs = 5 
        
        self.flash_per_sequence = 24
        self.flash_per_target = self.flash_per_sequence * self.epochs
        self.all_poss_letters = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ',
					 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป',
					  'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ' ,
                      'ะ', 'า' ,'ิ', 'ี' , 'ึ' , 'ื' , 'ุ', 'ู', 'เ' , 'แ' , 'โ' , 'ำ' , 'ไ' , 'ใ' , 'ๆ',
                      '่' , '้' , '๊' , '๋' , '์','0','1','2','3','4','5','6','7']

        self.letter_idx = 0
        self.random_letter = random.choices(self.all_poss_letters, k=self.trials)   #randomize [self.trials] number letters
        self.word = ''.join(self.random_letter)

        self.usable_images = []
        self.image_labels = []
        self.flash_sequence = []
        self.flash_image = None
        self.sequence_number = 0
        self.lsl_output = None
        self.flash_timestamps = []
        
        self.running = 0  #for pause

        self.image_frame = Frame(self.master,style='My.TFrame')
        self.image_frame.grid(row=0, column=0, rowspan=self.number_of_rows,
                              columnspan=self.number_of_columns,sticky = W, padx=0, pady=0, ipadx=0, ipady=0)

        self.start_btn_text = StringVar()
        self.start_btn_text.set('Start')
        self.start_btn = Button(self.master, textvariable=self.start_btn_text, command=self.start)
        self.start_btn.grid(row=self.number_of_rows//2 +1, column=self.number_of_columns + 2)

        self.pause_btn = Button(self.master, text='Pause', command=self.pause)
        self.pause_btn.grid(row=self.number_of_rows//2 + 2, column=self.number_of_columns + 2)  #-4 for center
        self.pause_btn.configure(state='disabled')

        self.close_btn = Button(self.master, text='Close', command=master.quit)
        self.close_btn.grid(row=self.number_of_rows// 2 + 3, column=self.number_of_columns + 2)

        self.show_highlight_letter(0)
        
        fontStyle = tkFont.Font(family="Courier", size=40)
        self.output = Text(root, height=1, font=fontStyle)
        self.output.tag_configure("red", foreground="red")
        self.output.tag_configure("green", foreground="green")
        self.output.configure(width=10)
        self.output.insert("end", "  ")
        self.output.grid(row=self.number_of_rows + 2, column=self.number_of_columns - 4)

        self.outputlabel = Label(root, text="Output: ", font=fontStyle)
        self.outputlabel.grid(row=self.number_of_rows + 2, column=self.number_of_columns - 5)

        self.show_images()
        self.white = [i*2 for i in range(len(self.all_poss_letters)//2)]
        self.black = [(i*2)+1 for i in range(len(self.all_poss_letters)//2)]
        self.create_flash_sequence()
        self.lsl_output = self.create_lsl_output()

    def open_images(self):
        self.usable_images = []
        self.highlight_letter_images = []
        
        letter_images = sorted(glob.glob(os.path.join(self.images_folder_path, 'all_th_grey/*.png')))
        letter_highlight_images = sorted(glob.glob(os.path.join(self.images_folder_path, 'all_th_letters_high/*.png')))

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
        
        for image_path in letter_highlight_images:
            image = Image.open(image_path)
            resized = image.resize((self.imagesize, self.imagesize), Image.BICUBIC)
            Tkimage = ImageTk.PhotoImage(resized)
            self.highlight_letter_images.append(Tkimage)

        flash_img = Image.open(self.flash_image_path)
        flash_img_res = flash_img.resize((self.imagesize, self.imagesize), Image.BICUBIC)
        self.flash_image = ImageTk.PhotoImage(flash_img_res)

    def show_images(self): # show grid with letters
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
                label = Label(self.image_frame, image=current_image,background='black')
                label.image = current_image
                label.grid(row=r, column=c)
                self.image_labels.append(label)

    def create_lsl_output(self):
        """Creates an LSL Stream outlet"""
        info = StreamInfo(name=self.lsl_streamname, type='Markers',
                          channel_count=1, channel_format='int8', nominal_srate=IRREGULAR_RATE,
                          source_id='marker_stream', handle=None)
        info.desc().append_child_value('flash_mode', 'Checkerboard')
        info.desc().append_child_value('num_rows', str(self.number_of_rows))
        info.desc().append_child_value('num_cols', str(self.number_of_columns))
        return StreamOutlet(info)

    def create_flash_sequence(self):
        self.flash_sequence = []
        for t in range(self.trials): 
            for e in range(self.epochs):
                white = random.sample(self.white, len(self.white))
                black = random.sample(self.black, len(self.black))
                white = np.array(white).reshape(6,6)
                black = np.array(black).reshape(6,6)
                for i in range(white.shape[0]):
                    self.flash_sequence.append(list(white[i]))
                for i in range(white.shape[0]):    
                    self.flash_sequence.append(list(black[i]))
                for i in range(white.shape[0]):
                    self.flash_sequence.append(list(white[:,i]))
                for i in range(white.shape[0]):    
                    self.flash_sequence.append(list(black[:,i]))

    def start(self):
        self.read_lsl_marker()
        self.running = 1
        letter = self.word[0]
        target_index = self.all_poss_letters.index(letter)
        time.sleep(10)
        self.highlight_image(target_index)
        self.start_btn.configure(state='disabled')
        self.pause_btn.configure(state='normal')
        self.master.quit

    def pause(self):
        self.running = 0
        self.start_btn_text.set('Resume')
        self.start_btn.configure(state='normal')
        self.pause_btn.configure(state='disabled')

    def start_flashing(self):
        if self.sequence_number == len(self.flash_sequence):  # stop flashing if all generated sequence number runs out
            print('All elements had flashed - run out of juice')
            self.running = 0
            self.sequence_number = 0
            return
        if self.running == 0:
            print('Flashing paused at sequence number ' + str(self.sequence_number))
            return
        
        element_to_flash = self.flash_sequence[self.sequence_number] # element from the flashing sequence
        letter = self.word[self.letter_idx]
        target_index = self.all_poss_letters.index(letter) # index of the target letter

        # Push real label
        if target_index in element_to_flash: # if the target letter is being flashed
            print("Pushed to the LSL: ", "Marker: ", [2], " Target letter: ", letter, "Flash element: ", element_to_flash)
            self.lsl_output.push_sample([2], local_clock())  # marker = 2 for targets
        else:
            self.lsl_output.push_sample([1], local_clock())  # marker = 1 for non-targets
        
        self.flash_multiple_elements(element_to_flash)
        if(self.letter_idx <= len(self.word)):
            if((self.sequence_number + 1) % self.flash_per_target == 0):  # every flash_per_target, change letter 
                print(f"=========== {self.word[self.letter_idx]} ===========")
                letter = self.word[self.letter_idx]
                target_index = self.all_poss_letters.index(letter)
                
                self.lsl_output.push_sample([99], local_clock())
                time.sleep(3)
                print("SIZE", self.workQueue.qsize())
                while not self.workQueue.empty():
                    receive = self.get_predicted_result()
                    print("receive : ", receive)
                    self.output_letter(receive, target_index)
                print(f"============================== {self.letter_idx} == {len(self.word)-1}")
                if (self.letter_idx == len(self.word)-1):
                    return
                else:
                    self.letter_idx += 1
                    letter = self.word[self.letter_idx]
                    target_index = self.all_poss_letters.index(letter)
                self.master.after(self.delay, self.highlight_target, target_index)
                                
            else:
                self.master.after(self.break_duration, self.start_flashing)
        self.sequence_number = self.sequence_number + 1  #change flash position
        
    def output_letter(self, receive, target_index): # output the predicted letter
        # after we get an int as receive from the classification
        if (target_index) == receive:
            self.output.insert("end", self.pos_to_char(receive), "green")
        else:
            self.output.insert("end", self.pos_to_char(receive), "red")
    
    def get_predicted_result(self, ): # should input the classification result here
        print(f"=========== Getpredicted {self.word[self.letter_idx]} ===========")
        print("Is Queue EMPTY???: ",self.workQueue.empty())
        self.predict = self.workQueue.get()
        p300_this_target = np.array(self.flash_sequence[self.letter_idx*self.flash_per_target:(self.letter_idx*self.flash_per_target)+self.flash_per_target])
        pred_flash = p300_this_target[self.predict].reshape(-1)
        values, counts = np.unique(pred_flash, return_counts=True)
        receive = values[np.argmax(counts)]
        return receive
    
    def read_lsl_marker(self):
        print("looking for a Markers stream...")
        marker_streams = resolve_byprop('type', 'Markers')
        eeg_streams = resolve_byprop('type', 'EEG')
        if marker_streams:
            self.inlet_marker = StreamInlet(marker_streams[0])
            marker_time_correction = self.inlet_marker.time_correction()
            print("Found Markers stream")
        if eeg_streams:
            self.inlet_eeg = StreamInlet(eeg_streams[0])
            eeg_time_correction = self.inlet_eeg.time_correction()
            print("Found EEG stream")

    def highlight_target(self, target_index):
        self.show_highlight_letter(self.letter_idx)
        self.highlight_image(target_index)

    def change_image(self, label, img):
        label.configure(image=img)
        label.image = img

    def highlight_image(self, element_no):
        self.change_image(self.image_labels[element_no], self.highlight_letter_images[element_no])
        self.master.after(self.delay, self.unhighlight_image, element_no)

    def unhighlight_image(self, element_no):
        self.change_image(self.image_labels[element_no], self.usable_images[element_no])
        self.master.after(self.flash_duration, self.start_flashing)
        self.lsl_output.push_sample([88], local_clock())

    def show_highlight_letter(self, pos):
        fontStyle = tkFont.Font(family="THSarabunNew", size=50)        
        fontStyleBig = tkFont.Font(family="THSarabunNew", size=80)
        fontStyleBold = tkFont.Font(family="THSarabunNew", size=50 )
        text = Text(root, height=2.0, font=fontStyle, background='black', foreground='grey')
        text.tag_configure("bold", font=fontStyleBold, foreground = 'white')
        text.tag_configure("center", justify='center')
        text.tag_configure("big", foreground = 'black',font=fontStyleBig)
        text.insert("end",'|',"big")
        for i in range(0, len(self.word)):
            if(i != pos):
                text.insert("end", self.word[i])
            else:
                text.insert("end", self.word[i],"bold")
        text.insert("end",'|',"big")
        text.configure(state="disabled", width=10)
        text.tag_add("center", "1.0", "end")
        text.grid(row=self.number_of_rows//2, column=self.number_of_columns + 2)
        
    def flash_multiple_elements(self, element_array):
        self.flash_timestamps.append(local_clock())
        if (self.sequence_number % self.flash_per_target == 118) :
            for element_no in (element_array):
                self.change_image(self.image_labels[element_no], self.usable_images[element_no])
        else:
            for element_no in (element_array):
                self.change_image(self.image_labels[element_no], self.flash_image)
        self.master.after(self.flash_duration, self.unflash_multiple_elements, element_array)

    def unflash_multiple_elements(self, element_array):
        for element_no in element_array:
            self.change_image(self.image_labels[element_no], self.usable_images[element_no])

    def pos_to_char(self, pos):
        return self.all_poss_letters[pos]

root = Tk()