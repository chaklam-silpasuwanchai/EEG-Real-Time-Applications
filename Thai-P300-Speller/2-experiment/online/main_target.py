from tkinter.ttk import Style
import threading
import queue

from th_online_cb_target import root, P300Window
from lsl_record_online import eeg_record

global workQueue
workQueue = queue.Queue(1)

def experiment(workQueue):
    root.configure(background='black')
    root.style = Style()
    root.style.configure('TButton', background='black')
    root.style.configure('TButton', foreground='grey')
    main_window = P300Window(root, workQueue)
    root.mainloop()  

t = threading.Thread(target = eeg_record, args = (workQueue, True))
t.start()

experiment(workQueue)


