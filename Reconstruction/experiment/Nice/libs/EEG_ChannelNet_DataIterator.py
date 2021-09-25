from torch.utils.data import DataLoader, Dataset
import torchvision, torch
import torchvision.transforms as transforms
import numpy as np
import random

class EEG_ChannelNet_Dataset(Dataset):
    
    def __init__(self,  eeg_data, positive_imgs, negative_imgs, eeg_label,transform=None):
        self.data   =  (eeg_data, positive_imgs, negative_imgs, eeg_label, )
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, index):
#         print(self.data[0].shape)
        _eeg_data       = self.data[0][index]
        _positive_imgs  = self.data[1][index]
        _negative_imgs  = self.data[2][index]
        _y_label        = self.data[3][index]
        return (  _eeg_data, _positive_imgs, _negative_imgs, _y_label)



class EEG_ChannelNet_DataIterator():
    def __init__(self, eeg_data, positive_images, negative_images, eeg_label, batch_size=64):
        super().__init__()
        self.eeg_data           = eeg_data
        self.positive_images    = positive_images
        self.negative_images    = negative_images
        self.eeg_label          = eeg_label
        self.train_iterator     = None
        self.val_iterator       = None
        self.test_iterator      = None
        
        self.shuffle_data()
        self.make_data_iterators( batch_size )

        
    def shuffle_data(self):
        n_sample        = self.eeg_data.shape[0]
        idx_ran         = np.array( random.sample( range(0, n_sample ) , n_sample) )
        self.eeg_data   = self.eeg_data[ idx_ran ]
        self.positive_images    = self.positive_images[ idx_ran ]
        self.negative_images    = self.negative_images[ idx_ran ]
        print("shuffle")
        print(self.eeg_label)
        self.eeg_label          = self.eeg_label[ idx_ran ]
        print(self.eeg_label)
    def get_data_iterators(self) :
        return self.train_iterator, self.val_iterator, self.test_iterator

    def make_data_iterators(self, batch_size ):
        # manully split the data
        eeg_train , eeg_val  = self.np_split( self.eeg_data , 0.7 )
        eeg_val   , eeg_test = self.np_split( eeg_val, 0.7 )

        positive_imgs_train, positive_imgs_val = self.np_split( self.positive_images, 0.7 )
        positive_imgs_val, positive_imgs_test  = self.np_split( positive_imgs_val, 0.7 )

        negative_imgs_train , negative_imgs_val = self.np_split( self.negative_images, 0.7 )
        negative_imgs_val, negative_imgs_test   = self.np_split( negative_imgs_val, 0.7 )

        y_label_train , y_label_val = self.np_split( self.eeg_label, 0.7 )
        y_label_val, y_label_test   = self.np_split( y_label_val, 0.7 )


        print("self.eeg_data.shape : ", self.eeg_data.shape )
        print("self.positive_images.shape : ", self.positive_images.shape )
        print("self.negative_images.shape : ", self.negative_images.shape )
        print("self.eeg_label.shape : ", self.eeg_label.shape )

        print("y_label_train.shape" , y_label_train.shape)
        print("y_label_val.shape" , y_label_val.shape)
        print("eeg_test.shape" , eeg_test.shape)
        print("y_label_test", y_label_test)

        # make the iterator which we use in training process

        train_dataset           = EEG_ChannelNet_Dataset( eeg_train, positive_imgs_train, negative_imgs_train, y_label_train )
        self.train_iterator     = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset             = EEG_ChannelNet_Dataset( eeg_val, positive_imgs_val, negative_imgs_val, y_label_val )
        self.val_iterator       = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        test_dataset            = EEG_ChannelNet_Dataset( eeg_test, positive_imgs_test, negative_imgs_test, y_label_test )
        self.test_iterator      = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    # input eeg 4 d [batch, channel, high, width]
    def np_split( self, data_in , ratio):
        # 0.7 
        
        _size = round(data_in.shape[0]*ratio )
        _result1 = data_in[ :_size ]
        _result2 = data_in[ _size: ]
        
        return _result1, _result2