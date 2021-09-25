
import torchvision.transforms.functional as F
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

# function to return key for any value
def get_key( dict_in  , val):
    for key, value in dict_in.items():
         if val == value:
            return key
    return "key doesn't exist"


def matchs_eegLabel_to_imgLabel( img_dataset, eeg_labels, eeg_class_name ) :
    ## change eeg lable to match with the ImageFolder class 
    new_y = np.zeros(eeg_labels.shape)
    for loop in range(len(eeg_class_name)):
        idx = np.where( eeg_labels == loop)
        #print(idx[0])
        new_y[idx[0]]  =  img_dataset.class_to_idx[ get_key( eeg_class_name, loop ) ]


    new_y = new_y.astype(int)
    print("Num of y for class 0 ",len( new_y [ new_y==0 ] ))
    return new_y



import numpy as np
import torchvision, torch
import torchvision.transforms as transforms
class EEG_ChannelNet_ImageData():
    def __init__(self, image_path , y_eeg,  y_eeg_dic,  image_size = 229  ):
        super().__init__( )

        trsfm=transforms.Compose([
            SquarePad(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        stim_ds     = torchvision.datasets.ImageFolder(root= image_path,  transform=trsfm)
        print("Image label dic : ",stim_ds.class_to_idx)


        ## match EEG label to image label
        self.y_eeg_new = matchs_eegLabel_to_imgLabel( stim_ds, y_eeg, y_eeg_dic )

        stim_loader = torch.utils.data.DataLoader(stim_ds, batch_size = 1, shuffle=False)

        ## Use DataLoader to load image ---> numpy
        num_img_channel = 3
        self.X_images_np    = torch.zeros((120, num_img_channel, image_size, image_size))
        self.y_images_np = []

        count = 0
        for idx, (img_batch, img_batch_label) in enumerate(stim_loader):
            self.X_images_np[idx] = img_batch
            self.y_images_np.append(img_batch_label)
            count += 1
        self.y_images_np = torch.tensor(self.y_images_np)

        print("X_images_np shape : ", self.X_images_np.shape)
        print("y_images_np shape : ", self.y_images_np)

    def get_y_eeg_new(self):
        return self.y_eeg_new
 

    def get_positive_img( self ): 
        ## require image data as numpy
        print("...Get positive image")
        img_batch_labels = self.y_images_np.clone().detach()
        
        repeat = len( self.y_eeg_new ) // len( self.X_images_np  )
        print("Repeat image for ", repeat, "time")

        imag_batch = torch.from_numpy(np.tile(self.X_images_np, (repeat,1,1,1)))
        img_batch_labels = torch.from_numpy(np.tile(img_batch_labels, (repeat,)))
        
        positive_imgs = []
        for label in self.y_eeg_new:
            for idx, img in enumerate( imag_batch ):
                if label==img_batch_labels[idx] :
                    positive_imgs.append( (img,  img_batch_labels[idx].clone().detach() ) )
                    img_batch_labels[idx] = -99
                    break

        
        if self.check_positive_img( self.y_eeg_new,  positive_imgs):
            print("No error")
            return self.unpack_imgs_data( positive_imgs , 0 )
        else :
            return False


    def get_negative_img( self ):
        ## require image data as numpy
        print("...Get negative image")
        print(" len(self.y_eeg_new)", len(self.y_eeg_new) )
        img_batch_labels = self.y_images_np.clone().detach()
        
        repeat = len( self.y_eeg_new ) // len( self.X_images_np  )
        print("Repeat image for ", repeat, "time")
        # if repeat==1:
        #     print( "****len( self.y_eeg_new )  ",len( self.y_eeg_new )  )
        #     print( "****len( self.X_images_np ",len ( self.X_images_np ))

        imag_batch = torch.from_numpy(np.tile(self.X_images_np, (repeat,1,1,1)))
        img_batch_labels = torch.from_numpy(np.tile(img_batch_labels, (repeat,)))

        # shuffle the img_batch_labels using "torch.randperm()"
        img_batch_labels = img_batch_labels[torch.randperm(img_batch_labels.size()[0])]

        
        negative_imgs = []
        for label in self.y_eeg_new:
            for idx, img in enumerate( imag_batch ):
                
                if label  != img_batch_labels[idx] and  img_batch_labels[idx] != -99 :
                    negative_imgs.append( (img,  img_batch_labels[idx].clone().detach() ) )
                    img_batch_labels[idx] = -99
                    break

        del img_batch_labels
        if self.check_negative_img( self.y_eeg_new,  negative_imgs):
            print("No error")
            return True, self.unpack_imgs_data( negative_imgs , 0 )
        elif len(negative_imgs) != repeat * self.X_images_np : return False, None
        else : return False, None


    def check_positive_img( self, eeg_label_in, positive_imgs_in ):
        # check that y_img_postitive  is differ with the y_eeg
        err_count = 0
        for batch in range(len(positive_imgs_in)):
            if eeg_label_in[batch] != positive_imgs_in[batch][1]:
                print(f'Error in batch : {batch}  differ class detect {eeg_label_in[batch]}, {positive_imgs_in[batch][1]} [eeg_label , img_label] ')
                err_count +=1
        if err_count == 0 :
            status=True
        else: status=False
        return status


    def  check_negative_img( self, eeg_label_in, negative_imgs_in ):
        # y_img_negative should not in y_eeg
        err_count = 0
        for batch in range(len(negative_imgs_in)):
            if eeg_label_in[batch] == negative_imgs_in[batch][1]:
                print(f'Error in batch : {batch} same class detect {eeg_label_in[batch]}, {negative_imgs_in[batch][1]} [eeg_label , img_label] ')
                err_count += 1
        if err_count == 0 :
            status=True
        else: status=False
        return status


    def unpack_imgs_data( self, data_in:tuple , wanted_idx:int):
        
        _batch   = len(data_in)
        _channel = data_in[0][0].shape[0]
        _high    = data_in[0][0].shape[1]
        _width   = data_in[0][0].shape[2]
                    
        _result = torch.empty([ _batch,  _channel,  _high, _width  ])
        for row in range(len(data_in)):
            _result[row] = data_in[row][ wanted_idx ].reshape(1,  _channel,  _high, _width)

        return _result