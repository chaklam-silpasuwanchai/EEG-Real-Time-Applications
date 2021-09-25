import torchvision.models as models
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
from IPython.display import clear_output

from .EEGModels.EEG_ChannelNet import EEG_ChannelNet, channelNetLoss
from .utilities import get_freer_gpu, squeeze_tensor_to_list

class Inception_Encoder(nn.Module):
    def __init__(self, num_class):
        super(Inception_Encoder, self).__init__()
        self.inceptionv3 = models.inception_v3(pretrained=True, aux_logits=False)
        self.classi_fc = nn.Linear(1000, num_class)

    def forward(self, x):
        x = self.inceptionv3(x)
        return x
    
    def forward_classify(self, x):
        x = self.forward(x)
        x = self.classi_fc(x)
#         x = F.softmax(x, dim=1)
        return x


class EEG_ChannelNet_Train():
    
    def __init__(self,  num_class, learning_rate=0.001, latent_size=1000):

        from libs.utilities import get_freer_gpu
        self.device         = torch.device( get_freer_gpu()) if torch.cuda.is_available() else torch.device("cpu")
        self.is_debug      = False
        self.n_class       = num_class
        self.learning_rate = learning_rate

        self.eeg_encoder   = EEG_ChannelNet(self.n_class).to(self.device )
        self.image_encoder = Inception_Encoder(self.n_class).to(self.device )
        self.encoder_optimizer    = torch.optim.Adam([
                                                {'params': self.eeg_encoder.parameters()},
                                                {'params': self.image_encoder.parameters()}
                                                 ], lr=self.learning_rate)
        self.criterion  = None
    
        self.encoder_train_losses = []
        self.encoder_val_losses = []
        
        self.eeg_train_losses     = []
        self.eeg_train_accuracies = []
        self.img_train_losses     = []
        self.img_train_accuracies = []
        
        self.eeg_val_losses     = []
        self.eeg_val_accuracies = []
        self.img_val_losses     = []
        self.img_val_accuracies = []
        
        self.eeg_best_val_loss    = float('inf')
        self.eeg_best_epoch       = 0
        
        # best value (only for EEG's classier model)
        self.best_model      = None
        self.best_train_acc  = None
        self.best_val_acc    = None
        self.test_acc        = None

    
    def train_encoder_epoch(self, train_iterator ):
        epoch_loss = 0
        for eeg, img_pos, img_neg, y_label in train_iterator:
            self.eeg_encoder.train()
            self.image_encoder.train()

            # Move tensors to the configured device
            eeg     = eeg.to(self.device).float()
            img_pos = img_pos.to(self.device).float()
            img_neg = img_neg.to(self.device).float()

            # Forward pass
            latent_eeg     = self.eeg_encoder(eeg)
            latent_img_pos = self.image_encoder(img_pos)
            latent_img_neg = self.image_encoder(img_neg)
            loss           = channelNetLoss(latent_eeg, latent_img_pos, latent_img_neg).sum()
            

            # Backward and optimize
            self.encoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            
            epoch_loss   += loss.item()
            
        self.encoder_train_losses.append(epoch_loss/len(train_iterator))

        pass
            
    def evaluate_encoder_epoch( self, val_iterator ):
        epoch_loss = 0
        with torch.no_grad():
            for eeg, img_pos, img_neg, y_label in val_iterator:
                self.eeg_encoder.eval()
                self.image_encoder.eval()

                # Move tensors to the configured device
                eeg     = eeg.to(self.device).float()
                img_pos = img_pos.to(self.device).float()
                img_neg = img_neg.to(self.device).float()

                # Forward pass
                latent_eeg     = self.eeg_encoder(eeg)
                latent_img_pos = self.image_encoder(img_pos)
                latent_img_neg = self.image_encoder(img_neg)

                loss    = channelNetLoss(latent_eeg, latent_img_pos, latent_img_neg).sum()
                epoch_loss   += loss.item()
            self.encoder_val_losses.append( epoch_loss/len(val_iterator) )
                
        pass

    def train_encoder(self, train_iterator, val_iterator, n_epochs= 5 ):
   
        for epoch in range(n_epochs):
            self.train_encoder_epoch(    train_iterator )
            self.evaluate_encoder_epoch( val_iterator   )
            

            if self.is_debug : display_every = 1
            else: display_every = 20

            if (epoch+1) % display_every == 0:             
                # if self.is_debug : 
                #     data_plot = { "train_loss" : self.encoder_train_losses ,  "val_loss" : self.encoder_val_losses }
                #     clear_output(wait=True)
                #     self.do_plot( data_plot , title="EEG-Image loss")
                print(f'Epoch: {epoch+1:02}/{n_epochs}  |' , end = '')
                print(f'\tTrain Loss: {self.encoder_train_losses[-1]:.5f} |' , end = '')
                print(f'\tVal Loss: {self.encoder_val_losses[-1]:.5f} ' )
                #print("-------------------------------------")
        pass
    
    def train_classifier_epoch(self, train_iterator , eeg_optimizer, img_optimizer ):
            
        # BEFORE training the classifer fc layer we must freeze the encoder part of the model first!!
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False
        self.eeg_encoder.classi_fc.weight.requires_grad = True

        for param in self.image_encoder.parameters():
            param.requires_grad = False 
        self.image_encoder.classi_fc.weight.requires_grad = True
                        
        eeg_total = 0
        eeg_correct = 0
        eeg_epoch_loss = 0
        eeg_epoch_acc = 0
        eeg_predicted_list = []

        img_total = 0
        img_correct = 0
        img_epoch_loss = 0
        img_epoch_acc = 0
        img_predicted_list = []

        for eeg, pos_img, _, y_label in train_iterator:

            self.eeg_encoder.train()
            self.image_encoder.train()

            # Move tensors to the configured device
            eeg     = eeg.to(self.device).float()
            pos_img = pos_img.to(self.device).float()
            y_label = y_label.to(self.device).long()

            # Forward pass
            pred_eeg     = self.eeg_encoder.forward_classify(eeg).to(self.device)
            pred_pos_img = self.image_encoder.forward_classify(pos_img).to(self.device)

            eeg_loss    = self.criterion(pred_eeg, y_label).to(self.device)
            img_loss    = self.criterion(pred_pos_img, y_label).to(self.device)

            # Backward and optimize
            eeg_optimizer.zero_grad()
            eeg_loss.backward()
            eeg_optimizer.step()

            img_optimizer.zero_grad()
            img_loss.backward()
            img_optimizer.step()

            self.eeg_encoder.eval()
            self.image_encoder.eval()

            # check accuracy
            eeg_predictions  = self.eeg_encoder.forward_classify(eeg).to(self.device)        
            _, eeg_predicted = torch.max(eeg_predictions.data, 1)  #returns max value, indices
            eeg_total        += y_label.size(0)  #keep track of total

            eeg_correct      += (eeg_predicted == y_label).sum().item()  #.item() give the raw number
            eeg_acc          = 100 * (eeg_correct / eeg_total)

            eeg_epoch_loss   += eeg_loss.item()
            eeg_epoch_acc    = eeg_acc
            
            
            

            img_predictions  = self.image_encoder.forward_classify(pos_img).to(self.device)        
            _, img_predicted = torch.max(img_predictions.data, 1)  #returns max value, indices
            img_total        += y_label.size(0)  #keep track of total

            img_correct      += (img_predicted == y_label).sum().item()  #.item() give the raw number
            img_acc          = 100 * (img_correct / img_total)

            img_epoch_loss   += img_loss.item()
            img_epoch_acc    = img_acc
            
  
            return_data_dic = {     "eeg_loss"  : eeg_epoch_loss / len(train_iterator), 
                                    "eeg_acc"   : eeg_epoch_acc,
                                    "eeg_predicted": eeg_predicted, 
                                    "y_label"   :    y_label, 
                                    "img_loss"  : img_epoch_loss / len(train_iterator) ,
                                    "img_acc"   : img_epoch_acc,
                                    "img_predicted": img_predicted
                            }
        # self.eeg_train_accuracies.append(     eeg_epoch_acc    )
        # self.img_train_accuracies.append(     img_epoch_acc    )
        # self.eeg_train_losses.append(         eeg_epoch_loss / len(train_iterator)    )    
        # self.img_train_losses.append(    img_epoch_loss / len(train_iterator)         )  

        return return_data_dic

    
    
    def evaluate_classifier_epoch(self, val_iterator , eeg_optimizer, img_optimizer, test_mode=False ):
            
        eeg_total = 0
        eeg_correct = 0
        eeg_epoch_loss = 0
        eeg_epoch_acc = 0
        eeg_predicted_list = []

        img_total = 0
        img_correct = 0
        img_epoch_loss = 0
        img_epoch_acc = 0
        img_predicted_list = []
        
        if test_mode:
            self.eeg_encoder =  copy.deepcopy( self.best_model )
            print("...Load best model weight to the eeg_encoder for test")
        
        self.eeg_encoder.eval()
        self.image_encoder.eval()
        with torch.no_grad():
            for eeg, pos_img, _, y_label in val_iterator:
                # Move tensors to the configured device
                eeg     = eeg.to(self.device).float()
                pos_img = pos_img.to(self.device).float()
                y_label = y_label.to(self.device).long()

                # Forward pass
                pred_eeg     = self.eeg_encoder.forward_classify(eeg).to(self.device)
                pred_pos_img = self.image_encoder.forward_classify(pos_img).to(self.device)

                eeg_loss    = self.criterion(pred_eeg, y_label).to(self.device)
                img_loss    = self.criterion(pred_pos_img, y_label).to(self.device)


                self.eeg_encoder.eval()
                self.image_encoder.eval()

                # check accuracy
                eeg_predictions  = self.eeg_encoder.forward_classify(eeg).to(self.device)        
                _, eeg_predicted = torch.max(eeg_predictions.data, 1)  #returns max value, indices
                eeg_total        += y_label.size(0)  #keep track of total

                eeg_correct      += (eeg_predicted == y_label).sum().item()  #.item() give the raw number
                eeg_acc          = 100 * (eeg_correct / eeg_total)

                eeg_epoch_loss   += eeg_loss.item()
                eeg_epoch_acc    = eeg_acc
                
                


                img_predictions  = self.image_encoder.forward_classify(pos_img).to(self.device)        
                _, img_predicted = torch.max(img_predictions.data, 1)  #returns max value, indices
                img_total        += y_label.size(0)  #keep track of total

                img_correct      += (img_predicted == y_label).sum().item()  #.item() give the raw number
                img_acc          = 100 * (img_correct / img_total)

                img_epoch_loss   += img_loss.item()
                img_epoch_acc    = img_acc


            return_data_dic = {     "eeg_loss"  : eeg_epoch_loss / len(val_iterator), 
                                    "eeg_acc"   : eeg_epoch_acc,
                                    "eeg_predicted": eeg_predicted, 
                                    "y_label"   :    y_label, 
                                    "img_loss"  : img_epoch_loss / len(val_iterator) ,
                                    "img_acc"   : img_epoch_acc,
                                    "img_predicted": img_predicted
                            }

        return return_data_dic
    
    def train_classifier(self,  train_iterator, val_iterator, n_epochs= 5 , 
                                    criterion = nn.CrossEntropyLoss() 
                             ):
        self.criterion = criterion.to(self.device)
        
        iteration     = 0
        eeg_best_model    = None
        img_best_model    = None
        
        eeg_classify_optimizer = torch.optim.Adam(self.eeg_encoder .parameters(), lr=self.learning_rate )
        img_classify_optimizer = torch.optim.Adam(self.image_encoder .parameters(), lr=self.learning_rate )
        

        for epoch in range(n_epochs):

            train_result_dic = self.train_classifier_epoch(     train_iterator , eeg_classify_optimizer, img_classify_optimizer   )
            eval_result_dic  = self.evaluate_classifier_epoch(  val_iterator , eeg_classify_optimizer, img_classify_optimizer     )


            self.eeg_train_losses.append(       train_result_dic["eeg_loss"]    )
            self.eeg_train_accuracies.append(   train_result_dic["eeg_acc" ]    ) 
            self.img_train_losses.append(       train_result_dic["img_loss"]    )
            self.img_train_accuracies.append(   train_result_dic["img_acc" ]    )

            self.eeg_val_losses.append(         eval_result_dic["eeg_loss"]     )
            self.eeg_val_accuracies.append(     eval_result_dic["eeg_acc" ]     )   
            self.img_val_losses.append(         eval_result_dic["img_loss"]     )
            self.img_val_accuracies.append(     eval_result_dic["img_acc" ]     )


            ## display progress              
            if self.is_debug : display_every = 1 
            else:  display_every = 20

            if (epoch+1) % display_every == 0:
                # if self.is_debug : 
                #     clear_output(wait=True)
                #     self.do_plot( { "train_loss" : self.eeg_train_losses ,  "val_loss" : self.eeg_val_losses} , title= "EEG classify loss" )
                #     self.do_plot( {"train_acc" : self.eeg_train_accuracies ,  "val_acc" : self.eeg_val_accuracies}, title= "EEG classify acc" )
                
                print(f'EEG Epoch: {epoch+1:02}/{n_epochs}  |',end='')
                print(f'\tTrain Loss: {self.eeg_train_losses[-1]:.5f}  | Train Acc: {self.eeg_train_accuracies[-1]:.2f}%  |', end='')
                print(f'\t Val. Loss: {self.eeg_val_losses[-1]:.5f}  | Val. Acc: {self.eeg_val_accuracies[-1]:.2f}% |' , end='')
                print(f"Best epoch is {self.eeg_best_epoch}")
                    
                    
                # if self.is_debug : 
                #     self.do_plot( { "train_loss" : self.img_train_losses ,  "val_loss" : self.img_val_losses },  title="Image classify loss"  )
                #     self.do_plot( {"train_acc" : self.img_train_accuracies ,  "val_acc" : self.img_val_accuracies},  title="Image classify acc"  )  
          
                print(f'IMG Epoch: {epoch+1:02}/{n_epochs}  |',end='')
                print(f'\tTrain Loss: {self.img_train_losses[-1]:.5f}  | Train Acc: {self.img_train_accuracies[-1]:.2f}%  |', end='')
                print(f'\t Val. Loss: {self.img_val_losses[-1]:.5f}  | Val. Acc: {self.img_val_accuracies[-1]:.2f}%')
                #print("-------------------------------------")
            ## keep the best model by validate loss
            if self.eeg_val_losses[-1] < self.eeg_best_val_loss:
                self.eeg_best_val_loss    = self.eeg_val_losses[-1]
                self.eeg_best_epoch       = epoch  # remember best epoch
                self.best_model           = copy.deepcopy( self.eeg_encoder )
                self.best_train_acc       = self.eeg_train_accuracies[-1]
                self.best_val_acc         = self.eeg_val_accuracies[-1]


    def save_state_dict(self, path):
        if self.best_model != None:
            print("Model:{} saved.".format(type(self.best_model).__name__))
            torch.save(self.best_model.state_dict(), f'{path}_{type(self.best_model).__name__}_dict.pt.tar')
        else:
            print("There is no model to save")


    
    def do_testing(self, test_iterator_in):
        eeg_classify_optimizer = torch.optim.Adam(self.eeg_encoder .parameters(), lr=self.learning_rate )
        img_classify_optimizer = torch.optim.Adam(self.image_encoder .parameters(), lr=self.learning_rate )

        eval_result_dic  = self.evaluate_classifier_epoch(  test_iterator_in , eeg_classify_optimizer, img_classify_optimizer     )
        print(f'Test Loss: {eval_result_dic["eeg_loss"]:.3f} | Test Acc: {eval_result_dic["eeg_acc"]:.2f}%')

        y_test     = squeeze_tensor_to_list(  eval_result_dic["y_label"]  )
        print(f"The number of test {len(y_test)}")

        y_hat_test = squeeze_tensor_to_list(eval_result_dic["eeg_predicted"])
        out = zip(y_test, y_hat_test)
        print("(Actual y , Predicted y)")
        print(list(out))

        self.test_acc   = eval_result_dic["eeg_acc"]

    
    def get_result(self):
        result = [ type(self.best_model).__name__ ,  self.best_train_acc, self.best_val_acc, self.test_acc , self.eeg_best_epoch ]
        return result

    
    def do_plot(self,  data_dic, title="" ):
        plt.figure(figsize=(15,3))
        
        for _key, _val in data_dic.items():
  
            plt.plot( _val , label=_key)
     
        plt.title(title)
        plt.legend()
        plt.show()

    def save_plot(self, save_path, data_dic, title="" ):
        import matplotlib as mpl
        mpl.use('agg')
        fig = plt.figure(figsize=(15,3))
        
        for _key, _val in data_dic.items():
  
            plt.plot( _val , label=_key)
     
        plt.title(title)
        plt.legend()
        plt.savefig(f'{save_path}-{title}.png')
        plt.close(fig)