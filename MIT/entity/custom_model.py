import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from MIT.utils import accuracy



class ImageClassificationBase(nn.Module):
    ## generating predictions and calculating loss
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        ## generating predictions, calculating loss and accuracy
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'validation_step_loss': loss.detach(), 'validation_step_accuracy': acc}
        
    def validation_epoch_end(self, outputs):
        ## combining losses and accuracies
        batch_losses = [x['validation_step_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['validation_step_accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'validation_step_loss': epoch_loss.item(), 'validation_step_accuracy': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['validation_step_loss'], result['validation_step_accuracy']))

class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.network = models.resnet18(pretrained=True)
        # Replacing the last layer with our defined features
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(self.num_classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

class ResNet_152(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.network = models.resnet152(pretrained=True)
        # Replacing the last layer with our defined features
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(self.num_classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))