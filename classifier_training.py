import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader 
from torch.backends import cudnn

import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet18, resnet50

from PIL import Image
from tqdm import tqdm
import copy
import time

import args
import utils
from networks import ImageClassifer

if not os.path.exists(args.CLASSIFIER_WEIGHT_DIR):
  os.mkdir(args.CLASSIFIER_WEIGHT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.ImageFolder(root=args.CLASSIFIER_TRAINING_DIR, transform=utils.content_img_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

NUM_CLASSES = len(train_dataset.classes)

net = ImageClassifer(NUM_CLASSES)

LR = 1e-3            # The initial Learning Rate 1e-2
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 30      # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20       # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10
FREEZE = 'conv_layers'   # Available choice: 'no_freezing', 'conv_layers', 'fc_layers'

criterion = nn.CrossEntropyLoss()

# Choose parameters to optimize and which one to freeze
if (FREEZE == 'no_freezing'):
  parameters_to_optimize = net.vgg16.parameters() # In this case we optimize over all the parameters of AlexNet
elif (FREEZE == 'conv_layers'):
  parameters_to_optimize = net.vgg16.classifier.parameters() # Updates only fully-connected layers (no conv)
elif (FREEZE == 'fc_layers'):
  parameters_to_optimize = net.vgg16.features.parameters() # Updates only conv layers (no fc)
else :
  raise (ValueError(f"Error Freezing layers (FREEZE = {FREEZE}) \n Possible values are: 'no_freezing', 'conv_layers', 'fc_layers' "))

optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
EVAL_ACCURACY_ON_TRAINING = True
criterion_val = nn.CrossEntropyLoss(reduction='sum') # for evaluation I don't want to avg over every minibatch

print("Current Hyperparameters: ")
print(f"N_EPOCHS: {NUM_EPOCHS}")
print(f"STEP_SIZE: {scheduler.step_size}")
print(f"Optimizer: \n{optimizer}")

start = time.time()

# By default, everything is loaded to cpu
net = net.to(device) # bring the network to GPU if DEVICE is cuda
cudnn.benchmark # Calling this optimizes runtime

# save best config
best_net = 0
best_epoch = 0
best_train_acc = 0.0

# save accuracy and loss
train_accuracies = []
train_losses = []

current_step = 0

# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
  print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}, LR = {scheduler.get_last_lr()}")
  
  net.train() # Sets module in training mode

  running_corrects_train = 0
  running_loss_train = 0.0

  # Iterate over the training dataset
  for images, labels in train_dataloader:

    # Bring data over the device of choice
    images = images.to(device)
    labels = labels.to(device)

    # PyTorch, by default, accumulates gradients after each backward pass
    # We need to manually set the gradients to zero before starting a new iteration
    optimizer.zero_grad() # Zero-ing the gradients
  	
    with torch.set_grad_enabled(True):

      # Forward pass to the network
      outputs_train = net(images)

      _, preds = torch.max(outputs_train, 1)

      # Compute loss based on output and ground truth
      loss = criterion(outputs_train, labels)

      # Log loss
      if current_step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(current_step, loss.item()))

      # Compute gradients for each layer and update weights
      loss.backward()  # backward pass: computes gradients
      optimizer.step() # update weights based on accumulated gradients

    current_step += 1

    # store loss and accuracy values
    running_corrects_train += torch.sum(preds == labels.data).data.item() 
    running_loss_train += loss.item() * images.size(0)
  
  train_acc = running_corrects_train / float(len(train_dataset))
  train_loss = running_loss_train / float(len(train_dataset))

  train_accuracies.append(train_acc)
  train_losses.append(train_loss) # loss computed as the average on mini-batches
  #train_loss.append(loss.item()) # loss computed only on the last batch

  ### END TRAINING PHASE OF AN EPOCH

  # Check if the current epoch val accuracy is better than the best found until now
  if (train_acc >= best_train_acc) :
    print(f"\nSave model: {best_epoch+1}\n{best_train_acc:.4f} (Training Accuracy)\n")
    print(f"> In {(time.time()-start)/60:.2f} minutes") 
    best_train_acc = train_acc
    best_epoch = epoch
    best_net = copy.deepcopy(net) # deep copy the model
    # save the model
    torch.save(best_net.state_dict(), args.CLASSIFIER_MODEL_PATH)
  
  # Step the scheduler
  scheduler.step() 

print(f"\nBest epoch: {best_epoch+1}\n{best_train_acc:.4f} (Training Accuracy)\n")
print(f"> In {(time.time()-start)/60:.2f} minutes")