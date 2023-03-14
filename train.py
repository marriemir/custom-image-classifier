import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

# For Keeping an Active session
from workspace_utils import active_session

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_dir', type=str, help='path to the dataset')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture (vgg13 or vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    return parser.parse_args()

def main():
    # Parse input arguments
    args = get_input_args()

    # Check if GPU is available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load data
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    
    # Load pre-trained model
    if args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Unsupported architecture')
        
    for param in model.parameters():
        param.requires_grad = False
        
        

    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )

    # Replace the pre-trained network's classifier with the new classifier
    model.classifier = classifier

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, valid_losses = [], []


    with active_session():

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/len(train_loader))
                    valid_losses.append(valid_loss/len(valid_loader))

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                    running_loss = 0
                    model.train()
                    
    
    # Save the trained model
    # Attach the class to index mapping to the model as an attribute
    model.class_to_idx = train_dataset.class_to_idx

    # Define the checkpoint dictionary
    checkpoint = {'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs}

    # Save the checkpoint to a file
    torch.save(checkpoint, '/saved_models/checkpoint.pth')

    
if __name__ == "__main__":
    main()
