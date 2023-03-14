import argparse
import torch
from PIL import Image
import numpy as np
import json

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference')
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Load the model
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Sorry, this checkpoint is not supported.")
        exit()
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
        
    # Crop the image
    left = (image.width-224)/2
    top = (image.height-224)/2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert to numpy array
    np_image = np.array(image)
    
    # Normalize the image
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the image to match PyTorch format
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert to a PyTorch tensor
    image_tensor = torch.from_numpy(np_image)
    
    return image_tensor


def predict(image_path, model, topk=5):
    
    # Preprocess the image
    image_tensor = process_image(image_path)
    
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.float()


    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass through the network
        output = model.forward(image_tensor)
        ps = torch.exp(output)
        
        # Get the top K probabilities and indices
        probs, indices = ps.topk(topk)
        probs = probs.numpy()[0]
        indices = indices.numpy()[0]

        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in indices]

        return probs, classes


def main():
    
    
    args = get_input_args()

    # Load Model from checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Use GPU if available
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)
    
    # Predict Image
    if args.top_k:
        probs, classes = predict(args.image_path, model, args.top_k)
    else:
        probs, classes = predict(args.image_path, model)
        
        
    # Map topk classes to names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    topk_names = [cat_to_name[c] for c in classes]

    # Print topk classes and probabilities
    for i in range(args.top_k):
        print("{}. {}: {:.3f}".format(i+1, topk_names[i], probs[i]))



        