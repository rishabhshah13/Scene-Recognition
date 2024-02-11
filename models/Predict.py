# import torch
import PIL as pil
import torchvision.transforms as transforms
import torch


def predict(model,image,device):
    
    # model = torch.load('/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/saved_models/run_2024-02-10_150948/resnet18_full_model_0.pt').to('mps')
    # model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485,  0.456,  0.406], std=[0.229,  0.224,  0.225]),
    ])

    # Load and preprocess the image
    # org_image = pil.Image.open('/Users/rishabhshah/Desktop/AIPI 590/Scene-Recognition/output/test/cathedral/gsun_1a8814d41a9e6565b1b947d8b79c4c39.jpg')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Perform inference
    with torch.no_grad():
        output = model(input_batch.to('mps'))

    # Get the predicted class
    _, predicted = torch.max(output,  1)
    classes = ['Campsite','Candy Store','Canyon','Castle','Cathedral']

    print(f'Predicted class: {classes[predicted.item()]}')

    return predicted.item(), classes[predicted.item()]

