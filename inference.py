import torch
from torchvision import transforms
from PIL import Image
from alexnet_model import OptimizedAlexNet

def predict_image(image_path, model_path="alexnet_optimized.pth", class_names=[]):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedAlexNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image.to(device))
        _, predicted = outputs.max(1)

    return class_names[predicted.item()]
