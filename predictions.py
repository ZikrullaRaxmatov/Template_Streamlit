import torch
import torch.nn.functional as F
from ml_models import CNN, CNN_Melanoma
from torchvision import transforms


def prediction_result(uploaded_img, best_model_path):

    # Apply transformations
    img_width, img_height = 180, 180

    test_transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.495, 0.455, 0.432],
                            std=[0.299, 0.225, 0.256])
    ])
    
    # Use transformation
    if uploaded_img.mode != 'RGB':
        uploaded_img = uploaded_img.convert('RGB')
            
    input_tensor = test_transform(uploaded_img)

    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the device
    device = torch.device("cpu")
    input_batch = input_batch.to(device)

    # Load the trained model
    best_model = CNN_Melanoma()
    best_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    best_model.to(device)

    best_model.eval()  # Set the model to evaluation mode

    # Get the model's output
    with torch.no_grad():
        output = best_model(input_batch)

    # Interpret the output
    _, predicted_class = torch.max(output, 1)
    class_index = predicted_class.item()
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get confidence
    confidence = torch.max(probabilities, dim=1).values

    #print(f"Predicted class: {class_index}, Confidence: {confidence.item()*100:.2f}%")
    
    return class_index, confidence



def prediction_result_binary(uploaded_img):

    # Apply transformations
    img_width, img_height = 180, 180

    test_transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.495, 0.455, 0.432],
                            std=[0.299, 0.225, 0.256])
    ])
    
    # Use transformation
    if uploaded_img.mode != 'RGB':
        uploaded_img = uploaded_img.convert('RGB')
            
    input_tensor = test_transform(uploaded_img)

    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the device
    device = torch.device("cpu")
    input_batch = input_batch.to(device)

    # Load the trained model
    best_model = CNN_Melanoma()
    best_model.load_state_dict(torch.load('./best_model_brain_cancer.pt', map_location=torch.device('cpu')))
    best_model.to(device)

    best_model.eval()  # Set the model to evaluation mode

    # Get the model's output
    with torch.no_grad():
        output = best_model(input_batch)

    # Interpret the output
    _, predicted_class = torch.max(output, 1)
    class_index = predicted_class.item()
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get confidence
    confidence = torch.max(probabilities, dim=1).values

    #print(f"Predicted class: {class_index}, Confidence: {confidence.item()*100:.2f}%")
    
    return class_index, confidence
