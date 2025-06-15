import torchvision


# Apply transformations
img_width, img_height = 180, 180
batch_size = 64

test_transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.495, 0.455, 0.432],
                        std=[0.299, 0.225, 0.256])
])
