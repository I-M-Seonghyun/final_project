from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

img = read_image("/content/KakaoTalk_20231115_094454205.jpg")
# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
# Step 2: Initialize the inference transforms
preprocess = weights.transforms()
# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)
# Forward pass up to the last convolutional layer (conv5)
features = model.conv1(batch)
features = model.bn1(features)
features = model.relu(features)
features = model.maxpool(features)

features = model.layer1(features)
features = model.layer2(features)
features = model.layer3(features)
features = model.layer4(features)
# 'features' now contains the output of the last convolutional layer (conv5)