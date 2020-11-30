import torch
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import im_convert, load_image, get_features, gram_matrix

vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)

content = load_image('input.jpg').to(device)
style = load_image('kubizmpic.jpg', shape=content.shape[-2:]).to(device)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.axis("off")
ax2.imshow(im_convert(style))
ax2.axis("off")

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.5,
                 'conv3_1': 0.5,
                 'conv4_1': 0.5,
                 'conv4_2': 0.6}

content_weight = 1  # alpha
style_weight = 1e6  # beta

target = content.clone().requires_grad_(True).to(device)

show_every = 300
optimizer = optim.Adam([target], lr=0.005)
steps = 20000

for ii in range(1, steps + 1):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv5_1'] - content_features['conv5_1']) ** 2)
    style_loss = 0

    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        print('Iteration: ', ii)
        plt.imshow(im_convert(target))
        plt.axis("off")
        plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.axis('off')
ax2.imshow(im_convert(style))
ax2.axis('off')
ax3.imshow(im_convert(target))
ax3.axis('off')

frame_height, frame_width, _ = im_convert(target).shape

img = im_convert(target)
img = img*255
img = np.array(img, dtype = np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite('output.png', img)
