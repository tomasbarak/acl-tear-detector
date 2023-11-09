from IPython.core.display import display,HTML
display(HTML('<style>.prompt{width: 0px; min-width: 0px; visibility: collapse}</style>'))

import os
import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb

from matplotlib import pyplot as plt

model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

LABELS_URL = 'https://raw.githubusercontent.com/tomasbarak/relleno/main/labels.txt'
# download the imagenet category list
classes = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

IMG_URL = "https://image.shutterstock.com/image-photo/pets-260nw-364732025.jpg"
response = requests.get(IMG_URL)
img_pil = Image.open(io.BytesIO(response.content))
img_pil.save('./images/test.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

logit = net(img_variable)
h_x = F.softmax(logit, dim=1).data.squeeze(0)
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

for i in range(0, 5):
    print('{} -> {}'.format(probs[i], classes[idx[i]]))

CAMs = returnCAM(features_blobs[0], weight_softmax, idx[:5])

img = cv2.imread('./images/test.jpg')

height, width, _ = img.shape

results = []

fig = plt.figure(figsize=(15, 6))

for i, cam in enumerate(CAMs):
    heatmap = cv2.cvtColor(cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    result = heatmap * 0.3 + img * 0.5  
    pil_img_cam = Image.fromarray(np.uint8(result))
    
    plt.subplot(2, 3, i + 1)
    
    plt.imshow(np.array(pil_img_cam))
    plt.title(classes[idx[i]])

plt.show()
