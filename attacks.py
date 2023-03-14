import torch.nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data import dataset
from models import ConvNet
import warnings
warnings.filterwarnings('ignore')

classes = ('Cloudy', 'Rain', 'Shine', 'Sunrise')
trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) # transform to make greyscale images have the same channels
composed_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150)), trans]) # sequential transform
data = dataset(root_dir='Multi-class Weather Dataset', transform=composed_transform) # loading in dataset
img, label = data.__getitem__(740)
label = torch.from_numpy(np.where(label==1)[0])
img = img.unsqueeze(0)

img_variable = Variable(img, requires_grad=True)

# loading in our model
model_cnn = ConvNet()
model_cnn.load_state_dict(torch.load('cnn.pth'))

#### FGSM Undirected ####
# forward pass
output_cnn = model_cnn.forward(img_variable)
predict_label = torch.argmax(output_cnn).item()
print("Output before", output_cnn)

# plotting our image before attack
plt.imshow(img.squeeze(0).permute(1, 2, 0))
plt.title(f'{classes[predict_label]} (before FGSM)')
plt.show()

# calculating our loss and gradient of variable
loss = torch.nn.CrossEntropyLoss()
loss_cal = loss(output_cnn, label)
loss_cal.backward(retain_graph=True)

eps = 0.02
x_grad = torch.sign(img_variable.grad.data) # calculate the sign of gradient of the loss func (with respect to input X) (adv)

# plotting perturbation
x_grad_img=x_grad.reshape(150, 150, 3)
plt.imshow(x_grad_img)
plt.show()

x_adversarial = img_variable.data + eps * x_grad # find adv example using formula shown above
output_adv_cnn = model_cnn.forward(Variable(x_adversarial)) # perform a forward pass on adv example
predict_label_2 = torch.argmax(output_adv_cnn).item()
print("Output after", output_adv_cnn)

plt.imshow(x_adversarial.squeeze(0).permute(1, 2, 0))
plt.title(f'{classes[predict_label_2]} (after FGSM)')
plt.show()

#### FGSM Directed ####
if img_variable.grad is not None: # make the gradients of the input image 0
    img_variable.grad.zero_()

y_target = torch.tensor([3]) # Sunrise

loss_cal2 = loss(output_cnn, y_target)
loss_cal2.backward(retain_graph=True)

epsilons = [0.001, 0.002, 0.01, 0.15, 0.5, 1, 1.5, 2, 5, 10]

x_grad = img_variable.grad.data
for i in epsilons:
    x_adversarial = img_variable.data - i*x_grad
    output_adv_cnn = model_cnn.forward(Variable(x_adversarial))
    predict_label_2 = torch.argmax(output_adv_cnn).item()
    plt.imshow(x_adversarial.squeeze(0).permute(1, 2, 0))
    plt.title(f'{classes[predict_label_2]} (after FGSM directed), $\epsilon={i}$')
    plt.show()

