import torch.nn
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data import dataset
from models import ConvNet
import warnings
import os
from tqdm import tqdm
import skimage
import math

warnings.filterwarnings('ignore')

trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) # transform to make greyscale images have the same channels
composed_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150)), trans]) # sequential transform
data = dataset(root_dir='Multi-class Weather Dataset', transform=composed_transform) # loading in dataset
img, label = data.__getitem__(0)
label = torch.from_numpy(np.where(label==1)[0])

# loading in our model
model_cnn = ConvNet()
model_cnn.load_state_dict(torch.load('cnn.pth'))

#### FGSM Undirected ####
def FGSM_undirected(model_cnn, img):
    img = img.unsqueeze(0)
    img_variable = Variable(img, requires_grad=True)
    if img_variable.grad is not None: # make the gradients of the input image 0
        img_variable.grad.zero_()

    # forward pass
    output_cnn = model_cnn.forward(img_variable)
    # predict_label = torch.argmax(output_cnn).item()
    # print("Output before", output_cnn)

    # plotting our image before attack
    # plt.imshow(img.squeeze(0).permute(1, 2, 0))
    # plt.title(f'{classes[predict_label]} (before FGSM)')
    # plt.show()

    # calculating our loss and gradient of variable
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output_cnn, label)
    loss_cal.backward(retain_graph=True)

    eps = 0.02
    x_grad = torch.sign(img_variable.grad.data) # calculate the sign of gradient of the loss func (with respect to input X) (adv)

    # plotting perturbation
    # x_grad_img=x_grad.reshape(150, 150, 3)
    # plt.imshow(x_grad_img)
    # plt.show()

    x_adversarial = img_variable.data + eps * x_grad # find adv example using formula shown above
    # output_adv_cnn = model_cnn.forward(Variable(x_adversarial)) # perform a forward pass on adv example
    # predict_label_2 = torch.argmax(output_adv_cnn).item()
    # print("Output after", output_adv_cnn)

    # plotting image after attack
    # plt.imshow(x_adversarial.squeeze(0).permute(1, 2, 0))
    # plt.title(f'{classes[predict_label_2]} (after FGSM)')
    # plt.show()

    return x_adversarial

#### FGSM Directed ####
def FGSM_directed(model_cnn, img):
    img = img.unsqueeze(0)
    img_variable = Variable(img, requires_grad=True)
    if img_variable.grad is not None: # make the gradients of the input image 0
        img_variable.grad.zero_()
        
    # defining our target
    y_target = Variable(torch.LongTensor(list(np.random.choice([0, 1, 2, 3], 1))), requires_grad=False)

    # forward pass
    output_cnn = model_cnn.forward(img_variable)

    # defining loss
    loss = torch.nn.CrossEntropyLoss()
    loss_cal2 = loss(output_cnn, y_target)
    loss_cal2.backward(retain_graph=True)

    epsilon = np.random.choice([0.002, 0.01, 0.15, 0.5, 1, 1.5, 2, 5, 10], 1)[0]

    x_grad = img_variable.grad.data

    # attacking image
    x_adversarial = img_variable.data - epsilon*x_grad

    # plotting images for sequential attack
    # for i in epsilons:
    #     x_adversarial = img_variable.data - i*x_grad
    #     output_adv_cnn = model_cnn.forward(Variable(x_adversarial))
    #     predict_label_2 = torch.argmax(output_adv_cnn).item()
    #     plt.imshow(x_adversarial.squeeze(0).permute(1, 2, 0))
    #     plt.title(f'{classes[predict_label_2]} (after FGSM directed), $\epsilon={i}$')
    #     plt.show() 
    return x_adversarial

def create_attacked_training_set(model_cnn, root_dir, transform):
    # loading in dataset
    data = dataset(root_dir=root_dir, transform=transform)
    len_data = data.__len__()

    # defining classes (used when saving image)
    classes = ('Cloudy', 'Rain', 'Shine', 'Sunrise')

    train_size = math.ceil(0.7*len_data)
    test_size = len_data - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size], generator = generator1)


    # performing the attacks randomly and adding them back to the new dataset
    for i in tqdm(range(train_size)):

        img, label = train_set.__getitem__(i)
        # D=directed, U=undirected
        attack = np.random.choice(['D', 'U'])

        try:
            attacked_img = FGSM_undirected(model_cnn, img)
            # saving the image in our attacked dataset
            label_text = classes[np.where(label==1)[0][0]]
            attacked_img = attacked_img.squeeze(0).permute(1, 2, 0)
            skimage.io.imsave(f'MCWD_attacked/{label_text}/direct_attack_{i+1}.jpg', attacked_img)

        except:
            pass

        try:
            attacked_img = FGSM_directed(model_cnn, img)
            # saving the image in our attacked dataset
            label_text = classes[np.where(label==1)[0][0]]
            attacked_img = attacked_img.squeeze(0).permute(1, 2, 0)
            skimage.io.imsave(f'MCWD_attacked/{label_text}/undirect_attack_{i+1}.jpg', attacked_img)
            img = img.squeeze(0).permute(1, 2, 0)
            skimage.io.imsave(f'MCWD_attacked/{label_text}/original_{i+1}.jpg', img)

        except:
            pass
        
# create_attacked_training_set(model_cnn, 'Multi-class Weather Dataset', composed_transform, 1000)

# train model after attacking dataset
# load in trained model here
model_cnn_2 = ConvNet()
model_cnn_2.load_state_dict(torch.load('cnn_attacked.pth'))

def evaluate_model_robustness(model_cnn_1, model_cnn_2, root_dir, transform):

    # loading in dataset
    data = dataset(root_dir=root_dir, transform=transform)
    len_data = data.__len__()
    train_size = math.ceil(0.7 * len_data)
    test_size = len_data - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size], generator=generator1)




    count_correct_directed_normal=0
    count_correct_undirected_normal=0
    count_correct_directed_attacked=0
    count_correct_undirected_attacked=0

    # Turn off issue with dropout. Google it if you care just don't change it.
    model_cnn_1.eval()
    model_cnn_2.eval()
    for i in tqdm(range(test_size)):
        img, label = test_set.__getitem__(i)
        # attacking each test image from our dataset against model 1
        try:
            attacked_img_undirected = FGSM_undirected(model_cnn_1, img)
        except:
            pass
        try:
            attacked_img_directed = FGSM_directed(model_cnn_1, img)
        except:
            pass

        # getting the regular model's prediction for the attacked image - directed and undirected
        output_adv_cnn_directed = model_cnn_1(attacked_img_directed)
        predict_label_directed = torch.argmax(output_adv_cnn_directed).item()
        output_adv_cnn_undirected = model_cnn_1(attacked_img_undirected)
        predict_label_undirected = torch.argmax(output_adv_cnn_undirected).item()
        if predict_label_directed==np.where(label==1)[0][0]:
            count_correct_directed_normal+=1
        if predict_label_undirected==np.where(label==1)[0][0]:
            count_correct_undirected_normal+=1


        # attacking each test image from our dataset against model 2
        try:
            attacked_img_undirected = FGSM_undirected(model_cnn_2, img)
        except:
            pass
        try:
            attacked_img_directed = FGSM_directed(model_cnn_2, img)
        except:
            pass

        # getting the attacked model's prediction for the attacked image - directed and undirected
        output_adv_cnn_directed = model_cnn_2(attacked_img_directed)
        predict_label_directed = torch.argmax(output_adv_cnn_directed).item()
        output_adv_cnn_undirected = model_cnn_2(attacked_img_undirected)
        predict_label_undirected = torch.argmax(output_adv_cnn_undirected).item()
        if predict_label_directed==np.where(label==1)[0][0]:
            count_correct_directed_attacked+=1
        if predict_label_undirected==np.where(label==1)[0][0]:
            count_correct_undirected_attacked+=1

    print(f"""
    Regular CNN: Accuracy on FGSM (Undirected) image: {100*count_correct_undirected_normal/test_size}% \n
    Regular CNN: Accuracy on FGSM (Directed) image: {100*count_correct_directed_normal/test_size}% \n
    Attacked CNN: Accuracy on FGSM (Undirected) image: {100*count_correct_undirected_attacked/test_size}% \n
    Attacked CNN: Accuracy on FGSM (Directed) image: {100*count_correct_directed_attacked/test_size}% 
    """)

evaluate_model_robustness(model_cnn, model_cnn_2, 'Multi-class Weather Dataset', composed_transform)
