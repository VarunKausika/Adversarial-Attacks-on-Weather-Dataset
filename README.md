# Adversarial-Attacks-on-Weather-Dataset

*Dataset can be found at https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset*

This is an implementation of adversarial attacks like undirected and directed FGSM, on both a custom trained CNN as well as on pretrained AlexNet Embeddings. Project done as a part of the course SP23 - Deep Learning - Master of Science Business Analytics, McCombs School of Business.

**Undirected FGSM**:
$$||X−X̂||∞⩽ϵ$$
$$X_{adv} = X + ϵ * sign(∇_{X}J(X,Y_{true}))$$

**Directed FGSM**:
$$X_{adv}=X−ϵ*sign(∇_{X}J(X,Y_{target}))$$

The attacks were performed on 1000 images in our dataset and augmented back into our training set. The results on re-training on the attacked datasets were tested for robustness for both types of attacks again.
