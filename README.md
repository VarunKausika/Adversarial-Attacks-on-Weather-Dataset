# Adversarial-Attacks-on-Weather-Dataset

*Dataset can be found at https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset*

This is an implementation of adversarial attacks like undirected and directed FGSM, PGD on both a custom trained CNN as well as a pretrained AlexNet. Project done as a part of the course SP23 - Deep Learning - Master of Science Business Analytics, McCombs School of Business

**Undirected FGSM**:
$$||X−X̂||∞⩽ϵ$$
$$X_{adv} = X + ϵ * sign(∇_{X}J(X,Y_{true})$$

**Directed FGSM**:
$$X_{adv}=X−ϵ*sign(∇_{X}J(X,Y_{target})$$
