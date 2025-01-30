"""
    visualisation.py
    Author: Milan Marocchi

    Purpose : To visualise parts of the model, for explainability.
"""
import os
import random
import logging

import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from lime import lime_image
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d
from skimage.segmentation import mark_boundaries

from classifier.transforms import (
    get_pil_transform, 
    get_pil_transform_numpy, 
    get_preprocess_transform, 
    get_normalise_transform
)

HERE = os.path.abspath(os.getcwd())

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    """
    Visualise some outputs of a specific model
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


"""
Hooks: To be added to models for explainability
"""
# Globals for the hooks
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    """
    Backwards hook to get the gradients for explainability
    """
    global gradients
    logging.debug('Backward hook running...')
    gradients = grad_output
    logging.debug(f'Gradients size: {gradients[0].size()}')


def forward_hook(module, args, output):
    """
    Forwards hook to get the activations for explainability
    """
    global activations
    logging.debug('Forward hook running...')
    activations = output
    logging.debug(f'Activations size: {activations.size()}')


def relu_hook_function(module, grad_in, grad_out):
    """
    Hook for guided backprop
    """
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)
    elif isinstance(module, BasicConv2d):
        return (torch.clamp(grad_in[0], min=0.0),)
"""
Hooks
"""

class Explainer():
    """
    Class to explain a model
    """

    def __init__(self, data_dir, model, model_code):
        self.data_dir = data_dir
        self.model_ft = model.model_ft
        self.model_code = model_code

    def get_explain_images(self):
        return None, None

    def explain(self):
        abnormal_images, normal_images = self.get_explain_images()

        if abnormal_images is not None and normal_images is not None:
            # Normal images
            logging.info(f"normal_images: {normal_images}")
            for path in normal_images:
                img = None
                img = PIL.Image.open(path).convert('RGB')

                self.lime_explain(img, 0)
                self.grad_cam_explain(img)
                self.saliency_explain(img)

            # Abnormal images
            logging.info(f"abnormal_images: {abnormal_images}")
            for path in abnormal_images:
                img = None
                img = PIL.Image.open(path).convert('RGB')

                self.lime_explain(img, 1)
                self.grad_cam_explain(img)
                self.saliency_explain(img)

    def batch_predict(self):
        """
        Predicts outputs of a batch for a specific model
        """
        def inner_batch_predict(images):
            """
            This inner function is to be used for utilities like lime that expect just one input
            NOTE: It is expected that images will be a numpy array
            """
            pre_process_transform = get_normalise_transform(size=299)

            # Pre-process data to be correct format
            images = np.transpose(images, (0, 3, 1, 2))
            images = torch.from_numpy(images).float()

            self.model_ft.eval()
            batch = torch.stack(tuple(pre_process_transform(i) for i in images), dim=0)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model_ft.to(device)
            batch = batch.to(device)

            outputs = self.model_ft(batch)
            probs = F.softmax(outputs, dim=1)

            return probs.detach().cpu().numpy()

        return inner_batch_predict


    def visualise_lime_explaination(self, image, mask):
        """
        Visualises a lime explaination
        """
        img_boundry = mark_boundaries(image, mask)
        plt.imshow(img_boundry)
        plt.tight_layout()
        plt.show()


    def lime_explain(self, img, label):
        """
        Uses lime to explain a normal and abnormal image classification
        """
        pil_transform = get_pil_transform_numpy(size=299)

        batch_predict_func = self.batch_predict()

        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(np.transpose(pil_transform(img), (1, 2, 0)),
                                                batch_predict_func,
                                                top_labels=2)
        print(label, explaination.top_labels)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=True,
                                                    num_features=5,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=False,
                                                    num_features=10,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)


    def grad_cam_explain(self, image):
        """
        Uses the gradCAM approach to exaplain a normal and abormal image classification
        """

        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Convert model to cpu for this part
        self.model_ft.to(torch.device("cpu"))
        self.model_ft.eval()

        # Apply hooks
        if self.model_code == "resnet":
            layers = [self.model_ft.layer4]
        elif self.model_code == "vgg":
            layers = [self.model_ft.features]
        elif self.model_code == "inception":
            layers = [self.model_ft.Mixed_7c]

        #for i, module in enumerate(self.model_ft.modules()):
        #    if isinstance(module, BasicConv2d):
        #        layer = module
        #        print(i)


        #layer.register_full_backward_hook(backward_hook, prepend=False)
        #layer.register_forward_hook(forward_hook, prepend=False)
        backwards_hooks = [layer.register_full_backward_hook(backward_hook, prepend=False)
                          for layer in layers]
        forwards_hooks = [layer.register_forward_hook(forward_hook, prepend=False)
                         for layer in layers]

        # Run to get gradients and activations
        out = self.model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        # Run GradCAM algorithm
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # Show figure
        plt.figure()
        pil_transform = get_pil_transform(size=299)
        image = pil_transform(image)
        plt.imshow(to_pil_image(image, mode="RGB"))

        overlay = to_pil_image(heatmap.detach(), mode='F').resize(
            ((299, 299)), resample=PIL.Image.BICUBIC
        )
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        plt.imshow(overlay, alpha=0.4, interpolation='nearest')
        plt.show()

        # Clean up
        [backwards_hook.remove() for backwards_hook in backwards_hooks]
        [forwards_hook.remove() for forwards_hook in forwards_hooks]


    def saliency_explain(self, image):
        """
        Uses the saliency map approach to explain a normal and abnormal image classification
        """
        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Setup model
        self.model_ft.eval()
        self.model_ft.to(torch.device("cpu"))

        # Setting to be guided so that it prevents backward flow of negative gradients on ReLU
        # NOTE: This is done through a hook so need to set this up
        for i, module in enumerate(self.model_ft.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)
            elif isinstance(module, BasicConv2d):
                module.register_backward_hook(relu_hook_function)

        # Run to get gradients and activations
        out = self.model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        saliency = img.grad.data

        # Display saliency map as a heatmap
        plt.figure()
        plt.imshow(np.abs(saliency.numpy()).max(axis=0), cmap=colormaps['jet'])
        plt.show()


class ModelExplainer(Explainer):

    def __init__(self, data_dir, model, model_code):
        super().__init__(data_dir, model, model_code)

    def get_explain_images(self):
        abnormal_dir = os.path.join(HERE, "image_datasets", self.data_dir, "test", "abnormal")
        normal_dir = os.path.join(HERE, "image_datasets", self.data_dir, "test", "normal")

        normal_images = [os.path.join(normal_dir, x)
                         for x in ["a0189:-1:0.png", "a0155:-1:0.png"]]
        abnormal_images = [os.path.join(abnormal_dir, x)
                           for x in ["a0005:1:0.png", "a0057:1:0.png"]]


        return abnormal_images, normal_images

class EnsembleModelExplainer(Explainer):

    def __init__(self, data_dir, model, model_code, ensemble):
        self.data_dir = data_dir
        self.models = model.model_ft.models
        self.model = model.model_ft
        self.model_code = model_code
        self.ensemble = ensemble

    def batch_predict(self, idx):
        """
        Predicts outputs of a batch for a specific model
        """
        def inner_batch_predict(images):
            """
            This inner function is to be used for utilities like lime that expect just one input
            NOTE: It is expected that images will be a numpy array
            """
            pre_process_transform = get_normalise_transform(size=299)

            # Pre-process data to be correct format
            images = np.transpose(images, (0, 3, 1, 2))
            images = torch.from_numpy(images).float()

            model_ft = self.models[idx]

            model_ft.eval()
            batch = torch.stack(tuple(pre_process_transform(i) for i in images), dim=0)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_ft.to(device)
            batch = batch.to(device)

            outputs = model_ft(batch)
            probs = F.softmax(outputs, dim=1)

            return probs.detach().cpu().numpy()

        return inner_batch_predict

    def lime_explain(self, img, label, idx):
        """
        Uses lime to explain a normal and abnormal image classification
        """
        pil_transform = get_pil_transform_numpy(size=299)

        batch_predict_func = self.batch_predict(idx)

        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(np.transpose(pil_transform(img), (1, 2, 0)),
                                                batch_predict_func,
                                                top_labels=2)
        print(label, explaination.top_labels)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=True,
                                                    num_features=5,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=False,
                                                    num_features=10,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)


    def grad_cam_explain(self, image, idx):
        """
        Uses the gradCAM approach to exaplain a normal and abormal image classification
        """

        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Convert model to cpu for this part
        model_ft = self.models[idx]
        model_ft.to(torch.device("cpu"))
        model_ft.eval()

        # Apply hooks
        if self.model_code == "resnet":
            layers = [model_ft.layer4]
        elif self.model_code == "vgg":
            layers = [model_ft.features]
        elif self.model_code == "inception":
            layers = [model_ft.Mixed_7c]

        #for i, module in enumerate(self.model_ft.modules()):
        #    if isinstance(module, BasicConv2d):
        #        layer = module
        #        print(i)


        #layer.register_full_backward_hook(backward_hook, prepend=False)
        #layer.register_forward_hook(forward_hook, prepend=False)
        backwards_hooks = [layer.register_full_backward_hook(backward_hook, prepend=False)
                          for layer in layers]
        forwards_hooks = [layer.register_forward_hook(forward_hook, prepend=False)
                         for layer in layers]

        # Run to get gradients and activations
        out = model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        # Run GradCAM algorithm
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # Show figure
        plt.figure()
        pil_transform = get_pil_transform(size=299)
        image = pil_transform(image)
        plt.imshow(to_pil_image(image, mode="RGB"))

        overlay = to_pil_image(heatmap.detach(), mode='F').resize(
            ((299, 299)), resample=PIL.Image.BICUBIC
        )
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        plt.imshow(overlay, alpha=0.4, interpolation='nearest')
        plt.show()

        # Clean up
        [backwards_hook.remove() for backwards_hook in backwards_hooks]
        [forwards_hook.remove() for forwards_hook in forwards_hooks]


    def saliency_explain(self, image, idx):
        """
        Uses the saliency map approach to explain a normal and abnormal image classification
        """
        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Setup model
        model_ft = self.models[idx]
        model_ft.eval()
        model_ft.to(torch.device("cpu"))

        # Setting to be guided so that it prevents backward flow of negative gradients on ReLU
        # NOTE: This is done through a hook so need to set this up
        for i, module in enumerate(model_ft.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)
            elif isinstance(module, BasicConv2d):
                module.register_backward_hook(relu_hook_function)

        # Run to get gradients and activations
        out = model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        saliency = img.grad.data

        # Display saliency map as a heatmap
        plt.figure()
        plt.imshow(np.abs(saliency.numpy()).max(axis=0), cmap=colormaps['jet'])
        plt.show()

    def get_weightings(self):
        """
        Finds the weightings between models in the ensemble
        """
        return self.model.classifier.weight

    def get_explain_images(self):
        abnormal_paths = [os.path.join(HERE, "image_datasets", self.data_dir, str(x), "test", "abnormal")
                          for x in range(int(self.ensemble))]
        normal_paths = [os.path.join(HERE, "image_datasets", self.data_dir, str(x), "test", "normal")
                        for x in range(int(self.ensemble))]

        abnormal_images = [[os.path.join(path, x)
                            for x in ["a0005:1:0.png", "a0057:1:0.png"]] for path in abnormal_paths]
        normal_images = [[os.path.join(path, x)
                          for x in ["a0189:-1:0.png", "a0155:-1:0.png"]] for path in normal_paths]

        return abnormal_images, normal_images

    def explain(self):
        abnormal_images, normal_images = self.get_explain_images()

        if abnormal_images is not None and normal_images is not None:
            # Normal images
            logging.info(f"normal_images: {normal_images}")
            for paths in normal_images:
                for idx, path in enumerate(paths):
                    img = None
                    img = PIL.Image.open(path).convert('RGB')

                    self.lime_explain(img, 0, idx)
                    self.grad_cam_explain(img, idx)
                    self.saliency_explain(img, idx)

            # Abnormal images
            logging.info(f"abnormal_images: {abnormal_images}")
            for paths in abnormal_images:
                for idx, path in enumerate(paths):
                    img = None
                    img = PIL.Image.open(path).convert('RGB')

                    self.lime_explain(img, 1, idx)
                    self.grad_cam_explain(img, idx)
                    self.saliency_explain(img, idx)

        # Find the weights for the final layer
        print(f"Weights of combination between models: {self.get_weightings()}")

class ExplainerFactory():

    def __init__(self, data_dir, model, model_code, ensemble):
        self.model = model
        self.data_dir = data_dir
        self.model_code = model_code
        self.ensemble = ensemble

    def create(self, explain):
        if not explain:
            return Explainer(self.data_dir, self.model, self.model_code)

        if self.ensemble is not None:
            return EnsembleModelExplainer(self.data_dir, self.model, self.model_code, self.ensemble)
        else:
            return ModelExplainer(self.data_dir, self.model, self.model_code)
