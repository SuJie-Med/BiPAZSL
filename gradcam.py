import torch
import torch.nn.functional as F
import numpy as np
import argparse
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model.modeling import build_gzsl_pipeline
from model.config import cfg
from data import build_dataloader


def get_image_info(image_path, new_img_files, label, attribute):
    """
    Retrieve class ID and attribute information of an image based on its path.
    
    Args:
        image_path (str): Path to the input image.
        new_img_files (np.ndarray): Array containing all image paths in the dataset.
        label (np.ndarray): Array of class labels corresponding to all images.
        attribute (np.ndarray): Array of class attributes.
    
    Returns:
        image_index (int): Index of the image in the dataset.
        label_id (int): Class ID of the image.
        image_attr (np.ndarray): Attribute vector corresponding to the image's class.
    """
    # Find the index of the image path in the dataset
    image_index = np.where(new_img_files == image_path)[0]
    
    if len(image_index) == 0:
        raise ValueError(f"Image path '{image_path}' not found in dataset.")
    
    image_index = image_index[0]  # Get the first matching index
    label_id = label[image_index]  # Get the class ID of the image
    image_attr = attribute[label_id]  # Get the attribute vector for the class
    
    return image_index, label_id, image_attr


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM module.
        
        Args:
            model: Trained neural network model.
            target_layer: Target layer in the model to generate CAM from.
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None  # Stores feature maps from the target layer
        self.gradients = None     # Stores gradients from the target layer
        
        # Register hooks to capture feature maps and gradients
        self.hook = self.target_layer.register_forward_hook(self.save_feature_maps)
        self.hook_backward = self.target_layer.register_full_backward_hook(self.save_gradients)

    
    def save_feature_maps(self, module, input, output):
        """Hook function to save feature maps from the target layer during forward pass."""
        self.feature_maps = output
        print(f"Feature maps shape: {self.feature_maps.shape}")  # Print shape for debugging
    
    def save_gradients(self, module, grad_input, grad_output):
        """Hook function to save gradients from the target layer during backward pass."""
        self.gradients = grad_output[0]
    
    def generate_cam(self, target_class):
        """
        Generate class activation map (CAM) for the target class.
        
        Args:
            target_class (int): Class index for which to generate the CAM.
        
        Returns:
            cam (np.ndarray): Normalized class activation map.
        """
        # Compute weights as average of gradients (global average pooling)
        gradients = self.gradients
        print('gradients.shape', gradients.shape)
        weights = torch.mean(gradients, dim=(2), keepdim=True)
        
        # Weighted sum of feature maps, followed by ReLU
        cam = F.relu(torch.sum(weights * self.feature_maps, dim=1, keepdim=True))
        
        # Normalize CAM to [0, 1] and convert to numpy array
        cam = cam.squeeze().cpu().data.numpy()
        cam -= np.min(cam)
        cam /= np.max(cam) if np.max(cam) != 0 else 1e-8  # Avoid division by zero
        
        return cam


    def remove_hooks(self):
        """Remove registered hooks to free resources."""
        self.hook.remove()
        self.hook_backward.remove()


def process_image(image_path, size=224):
    """
    Preprocess an image for model input.
    
    Args:
        image_path (str): Path to the image.
        size (int): Target size for resizing (default: 224).
    
    Returns:
        img_tensor (torch.Tensor): Preprocessed image tensor with batch dimension.
    """
    img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


def predict_image(model, image_tensor, att_seen, att_all, test_id, device):
    """
    Predict the class of an image using the trained model.
    
    Args:
        model: Trained model for prediction.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        att_seen: Attributes of seen classes.
        att_all: Attributes of all classes.
        test_id: List of class IDs for testing.
        device: Computing device (CPU/GPU).
    
    Returns:
        pred (int): Predicted class index.
        pred_cls_name: Name of the predicted class.
    """
    image_tensor = image_tensor.to(device)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(image_tensor, seen_att=att_seen, att_all=att_all)
    _, pred = torch.max(output, dim=1)  # Get predicted class index
    pred_cls_name = test_id[pred.item()]  # Map index to class name
    return pred.item(), pred_cls_name


def visualize_gradcam(image_path, model, att_seen, att_all, test_id, device, target_layer, size=224, save_dir="grad_cam"):
    """
    Generate and visualize Grad-CAM for a given image.
    
    Args:
        image_path (str): Path to the input image.
        model: Trained model.
        att_seen: Attributes of seen classes.
        att_all: Attributes of all classes.
        test_id: List of class IDs for testing.
        device: Computing device (CPU/GPU).
        target_layer: Target layer for Grad-CAM.
        size (int): Image size for processing (default: 224).
        save_dir (str): Directory to save visualization results (default: "grad_cam").
    
    Returns:
        pred_cls_name: Predicted class name.
        cam (np.ndarray): Generated class activation map.
    """
    # Preprocess the image
    image_tensor = process_image(image_path, size=size).to(device)
    
    # Predict the image class
    pred, pred_cls_name = predict_image(model, image_tensor, att_seen, att_all, test_id, device)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Compute gradients via backpropagation
    model.zero_grad()
    output = model(image_tensor, seen_att=att_seen, att_all=att_all)
    class_idx = pred  # Use the predicted class as target
    output[:, class_idx].backward()  # Backpropagate to get gradients
    
    # Generate CAM
    cam = gradcam.generate_cam(class_idx)
    
    # Load and preprocess original image
    img1 = Image.open(image_path).convert("RGB")  # Original size
    img = Image.open(image_path).convert("RGB").resize((size, size))  # Resized for CAM overlay
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    print(f"CAM shape: {cam.shape}")

    
    # Overlay CAM on original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    superimposed_img = np.float32(heatmap) + np.float32(img)
    superimposed_img = superimposed_img / np.max(superimposed_img)  # Normalize to [0, 1]

    # Convert to PIL image for saving
    superimposed_img_pil = Image.fromarray((superimposed_img * 255).astype(np.uint8))

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save results
    result_filename = os.path.join(save_dir, f"{pred_cls_name}_gradcam.png")
    result_filename1 = os.path.join(save_dir, f"{pred_cls_name}_original.png")
    
    img1.save(result_filename1)  # Save original image
    superimposed_img_pil.save(result_filename)  # Save CAM overlay
    
    # Visualize the result
    plt.imshow(superimposed_img)
    plt.title(f"Predicted: {pred_cls_name}")
    plt.axis('off')
    plt.show()

    # Clean up hooks
    gradcam.remove_hooks()
    
    return pred_cls_name, cam


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization for Zero-Shot Learning")
    parser.add_argument(
        "--config-file",
        default="config/cub.yaml",
        metavar="FILE",
        help="Path to the configuration file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()

    # Load configuration
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # Build and load the model
    model = build_gzsl_pipeline(cfg)
    model_dict = model.state_dict()
    saved_dict = torch.load('/root/shared-nvme/lichong/BiPAZSL/checkpoints/cub_best_model_1.pth')
    # Filter saved parameters to match model structure
    saved_dict = {k: v for k, v in saved_dict['model'].items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    
    # Move model to device
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)
    
    # Build dataloader and load dataset information
    is_distributed = False
    train_loader, test_unseen_loader, test_seen_loader, res = build_dataloader(cfg, is_distributed)

    # Example image path (modify as needed)
    image_path = '/root/shared-nvme/lichong/data/CUB/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'

    
    # Extract attribute information
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)
    att_all = torch.cat((att_seen, att_unseen), dim=0).to(device)
    test_id = res['train_test_id']
    
    # Define target layer for Grad-CAM (adjust based on model architecture)
    target_layer = model.backbone_0[10].norm1  # Example: 11th block's normalization layer in backbone_0
    
    # Generate and visualize Grad-CAM
    visualize_gradcam(image_path, model, att_seen, att_all, test_id, device, target_layer, save_dir="grad_cam")


if __name__ == '__main__':
    main()
