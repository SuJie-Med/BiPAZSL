from PIL import Image
import torch
from torchvision import transforms

def process_image(image_path, size=224):
    """
    Preprocess a single image for model input.
    
    Args:
        image_path (str): Path to the image file
        size (int): Target size for image resizing (default: 224)
    
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
    """
    # Load image and convert to RGB
    img = Image.open(image_path).convert("RGB")

    # Define preprocessing pipeline (matches training configuration)
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),        # Resize image to target dimensions
        transforms.ToTensor(),                  # Convert to tensor (HWC -> CHW)
        transforms.Normalize(                   # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Add batch dimension (1, C, H, W)
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


def predict_image(model, image_tensor, att_seen, att_unseen, att_all, test_id, device):
    """
    Perform inference on a single image and return prediction results.
    
    Args:
        model (torch.nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        att_seen (torch.Tensor): Attributes of seen classes
        att_unseen (torch.Tensor): Attributes of unseen classes
        att_all (torch.Tensor): Attributes of all classes
        test_id (np.array): Array of test class IDs/names
        device (torch.device): Computing device (CPU/GPU)
    
    Returns:
        int: Predicted class index
        str: Predicted class name
    """
    model.eval()  # Set model to evaluation mode
    image_tensor = image_tensor.to(device)  # Move tensor to target device
    
    # Inference with no gradient computation
    with torch.no_grad():
        scores = model(image_tensor, seen_att=att_seen, att_all=att_all)  # Get class scores
        _, pred = scores.max(dim=1)  # Get index of highest score
        pred = pred.item()  # Convert tensor to scalar

    # Map predicted index to class name
    pred_cls_name = test_id[pred]
    return pred, pred_cls_name


def test_single_image(image_path, model, att_seen, att_unseen, att_all, test_id, device):
    """
    End-to-end testing pipeline for a single image.
    
    Args:
        image_path (str): Path to the image file
        model (torch.nn.Module): Trained model
        att_seen (torch.Tensor): Attributes of seen classes
        att_unseen (torch.Tensor): Attributes of unseen classes
        att_all (torch.Tensor): Attributes of all classes
        test_id (np.array): Array of test class IDs/names
        device (torch.device): Computing device (CPU/GPU)
    """
    # Preprocess the image
    image_tensor = process_image(image_path, size=224)
    
    # Get prediction results
    pred, pred_cls_name = predict_image(
        model, 
        image_tensor, 
        att_seen, 
        att_unseen, 
        att_all, 
        test_id, 
        device
    )
    
    # Print results
    print(f"Predicted Class ID: {pred}")
    print(f"Predicted Class Name: {pred_cls_name}")


def main():
    # --------------------------
    # Configuration and Setup
    # --------------------------
    # Note: You need to properly initialize cfg and res before running
    # These should be loaded using your actual configuration and dataset setup
    
    # Load model architecture
    model = build_gzsl_pipeline(cfg)  # Replace with actual model builder
    
    # Load trained weights
    checkpoint_path = 'checkpoints/gzsl_cub_train_1ceng.pth'
    saved_dict = torch.load(checkpoint_path)['model']
    model_dict = model.state_dict()
    
    # Filter and load compatible weights
    saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    
    # Move model to device
    device = torch.device(cfg.MODEL.DEVICE)  # e.g., 'cuda' or 'cpu'
    model = model.to(device)
    
    # Load attribute data (from dataset metadata)
    att_unseen = res['att_unseen'].to(device)  # Replace 'res' with actual dataset info
    att_seen = res['att_seen'].to(device)
    att_all = torch.cat((att_seen, att_unseen), dim=0).to(device)
    test_id = res['test_id']  # Class ID/name mapping
    
    # --------------------------
    # Run Single Image Test
    # --------------------------
    image_path = 'path_to_your_image.jpg'  # Replace with actual image path
    test_single_image(
        image_path, 
        model, 
        att_seen, 
        att_unseen, 
        att_all, 
        test_id, 
        device
    )


if __name__ == '__main__':
    main()
