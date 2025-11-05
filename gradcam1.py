import torch
import torch.nn.functional as F
import numpy as np
import argparse
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from models.modeling import build_gzsl_pipeline
from models.config import cfg
from data import build_dataloader

def get_image_info(image_path, new_img_files, label, attribute):
    """
    根据图片路径获取其类别 ID 和属性信息。
    
    参数：
    - image_path (str): 输入图片的路径。
    - new_img_files (np.ndarray): 所有图片路径的数组。
    - label (np.ndarray): 所有图片的类别标签数组。
    - attribute (np.ndarray): 类别属性数组。
    
    返回：
    - image_id (int): 图片的类别 ID。
    - image_attr (np.ndarray): 图片的属性向量。
    """
    # 查找图片路径的索引
    image_index = np.where(new_img_files == image_path)[0]
    
    if len(image_index) == 0:
        raise ValueError(f"Image path '{image_path}' not found in dataset.")
    
    image_index = image_index[0]  # 获取第一个匹配的索引
    lable_id = label[image_index]  # 获取图片类别 ID
    image_attr = attribute[image_id]  # 获取图片对应的属性向量
    
    return image_index, lable_id, image_attr


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # 注册钩子函数来获取特征图和梯度
        self.hook = self.target_layer.register_forward_hook(self.save_feature_maps)
        self.hook_backward = self.target_layer.register_full_backward_hook(self.save_gradients)

    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
        print(f"Feature maps shape: {self.feature_maps.shape}")  # 打印特征图的形状
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, target_class):
        # 获取梯度的平均值作为权重
        gradients = self.gradients
        print('gradients.shape', gradients.shape)
        weights = torch.mean(gradients, dim=(2), keepdim=True)
        
        # 计算加权特征图
        cam = F.relu(torch.sum(weights * self.feature_maps, dim=1, keepdim=True))
        
        # 归一化并调整大小以匹配输入图像的尺寸
        cam = cam.squeeze().cpu().data.numpy()
        cam -= np.min(cam)
        cam /= np.max(cam)
        
        return cam


    def remove_hooks(self):
        # 移除钩子函数
        self.hook.remove()
        self.hook_backward.remove()


# 图像处理函数
def process_image(image_path, size=224):
    img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
    return img_tensor

# 预测函数
def predict_image(model, image_tensor, att_seen, att_all, test_id, device):
    image_tensor = image_tensor.to(device)
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        output = model(image_tensor, seen_att=att_seen, att_all=att_all)
    _, pred = torch.max(output, dim=1)
    pred_cls_name = test_id[pred.item()]  # 假设test_id是类名称的列表
    return pred.item(), pred_cls_name


def visualize_gradcam(image_path, model, att_seen, att_all, test_id, device, target_layer, size=224, save_dir="grad_cam"):
    # 处理图片
    image_tensor = process_image(image_path, size=size).to(device)
    
    # 对图片进行推理
    pred, pred_cls_name = predict_image(model, image_tensor, att_seen, att_all, test_id, device)
    
    # 创建 Grad-CAM 对象
    gradcam = GradCAM(model, target_layer)
    
    # 获取目标类别的梯度
    model.zero_grad()
    output = model(image_tensor, seen_att=att_seen, att_all=att_all)
    class_idx = pred  # 使用模型预测的类别
    output[:, class_idx].backward()  # 反向传播，计算梯度
    
    # 生成 Grad-CAM
    cam = gradcam.generate_cam(class_idx)
    
    # 加载原始图片
    img1 = Image.open(image_path).convert("RGB")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    img = np.array(img) / 255.0
    
    # 将 CAM 图像映射回原始图像的尺寸
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    print(f"CAM shape: {cam.shape}")

    
    # 叠加 CAM 和原始图像
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    superimposed_img = np.float32(heatmap) + np.float32(img)
    superimposed_img = superimposed_img / np.max(superimposed_img)  # 归一化

    # 将 numpy 数组转换为 PIL 图像
    superimposed_img_pil = Image.fromarray((superimposed_img * 255).astype(np.uint8))

    # 创建保存文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存结果图像
    result_filename = os.path.join(save_dir, f"{pred_cls_name}_gradcam.png")
    result_filename1 = os.path.join(save_dir, f"{pred_cls_name}_yuantu.png")
    
    # plt.imsave(result_filename, superimposed_img)
    # plt.imsave(result_filename1, img1)
    # 使用 PIL 保存原图
    img1.save(result_filename1)
    superimposed_img_pil.save(result_filename)
    
    # 可视化结果
    plt.imshow(superimposed_img)
    plt.title(f"Predicted: {pred_cls_name}")
    plt.axis('off')
    plt.show()

    # 清除 Grad-CAM 钩子
    gradcam.remove_hooks()
    
    return pred_cls_name, cam


def main():
    parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")
    parser.add_argument(
        "--config-file",
        default="config/cub.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # 加载配置文件
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # 构建模型
    model = build_gzsl_pipeline(cfg)
    model_dict = model.state_dict()
    saved_dict = torch.load('/root/shared-nvme/lichong/PSVMA/checkpoints/cub_best_model_1.pth')
    saved_dict = {k: v for k, v in saved_dict['model'].items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)
    
    # 构建数据加载器并加载数据
    is_distributed = False
    train_loader, test_unseen_loader, test_seen_loader, res = build_dataloader(cfg, is_distributed)

    image_path = '/root/shared-nvme/lichong/data/CUB/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'

    
    # 提取属性信息
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)
    att_all = torch.cat((att_seen, att_unseen), dim=0).to(device)
    test_id = res['train_test_id']
    
    # 打印属性信息
    # print(f"Image Path: {image_path}")


    # 选择最后一层卷积层作为 Grad-CAM 目标层
    target_layer = model.backbone_0[10].norm1  # 例如，访问 `backbone_0` 中的第3个 `Block` 的 `attn` 层
    
    # 测试单张图片并生成 Grad-CAM 可视化
    visualize_gradcam(image_path, model, att_seen, att_all, test_id, device, target_layer, save_dir="grad_cam")

if __name__ == '__main__':
    main()
