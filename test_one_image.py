from PIL import Image
import torch
from torchvision import transforms

def process_image(image_path, size=224):
    """
    预处理单张图片
    Args:
        image_path (str): 图片文件的路径
        size (int): 图片尺寸
    Returns:
        torch.Tensor: 处理后的图片张量
    """
    img = Image.open(image_path).convert("RGB")

    # 你可以根据训练时的预处理方式来选择
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),        # 调整图片大小
        transforms.ToTensor(),                  # 转换为tensor
        transforms.Normalize(                   # 归一化（根据训练时使用的均值和标准差）
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    img_tensor = preprocess(img).unsqueeze(0)  # 添加 batch 维度
    return img_tensor

def predict_image(model, image_tensor, att_seen, att_unseen, att_all, test_id, device):
    """
    对单张图片进行推理并输出结果
    Args:
        model (torch.nn.Module): 训练好的模型
        image_tensor (torch.Tensor): 处理后的图片张量
        att_seen (torch.Tensor): 见过类别的属性
        att_unseen (torch.Tensor): 未见过类别的属性
        att_all (torch.Tensor): 所有类别的属性
        test_id (np.array): 测试集类别ID
        device (torch.device): 计算设备
    Returns:
        int: 预测类别
        str: 预测类别名称
    """
    model.eval()  # 设置模型为评估模式
    image_tensor = image_tensor.to(device)  # 将图片数据移至GPU（如果使用GPU）
    
    # 推理
    with torch.no_grad():
        scores = model(image_tensor, seen_att=att_seen, att_all=att_all)  # 预测分数
        _, pred = scores.max(dim=1)  # 获取最大分数的索引作为预测结果
        pred = pred.item()  # 获取预测的类别标签

    # 获取类别名称
    pred_cls_name = test_id[pred]  # 使用对应的ID映射到类别名称
    return pred, pred_cls_name

def test_single_image(image_path, model, att_seen, att_unseen, att_all, test_id, device):
    # 处理单张图片
    image_tensor = process_image(image_path, size=224)  # 你可以根据需要调整图片大小
    
    # 获取预测结果
    pred, pred_cls_name = predict_image(model, image_tensor, att_seen, att_unseen, att_all, test_id, device)
    
    # 输出结果
    print(f"Predicted Class ID: {pred}")
    print(f"Predicted Class Name: {pred_cls_name}")


def main():
    # 假设你已经加载了模型、数据等
    model = build_gzsl_pipeline(cfg)
    model_dict = model.state_dict()
    saved_dict = torch.load('checkpoints/gzsl_cub_train_1ceng.pth')
    saved_dict = {k: v for k, v in saved_dict['model'].items() if k in model_dict}
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict)
    
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)
    
    # 加载其他必要的数据（例如属性信息等）
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)
    att_all = torch.cat((att_seen, att_unseen), dim=0).to(device)
    test_id = res['test_id']

    image_path = 'path_to_your_image.jpg'  # 替换为实际图片路径

    # 测试单张图片
    test_single_image(image_path, model, att_seen, att_unseen, att_all, test_id, device)

if __name__ == '__main__':
    main()
