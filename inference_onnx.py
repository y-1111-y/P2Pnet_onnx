
import argparse
import onnxruntime as ort
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from PIL import Image
import cv2
import os
import time

def process_one_image(dir_path, img_name, transform, session):#此函数默认batch_size为1
    img_path = os.path.join(dir_path, img_name)
    img_raw = Image.open(img_path).convert('RGB')
    width, height = img_raw.size #PIL image 宽在前高在后，tensor高在前宽在后
    if width < height:#调整方向，防止图片失真
        img_raw = img_raw.transpose(Image.ROTATE_90)
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)

    # 定义模型输入，进行推理
    ort_inputs = {'input': samples.numpy()}
    pred_logits, pred_points = session.run(['pred_logits', 'pred_points'], ort_inputs)
    # 推理结果
    outputs_scores = torch.nn.functional.softmax(torch.Tensor(pred_logits), -1)[:, :, 1][0]
    outputs_points = torch.Tensor(pred_points[0])
    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    #cv2.imshow('result', img_to_draw)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(args.output_dir, img_name.replace('.jpg', 'pred{}.jpg'.format(predict_cnt))), img_to_draw)


def main(args):
    #读取模型
    weight_path = args.weight_path
    cuda = args.cuda
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(weight_path, providers=providers)
    #定义数据预处理的正则化部分
    transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    #读取图片
    dir_path = args.input_dir
    img_paths = [i for i in os.listdir(dir_path) if 'jpg' in i]
    #img_paths = ['3_2_89.jpg']
    for img_name in img_paths:
        t1 = time.time()
        process_one_image(dir_path, img_name, transform, session)
        t2 = time.time()
        print('{} spend time: {}'.format(img_name, t2 - t1))

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation by onnx', add_help=False)

    parser.add_argument('--input_dir', default='',
                        help='path where to read images')
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--cuda', action='store_true', help='if use cuda')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet_onnx evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
