import argparse
from engine import *
from models import build_model
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet export', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--img-size', nargs='+', type=int, default=[1280, 1280], help='image size')  # height, width

    parser.add_argument('--batch-size', type=int, default=1, help='batch size')

    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--device', default='cpu', type=str, help='the device used for evaluation, cpu or cuda:device_id')

    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')

    return parser

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser('P2PNet export script', parents=[get_args_parser()])
    args = parser.parse_args()
    print('args', args)
    device = args.device
    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = build_model(args)
    # move to device
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
    model_name = args.weight_path.replace('pth', 'onnx')
    input_data = torch.randn(args.batch_size, 3, *args.img_size).to(device)
    output = model(input_data)

    input_name = 'input'
    pred_logits = 'pred_logits'
    pred_points = 'pred_points'
    torch.onnx.export(model,
                  input_data,
                  model_name,
                  opset_version=11,
                  input_names=[input_name],
                  output_names=[pred_logits, pred_points],
                  dynamic_axes={
                      input_name: {0: 'batch_size'},
                      pred_logits: {0: 'batch_size'},
                      pred_points: {0: 'batch_size'}}
                  )
    import onnx

    # load ONNX model
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    if args.simplify:
        try:
            import onnxsim

            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    onnx.save(onnx_model,model_name)
    print('ONNX export success, saved as %s' % model_name)
