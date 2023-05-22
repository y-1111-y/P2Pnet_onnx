Put these two files under the [P2Pnet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) directory. 

Run the export.py with your parameters. Such as this.
``` shell
python export.py --weight_path ckpt/best_mae.pth --simplify
```
Run inference_onnx.py with your parameters. Such as this.
``` shell
python inference_onnx.py --input_dir datasets/test --output_dir output --weight_path ckpt/best_mae.onnx
```
