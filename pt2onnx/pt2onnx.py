# Conda Environment : Model 
# paddleocr : PaddleOCR / PP-LCNetV2-Reid / PPLCNetV1-Pedes
# torchreid : osnet / p2pnet / apgcc
# yolo_rknn : yolov8 / yolo26
# mmpose    : mmpose (rtmpose3d), actionmm2
# tf_train  : MobileNetV2, MobileNetV3
# vlm_infer : VideoMAE

import os
import re
import sys
import math
import onnx
import torch
import shutil
import tempfile
import subprocess

from pathlib import Path

# General Config
MODEL_CKPT = '/home/paiworker1/KX/.Workspace/.Projects/videomae/videomae_small'
MODEL_TYPE = 'VideoMAE'; assert MODEL_TYPE in ['OpenMM', 
                                             'YOLOv8', 'YOLO26',
                                             'YOLOE', # deprecated
                                             'OSNet', 
                                             'PaddleOCR', 'PPLCNetV1-Pedes',
                                             'PPLCNetV2',
                                             'P2PNet', 'APGCC',
                                             'MobileNetV2', 'MobileNetV3',
                                             'VideoMAE']

# Only for Yolov8/Yolo26
TASK = 'CLS'; assert TASK in ['CLS', 'DET', 'SEG', 'POS'] # Classification, Detection, Segmentation, Pose-Estimation
BATCH_SIZE = 1
IMGSZ = (128, 128) # (Height, Width)

# Only for YoloE
CLASSES = ["person", "bus"]

# Only for OpenMM (MMaction / MMpose) model convertion
CONVERT_TOOL = '/home/paiworker1/KX/.Workspace/.Tools/mmdeploy/tools/deploy.py'
DEPLOY_CONFIG = '/home/paiworker1/KX/.Workspace/.Tools/mmdeploy/configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py'
MODEL_CONFIG = '/home/paiworker1/KX/.Workspace/.Tools/mmaction2/configs/recognition/tsm/v3.py'
TEST_INPUT = '/home/paiworker1/KX/.Workspace/.Projects/action_recognization/ActionMM2/input_for_v2.mp4' # random input to verity the output, if your model input video, then this is a path to a random video, etc.

# DEPLOY_CONFIG = '/home/paiworker1/KX/.Workspace/.Tools/mmdeploy/configs/mmpose/pose-detection_onnxruntime_static.py'
# MODEL_CONFIG = '/home/paiworker1/KX/.Workspace/.Tools/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py'
# TEST_INPUT = '/home/paiworker1/KX/.Workspace/.Projects/mmpose/py/frame_086.jpg'

# The Export for RTM3D is modified to output 2 parts, main model and the remaining sub-head, which is not compatible to convert to rknn, so leave the last part as onnx
# There should be 2 onnx file in the output folder, one is the main model, the other is the sub-head only
# you should only convert the main model to rknn, leaving the sub-head only in onnx to run in cpu

# Only for OSNet model convertion
MODEL_NAME = 'osnet_x1_0'
INPUT_SIZE = [1, 3, 256, 128]

# Only for PaddleOCR model convertion
# RKNN runtime does not reliably support dynamic batch inference, even if ONNX does.
# Therefore, we fix the batch size here to avoid shape inconsistency and runtime crashes.
IMG_BATCH_SIZE = 1
REGION_BATCH_SIZE = 20
MAX_WIDTH_RECOG = 650
DOC_PREPROCESS_CROP = 224
DET_IMG_SIZE = 1696

# Only for PP-LCNetV1-Pedestrian model conversion
PP_LCNETV1_BATCH_SIZE = 1

# Only for PP-LCNetV2 model conversion
PP_LCNETV2_BATCH_SIZE = 1
PP_LCNETV2_IMG_SIZE = (224, 224) # (Height, Width)

# Only for P2PNet
BACKBONE = 'vgg16_bn'
BACKBONE_CKPT = '/home/paiworker1/KX/.Workspace/.Projects/crowd_count/P2PNet/weights/vgg16_bn-6c64b313.pth'
ROW, LINE = 2, 2
INPUT_SIZE_P2P = [6, 3, 1024, 1280]

# Only for APGCC
BACKBONE_CKPT = '/home/paiworker1/KX/.Workspace/.Projects/crowd_count/APGCC-ONNX/weights/vgg16_bn-6c64b313.pth'
INPUT_SIZE_APGCC = [6, 3, 640, 800]

# Only for MobileNetV2
QUANT_MOBILENETV2 = False # (6/2/2026) deprecated, dont use, use onnx2rknn.py to do quantization
QUANT_DATASET_FOLDER = '/home/paiworker1/KX/.Workspace/.Projects/PianoTag/bigdata/train'
QUANT_LABEL_TXT = '/home/paiworker1/KX/.Workspace/.Projects/PianoTag/bigdata/train/labels.txt'
# each line is like relative path of the images from the dataset folder, then a space, then the label index
# e.g., noise/img1.jpg 0
INPUT_SIZE_MOBILENETV2 = [2, 3, 32, 32]

# Only for MobileNetV3
INPUT_SIZE_MOBILENETV3 = [16, 3, 224, 224]

# Only for VideoMAE
INPUT_SIZE_VIDEOMAE = [16, 3, 224, 224]

if MODEL_TYPE in ['YOLOv8', 'YOLO26']:
    from ultralytics import YOLO
    if TASK in ['DET', 'POS']:
        if TASK == 'POS' and MODEL_TYPE == 'YOLO26':
            raise NotImplementedError
        if IMGSZ is not None:
            YOLO(MODEL_CKPT).export(format="rknn", imgsz=IMGSZ)
        else:
            YOLO(MODEL_CKPT).export(format="rknn")
    else:
        print(f'26/01/2026, Never test opset=19 on yolo model for task segmentation or classification')
        print(f'if you encounter error, try changing the opset')
        if IMGSZ is not None:
            YOLO(MODEL_CKPT).export(format="onnx", imgsz=IMGSZ, batch=BATCH_SIZE, opset=19, simplify=False)
        else:
            YOLO(MODEL_CKPT).export(format="onnx", batch=BATCH_SIZE, opset=19, simplify=False)
        
elif MODEL_TYPE == 'YOLOE':
    from ultralytics import YOLOE
    model = YOLOE(MODEL_CKPT)
    model.set_classes(CLASSES, model.get_text_pe(CLASSES))
    if IMGSZ is not None:
        model.export(format="onnx", imgsz=IMGSZ, opset=19)
    else:
        model.export(format="onnx", opset=19)

elif MODEL_TYPE == 'OpenMM':

    work_dir = os.path.dirname(MODEL_CKPT)
    default_onnx = os.path.join(work_dir, "end2end.onnx")
    ckpt_name = os.path.splitext(os.path.basename(MODEL_CKPT))[0]
    final_onnx_path = os.path.join(work_dir, ckpt_name + ".onnx")

    subprocess.run([
        "python3",
        CONVERT_TOOL,
        DEPLOY_CONFIG,
        MODEL_CONFIG,
        MODEL_CKPT,
        TEST_INPUT,
        '--work-dir', work_dir
    ])

    if os.path.exists(default_onnx):
        shutil.move(default_onnx, final_onnx_path)
        print(f"ONNX model saved at: {final_onnx_path}")
    else:
        print(f"Export failed: {default_onnx} not found")

    if 'rtmw3d' in MODEL_CONFIG:
        if os.path.exists(default_onnx.replace('.onnx', '_subhead.onnx')):
            shutil.move(default_onnx.replace('.onnx', '_subhead.onnx'), final_onnx_path.replace('.onnx', '_subhead.onnx'))
            print(f"Sub-head ONNX model saved at: {final_onnx_path.replace('.onnx', '_subhead.onnx')}")
        else:
            print('Sub-head export failed: subhead ONNX not found')

elif MODEL_TYPE == 'OSNet':
    from torchreid.utils.feature_extractor import FeatureExtractor
    def export_onnx(model, im, file, opset, train=False, dynamic=True, simplify=False):
        
        try:
            f = file.with_suffix('.onnx')
            print(f'\nStarting export with onnx {onnx.__version__}...')

            torch.onnx.export(
                model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
                im.cpu() if dynamic else im,
                f,
                verbose=False,
                opset_version=opset,
                training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                do_constant_folding=not train,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {
                        0: 'batch',
                    },  # shape(x,3,256,128)
                    'output': {
                        0: 'batch',
                    }  # shape(x,2048)
                } if dynamic else None
            )
            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            onnx.save(model_onnx, f)

            # Simplify
            if simplify:
                try:
                    import onnxsim

                    print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        dynamic_input_shape=dynamic,
                        input_shapes={'t0': list(im.shape)} if dynamic else None)
                    assert check, 'assert check failed'
                    onnx.save(model_onnx, f)
                except Exception as e:
                    print(f'simplifier failure: {e}')
            print(f'export success, saved as {f}')
        except Exception as e:
            print(f'export failure: {e}')
        return f

    MODEL_CKPT = Path(MODEL_CKPT)

    extractor = FeatureExtractor(
        # get rid of dataset information DeepSort model name
        model_name=MODEL_NAME,
        model_path=MODEL_CKPT,
        device='cuda'
    )

    im = torch.zeros(INPUT_SIZE).cuda()
    export_onnx(extractor.model.eval(), im, MODEL_CKPT, 12, train=False, dynamic=False, simplify=False)  # opset 12

elif MODEL_TYPE == 'PaddleOCR':

    print(f"""
    ################################################################################
    #  CONVERSION WARNING: Paddle Model RKNN DEPLOYMENT COMPATIBILITY              #
    #  --------------------------------------------------------------------------  #
    #  Known issue: The conversion randomly injects problematic node patterns.     #
    #  Pattern: [Identity] -> [Reshape] -> [Add] appearing beside inputs.          #
    #  Status:  This issue is sporadically appearing/disappearing.                 #
    #                                                                              #
    #  LAST KNOWN SUCCESSFUL CONFIGURATION:                                        #
    #  - paddleocr           : 3.2.0                                               #
    #  - paddlepaddle-gpu    : 3.2.0 @https://www.paddlepaddle.org.cn/             #
    #                                 packages/stable/cu118/                       #
    #  - paddlex             : 3.2.1                                               #
    #  - paddle2onnx         : 2.0.2rc3 (Installed via: paddlex --install)         #
    #  - onnx                : 1.17.0                                              #
    #  - onnxruntime         : 1.23.2                                              #
    #                                                                              #
    #  TROUBLESHOOTING IF RKNN BUILD FAILS:                                        #
    #  1. Check if packages version match.                                         #
    #  2. Nothing we can do for now, the last error fix it by itself randomly, one #
    #     possible reason could be out of gpu memory, but it never being confirmed #
    ################################################################################
    """)

    # Force stability settings, possibly helps, i dont know
    os.environ['FLAGS_enable_pir_api'] = '0'
    os.environ['FLAGS_enable_pir_in_executor'] = '0'

    assert os.path.isdir(MODEL_CKPT), 'For PaddleOCR, the MODEL_CKPT should be a folder, containing a list of models'
    
    for model in os.listdir(MODEL_CKPT):
        
        with tempfile.TemporaryDirectory() as tmpdir:

            subprocess.run([
                "paddlex",
                "--paddle2onnx",
                "--paddle_model_dir", os.path.join(MODEL_CKPT, model),
                "--onnx_model_dir", MODEL_CKPT,
                "--opset_version", "11"
            ], cwd=tmpdir, check=True, stdout=subprocess.DEVNULL)

        os.remove(f'{MODEL_CKPT}/inference.yml')

        def fix_onnx_dims_by_name(model_path, dim_name_map, save_path):
            model = onnx.load(model_path)
            graph = model.graph

            def fix_tensor_shape(tensor_shape):
                for d in tensor_shape.dim:
                    replaced = False
                    need_eval = False
                    expr = d.dim_param
                    for dim_name in dim_name_map.keys():
                        if expr == dim_name:
                            d.dim_value = dim_name_map[dim_name] # just set value
                            replaced = True
                        elif dim_name in expr:
                            expr = expr.replace(dim_name, str(dim_name_map[dim_name]))
                            replaced = True
                            need_eval = True

                    if 'DynamicDimension.' in expr and not replaced:
                        print(f'Warning: exist {d.dim_param}')
                    elif need_eval:
                        try:
                            allowed_names = {"floor": math.floor, "ceil": math.ceil}
                            d.dim_value = eval(expr, {"__builtins__": None}, allowed_names)
                        except Exception as e:
                            assert 0, f"Cannot evaluate {d.dim_param}: {e}"

            # Fix inputs, outputs, value_info
            for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
                t = value_info.type
                if t.tensor_type.HasField("shape"):
                    fix_tensor_shape(t.tensor_type.shape)

            onnx.save(model, save_path)
            os.remove(model_path)

        # Fix Batch size of onnx model
        if model.endswith('doc_ori'):

            fix_onnx_dims_by_name(
                f"{MODEL_CKPT}/inference.onnx", 
                {
                    'DynamicDimension.0': IMG_BATCH_SIZE,
                    'DynamicDimension.1': DOC_PREPROCESS_CROP,
                    'DynamicDimension.2': DOC_PREPROCESS_CROP
                },
                f"{MODEL_CKPT}/{model}.onnx"
            )
        elif model.endswith('det'):
            if DET_IMG_SIZE % 32 != 0:
                print(f'[Warning] {DET_IMG_SIZE} // 32 != 0, this model will possibly failed on conversion to rknn')
            fix_onnx_dims_by_name(
                f"{MODEL_CKPT}/inference.onnx", 
                {
                    'DynamicDimension.0': IMG_BATCH_SIZE,
                    'DynamicDimension.1': DET_IMG_SIZE,
                    'DynamicDimension.2': DET_IMG_SIZE,
                },
                f"{MODEL_CKPT}/{model}.onnx"
            )
        elif model.endswith('textline_ori'):
            
            fix_onnx_dims_by_name(
                f"{MODEL_CKPT}/inference.onnx", 
                {
                    'DynamicDimension.0': REGION_BATCH_SIZE
                },
                f"{MODEL_CKPT}/{model}.onnx"
            )
        elif model.endswith('rec'):
            fix_onnx_dims_by_name(
                f"{MODEL_CKPT}/inference.onnx", 
                {
                    'DynamicDimension.0': REGION_BATCH_SIZE,
                    'DynamicDimension.1': MAX_WIDTH_RECOG
                },
                f"{MODEL_CKPT}/{model}.onnx"
            )
        
        else:
            os.rename(f"{MODEL_CKPT}/inference.onnx", f"{MODEL_CKPT}/{model}.onnx")

elif MODEL_TYPE == 'PPLCNetV1-Pedes':

    assert os.path.isdir(MODEL_CKPT), 'For PPLCNetV1-Pedes, the MODEL_CKPT should be a folder'
    
    with tempfile.TemporaryDirectory() as tmpdir:

        subprocess.run([
            "paddlex",
            "--paddle2onnx",
            "--paddle_model_dir", MODEL_CKPT,
            "--onnx_model_dir", Path(MODEL_CKPT).parent,
            "--opset_version", "11"
        ], cwd=tmpdir, check=True, stdout=subprocess.DEVNULL)

    def fix_onnx_dims_by_name(model_path, dim_name_map, save_path):
        model = onnx.load(model_path)
        graph = model.graph

        def fix_tensor_shape(tensor_shape):
            for d in tensor_shape.dim:
                replaced = False
                need_eval = False
                expr = d.dim_param
                for dim_name in dim_name_map.keys():
                    if expr == dim_name:
                        d.dim_value = dim_name_map[dim_name] # just set value
                        replaced = True
                    elif dim_name in expr:
                        expr = expr.replace(dim_name, str(dim_name_map[dim_name]))
                        replaced = True
                        need_eval = True

                if 'DynamicDimension.' in expr and not replaced:
                    print(f'Warning: exist {d.dim_param}')
                elif need_eval:
                    try:
                        allowed_names = {"floor": math.floor, "ceil": math.ceil}
                        d.dim_value = eval(expr, {"__builtins__": None}, allowed_names)
                    except Exception as e:
                        assert 0, f"Cannot evaluate {d.dim_param}: {e}"

        # Fix inputs, outputs, value_info
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            t = value_info.type
            if t.tensor_type.HasField("shape"):
                fix_tensor_shape(t.tensor_type.shape)

        onnx.save(model, save_path)
        os.remove(model_path)

    # Fix Batch size of onnx model
    fix_onnx_dims_by_name(
        f"{Path(MODEL_CKPT).parent}/inference.onnx", 
        {
            'DynamicDimension.0': PP_LCNETV1_BATCH_SIZE
        },
        f"{Path(MODEL_CKPT).parent}/{Path(MODEL_CKPT).name}.onnx"
    )

elif MODEL_TYPE == 'PPLCNetV2':

    import paddle

    from paddleclas.ppcls.arch.backbone import PPLCNetV2_base
    os.environ['SKIP_FC'] = '1'

    model = PPLCNetV2_base(pretrained=True)
    state_dict = paddle.load(MODEL_CKPT)
    model.set_state_dict(state_dict)
    model.eval()

    input_spec = [paddle.static.InputSpec(shape=[PP_LCNETV2_BATCH_SIZE, 3, PP_LCNETV2_IMG_SIZE[0], PP_LCNETV2_IMG_SIZE[1]], dtype='float32', name='input')]

    paddle.onnx.export(
        model,
        MODEL_CKPT.replace('.pdparams', ''),
        input_spec=input_spec,
        opset_version=11,
        export_for_deployment=True
    )

elif MODEL_TYPE == 'P2PNet':
    import p2pnet.vgg_ as models
    from p2pnet.p2pnet import P2PNet
    from p2pnet.backbone import Backbone_VGG
    import onnxsim

    os.environ['BACKBONE_CKPT'] = BACKBONE_CKPT
    backbone = Backbone_VGG(BACKBONE, True)
    model = P2PNet(backbone, ROW, LINE)

    checkpoint = torch.load(MODEL_CKPT, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    model_name = MODEL_CKPT.replace('pth', 'onnx')
    input_data = torch.randn(*INPUT_SIZE_P2P)
    output = model(input_data)

    torch.onnx.export(model,
                    input_data,
                    model_name,
                    opset_version=11,
                    input_names=['input'],
                    output_names=['pred_logits', 'pred_points']
                    )

    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'

    onnx.save(onnx_model,model_name)

elif MODEL_TYPE == 'APGCC':
    from apgcc.model import Model_builder
    import onnxsim

    os.environ['BACKBONE_CKPT'] = BACKBONE_CKPT
    model = Model_builder()

    pretrained_dict = torch.load(MODEL_CKPT, map_location='cpu')
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)

    model.eval()

    model_name = MODEL_CKPT.replace('pth', 'onnx')
    input_data = torch.randn(*INPUT_SIZE_APGCC)
    output = model(input_data)

    torch.onnx.export(model,
                    input_data,
                    model_name,
                    opset_version=17,
                    input_names=['input'],
                    output_names=['pred_logits', 'pred_points', 'offset']
                    )

    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'

    onnx.save(onnx_model,model_name)

elif MODEL_TYPE == 'MobileNetV2':
    model = torch.load(MODEL_CKPT, weights_only=False).cpu()
    model_name = MODEL_CKPT.replace('pth', 'onnx')

    input_data = torch.randn(*INPUT_SIZE_MOBILENETV2).cpu() # Match your input_shape

    if QUANT_MOBILENETV2:
        # https://github.com/intel/neural-compressor/tree/master/examples/deprecated/onnxrt/image_recognition/mobilenet_v2/quantization/ptq_static

        import cv2
        import numpy as np
        import collections
        import onnxruntime as ort
        
        from PIL import Image
        from sklearn.metrics import accuracy_score
        from neural_compressor.config import AccuracyCriterion
        from neural_compressor import quantization, PostTrainingQuantConfig
        
        def _topk_shape_validate(preds, labels):
            # preds shape can be Nxclass_num or class_num(N=1 by default)
            # it's more suitable for 'Accuracy' with preds shape Nx1(or 1) output from argmax
            if isinstance(preds, int):
                preds = [preds]
                preds = np.array(preds)
            elif isinstance(preds, np.ndarray):
                preds = np.array(preds)
            elif isinstance(preds, list):
                preds = np.array(preds)
                preds = preds.reshape((-1, preds.shape[-1]))

            # consider labels just int value 1x1
            if isinstance(labels, int):
                labels = [labels]
                labels = np.array(labels)
            elif isinstance(labels, tuple):
                labels = np.array([labels])
                labels = labels.reshape((labels.shape[-1], -1))
            elif isinstance(labels, list):
                if isinstance(labels[0], int):
                    labels = np.array(labels)
                    labels = labels.reshape((labels.shape[0], 1))
                elif isinstance(labels[0], tuple):
                    labels = np.array(labels)
                    labels = labels.reshape((labels.shape[-1], -1))
                else:
                    labels = np.array(labels)
            # labels most have 2 axis, 2 cases: N(or Nx1 sparse) or Nxclass_num(one-hot)
            # only support 2 dimension one-shot labels
            # or 1 dimension one-hot class_num will confuse with N

            if len(preds.shape) == 1:
                N = 1
                class_num = preds.shape[0]
                preds = preds.reshape([-1, class_num])
            elif len(preds.shape) >= 2:
                N = preds.shape[0]
                preds = preds.reshape([N, -1])
                class_num = preds.shape[1]

            label_N = labels.shape[0]
            assert label_N == N, 'labels batch size should same with preds'
            labels = labels.reshape([N, -1])
            # one-hot labels will have 2 dimension not equal 1
            if labels.shape[1] != 1:
                labels = labels.argsort()[..., -1:]
            return preds, labels

        class TopK:
            def __init__(self, k=1):
                self.k = k
                self.num_correct = 0
                self.num_sample = 0

            def update(self, preds, labels, sample_weight=None):
                preds, labels = _topk_shape_validate(preds, labels)
                preds = preds.argsort()[..., -self.k:]
                if self.k == 1:
                    correct = accuracy_score(preds, labels, normalize=False)
                    self.num_correct += correct

                else:
                    for p, l in zip(preds, labels):
                        # get top-k labels with np.argpartition
                        # p = np.argpartition(p, -self.k)[-self.k:]
                        l = l.astype('int32')
                        if l in p:
                            self.num_correct += 1

                self.num_sample += len(labels)

            def reset(self):
                self.num_correct = 0
                self.num_sample = 0

            def result(self):
                if self.num_sample == 0:
                    logger.warning("Sample num during evaluation is 0.")
                    return 0
                return self.num_correct / self.num_sample

        class Dataloader:
            def __init__(self, dataset_location, image_list, batch_size):
                self.batch_size = batch_size
                self.image_list = []
                self.label_list = []
                self.random_crop = False
                self.resize_side= 256
                self.mean_value = [0, 0, 0] #[0.485, 0.456, 0.406]
                self.std_value = [1, 1, 1] #[0.229, 0.224, 0.225]

                self.height = 32
                self.width = 32
                with open(image_list, 'r') as f:
                    for s in f:
                        image_name, label = re.split(r"\s+", s.strip())
                        src = os.path.join(dataset_location, image_name)
                        if not os.path.exists(src):
                            continue

                        self.image_list.append(src)
                        self.label_list.append(int(label))

            def _preprpcess(self, src):
                with Image.open(src) as image:
                    image = np.array(image.convert('RGB'))

                    height, width = image.shape[0], image.shape[1]
                    scale = self.resize_side / width if height > width else self.resize_side / height
                    new_height = int(height*scale)
                    new_width = int(width*scale)
                    image = cv2.resize(image, (new_height, new_width))
                    image = image / 255.
                    shape = image.shape
                    if self.random_crop:
                        y0 = np.random.randint(low=0, high=(shape[0] - self.height +1))
                        x0 = np.random.randint(low=0, high=(shape[1] - self.width +1))
                    else:
                        y0 = (shape[0] - self.height) // 2
                        x0 = (shape[1] - self.width) // 2
                    if len(image.shape) == 2:
                        image = np.array([image])
                        image = np.repeat(image, 3, axis=0)
                        image = image.transpose(1, 2, 0)
                    image = image[y0:y0+self.height, x0:x0+self.width, :]
                    image = ((image - self.mean_value)/self.std_value).astype(np.float32)
                    image = image.transpose((2, 0, 1))
                return image

            def __iter__(self):
                return self._generate_dataloader()

            def _generate_dataloader(self):
                sampler = iter(range(0, len(self.image_list), 1))

                def collate(batch):
                    """Puts each data field into a pd frame with outer dimension batch size"""
                    elem = batch[0]
                    if isinstance(elem, collections.abc.Mapping):
                        return {key: collate([d[key] for d in batch]) for key in elem}
                    elif isinstance(elem, collections.abc.Sequence):
                        batch = zip(*batch)
                        return [collate(samples) for samples in batch]
                    elif isinstance(elem, np.ndarray):
                        try:
                            return np.stack(batch)
                        except:
                            return batch
                    else:
                        return batch

                def batch_sampler():
                    batch = []
                    for idx in sampler:
                        batch.append(idx)
                        if len(batch) == self.batch_size:
                            yield batch
                            batch = []
                    if len(batch) > 0:
                        yield batch

                def fetcher(ids):
                    data = [self._preprpcess(self.image_list[idx]) for idx in ids]
                    label = [self.label_list[idx] for idx in ids]
                    return collate(data), label

                for batched_indices in batch_sampler():
                    try:
                        data = fetcher(batched_indices)
                        yield data
                    except StopIteration:
                        return

        def eval_func(model, dataloader, metric):
            metric.reset()
            sess = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
            input_names = [i.name for i in sess.get_inputs()]
            for input_data, label in dataloader:
                output = sess.run(None, dict(zip(input_names, [input_data])))
                metric.update(output, label)
            return metric.result()

        quant_model_name = model_name.replace('.onnx', '_i8.onnx')

        torch.onnx.export(model, input_data, model_name, opset_version=13)

        model = onnx.load(model_name)
        dataloader = Dataloader(QUANT_DATASET_FOLDER, QUANT_LABEL_TXT, INPUT_SIZE_MOBILENETV2[0])

        def eval(onnx_model):
            return eval_func(onnx_model, dataloader, TopK())
        
        accuracy_criterion = AccuracyCriterion()
        accuracy_criterion.relative = 0.03

        config = PostTrainingQuantConfig(
            quant_format="QOperator", #"QDQ",
            accuracy_criterion=accuracy_criterion)

        q_model = quantization.fit(model, config, calib_dataloader=dataloader,
                    eval_func=eval)

        q_model.save(quant_model_name)

        if os.path.exists('./nc_workspace'):
            shutil.rmtree('./nc_workspace')

    else:
        torch.onnx.export(model, input_data, model_name, opset_version=17)

elif MODEL_TYPE == 'MobileNetV3':
    from mobilenetv3.model import MobileNetV3_1DTCN

    state_dict = torch.load(MODEL_CKPT, weights_only=False)

    num_classes = state_dict["classifier.weight"].shape[0]
    model = MobileNetV3_1DTCN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    model_name = MODEL_CKPT.replace('pt', 'onnx')

    torch.onnx.export(
        model,
        torch.randn(INPUT_SIZE_MOBILENETV3),
        model_name,
        input_names=["pixel"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True
    )

elif MODEL_TYPE == 'VideoMAE':
    assert os.path.isdir(MODEL_CKPT), 'For VideoMAE, the MODEL_CKPT should be a folder, export the model using model.save_pretrained'

    # Replace GELU operator with equivalent function to prevent CPU fallback
    def _gelu_tanh(x):
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    torch.nn.functional.gelu = _gelu_tanh
    torch.nn.GELU.forward = lambda self, x: _gelu_tanh(x)

    from transformers import VideoMAEForVideoClassification

    class VideoMAEWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            pixel_values = pixel_values.unsqueeze(0)
            out = self.model(pixel_values=pixel_values)
            return out.logits

    model = VideoMAEForVideoClassification.from_pretrained(MODEL_CKPT)
    model_name = os.path.join(os.path.dirname(MODEL_CKPT), os.path.basename(os.path.normpath(MODEL_CKPT)) + ".onnx")
    wrapper = VideoMAEWrapper(model).eval()

    torch.onnx.export(
        wrapper,
        torch.randn(INPUT_SIZE_VIDEOMAE),
        model_name,
        input_names=["pixel"],
        output_names=["features"],
        opset_version=18,
        do_constant_folding=True
    )