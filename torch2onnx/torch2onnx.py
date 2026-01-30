# Conda Environment : Model 
# paddleocr : paddleocr
# torchreid : osnet / p2pnet / apgcc
# yolo_rknn : yolov8 / yolo26
# mmdeploy  : actionmm2

import os
import sys
import math
import onnx
import torch
import shutil
import tempfile
import subprocess

from pathlib import Path

# General Config
MODEL_CKPT = '/home/paiworker1/KX/.Workspace/.Projects/yolov8n-pose.pt'
MODEL_TYPE = 'YOLOv8'; assert MODEL_TYPE in ['ActionMM2', 
                                                'YOLOv8',
                                                'YOLO26',
                                                'YOLOE', # deprecated
                                                'OSNet', 
                                                'PaddleOCR', 
                                                'P2PNet',
                                                'APGCC']

# Only for Yolov8/Yolo26
TASK = 'POS'; assert TASK in ['CLS', 'DET', 'SEG', 'POS'] # Classification, Detection, Segmentation, Pose-Estimation
IMGSZ = (640, 640)

# Only for YoloE
CLASSES = ["person", "bus"]

# Only for ActionMM2 model convertion
CONVERT_TOOL = '/home/paiworker1/KX/.Workspace/.Tools/mmdeploy/tools/deploy.py'
DEPLOY_CONFIG = '/home/paiworker1/KX/.Workspace/.Tools/mmdeploy/configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py'
MODEL_CONFIG = '/home/paiworker1/KX/.Workspace/.Tools/mmaction2/configs/recognition/tsm/v3.py'
TEST_INPUT = '/home/paiworker1/KX/.Workspace/.Projects/action_recognization/ActionMM2/input_for_v2.mp4' # random input to verity the output, if your model input video, then this is a path to a random video, etc.

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

# Only for P2PNet
BACKBONE = 'vgg16_bn'
BACKBONE_CKPT = '/home/paiworker1/KX/.Workspace/.Projects/crowd_count/P2PNet/weights/vgg16_bn-6c64b313.pth'
ROW, LINE = 2, 2
INPUT_SIZE_P2P = [6, 3, 1024, 1280]

# Only for APGCC
BACKBONE_CKPT = '/home/paiworker1/KX/.Workspace/.Projects/crowd_count/APGCC-ONNX/weights/vgg16_bn-6c64b313.pth'
INPUT_SIZE_APGCC = [6, 3, 640, 800]

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
            YOLO(MODEL_CKPT).export(format="onnx", imgsz=IMGSZ, opset=19, simplify=False)
        else:
            YOLO(MODEL_CKPT).export(format="onnx", opset=19, simplify=False)
        
elif MODEL_TYPE == 'YOLOE':
    from ultralytics import YOLOE
    model = YOLOE(MODEL_CKPT)
    model.set_classes(CLASSES, model.get_text_pe(CLASSES))
    if IMGSZ is not None:
        model.export(format="onnx", imgsz=IMGSZ, opset=19)
    else:
        model.export(format="onnx", opset=19)

elif MODEL_TYPE == 'ActionMM2':

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
