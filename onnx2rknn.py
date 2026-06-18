# conda activate rknn_vlm && python onnx2rknn.py

import os
from rknn.api import RKNN
from itertools import product

# Quantization
DO_QUANT = False
DATASET_PATH = '/home/paiworker1/KX/.Workspace/.Projects/pp_lcnetv2/calib.txt'
QUANTIZED_HYBRID_LEVEL = 0
# if observed output range is not good during quantization, try to set quantized_hybrid_level=1 or 2
# the higher the level, the less weight being quantized, accuracy preserved but inference speed sacrificed
# currently only observed mobilenetv2 need to set this to 1 or 2, other models can keep it to 0
# PPLCNetV1-Pedestrian / PPLCNetV2-Reid need to set this to 1

# Convertion Settings
PLATFORM = 'rk3588'; assert PLATFORM in ['rk3588', 'rv1126b', 'rv1106']
ONNX_PATH = '/home/paiworker1/KX/.Workspace/.Projects/videomae/videomae_small.onnx'
RKNN_PATH = f'{os.path.splitext(ONNX_PATH)[0]}{"_u8" if DO_QUANT else "_fp16"}_{PLATFORM}.rknn'

ENABLE_DYNAMIC_INPUT = False
DYNAMIC_INPUT = [
    [1, 20],
    80,
    160,
    3
]

def generate_dynamic_input(DYNAMIC_INPUT):
    """
    Generate RKNN-compatible dynamic_input list from DYNAMIC_INPUT.
    Each element in DYNAMIC_INPUT can be:
        - int: fixed value or batch range
        - list [min, max]: dynamic range
    Returns a list of [[shape]] for RKNN.
    """
    ranges = []
    for dim in DYNAMIC_INPUT:
        if isinstance(dim, list) and len(dim) == 2:
            # create range from min to max (inclusive)
            ranges.append(range(dim[0], dim[1]+1))
        elif isinstance(dim, int):
            # single value range
            ranges.append([dim])
        else:
            raise ValueError(f"Invalid DYNAMIC_INPUT element: {dim}")

    # create all combinations
    dynamic_input = [[[ *shape ] ] for shape in product(*ranges)]
    return dynamic_input

# If don't apply normalization
MEAN = None
STD = None

# If apply typical 0-1 normalization
# For Yolo or MobileNetV2
# MEAN = [[0, 0, 0]]
# STD = [[255, 255, 255]]

# Normalization for PaddleOCR Text Orientation Classification layer (PP-LCNet_x1_0_textline_ori)
# Normalization for PaddleOCR Doc Orientation Preprocessing layer (PP-LCNet_x1_0_doc_ori)
# Normalization for OSNet
# Normalization for P2PNet / APGCC
# Normalization for PPLCNetV1-Pedestrian / PPLCNetV2-Reid
# MEAN = [[0.485*255, 0.456*255, 0.406*255]]
# STD = [[0.229*255, 0.224*255, 0.225*255]]

# Normalization for PaddleOCR Text Recognization layer (en_PP-OCRv5_mobile_rec)
# MEAN = [[0.5*255,0.5*255, 0.5*255]]
# STD = [[0.5*255, 0.5*255, 0.5*255]]

# Normalization for PaddleOCR Text Detection layer (PP-OCRv5_server_det)
# Normalization for MMPose (RTMPose3D)
# MEAN = [[123.675, 116.28, 103.53]]
# STD = [[58.395, 57.12, 57.375]]

# For others
DISABLE_RULES = ['reduce_reshape_op_around_split']

# For Sam3 Decoder
# DISABLE_RULES = ['swap_cast_reshape', 'reduce_tp_in_mesh_forward', 'swap_expand_transpose']

# For Sam3 Image Encoder
# DISABLE_RULES = ['convert_rs_add_rs_to_rs_gather_elements']

if __name__ == '__main__':

    rknn = RKNN(verbose=False)

    dynamic_input = None
    if ENABLE_DYNAMIC_INPUT:
        dynamic_input = generate_dynamic_input(DYNAMIC_INPUT)

    rknn.config(mean_values=MEAN, 
                std_values=STD, 
                target_platform=PLATFORM, 
                dynamic_input=dynamic_input, 
                disable_rules=DISABLE_RULES,
                quantized_hybrid_level=QUANTIZED_HYBRID_LEVEL
                )
    ret = rknn.load_onnx(model=ONNX_PATH)
    assert ret == 0, f'ret = {ret}, Load model failed!'

    ret = rknn.build(do_quantization=DO_QUANT, dataset=DATASET_PATH)
    assert ret == 0, f'ret = {ret}, Build model failed!'

    ret = rknn.export_rknn(RKNN_PATH)
    assert ret == 0, f'ret = {ret}, Export rknn model failed!'

    rknn.release()