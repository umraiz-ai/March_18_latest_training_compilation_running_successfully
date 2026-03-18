these weights are trained on the 14 classes. 
and with these parameters in the compile script

PRE_PARAM_DIR = "runs/train/training2/weights"

OPT_PARAM_DIR = "runs/train/training3/weights"


ONNX_INPUT_SHAPE = "1 3 416 416"    # (B,C,H,W)
TACHY_INPUT_SHAPE = "1 256 416 3"   # (B,H,W,C)


