1. models/common.py -> common_dpi.py: Update layer components
2-1. DFL off: utils/loss_tal.py:108 
    def __init__(self, model, use_dfl=True): -> def __init__(self, model, use_dfl=False):
2-2. DFL off: models/yolo.py:90 
    self.reg_max = 16 -> self.reg_max = 1
2-3. DFL off: utils/metrics.py:277 # Add line
    iou = torch.nan_to_num(iou, nan=0.0) # if use_dfl is false
3. models/detect/gelan-t-dpi.yaml: Run
###############################################
1-1. models/common.py -> add ConvRes
1-2. models/yolo.py:751~753 -> add ConvRes forward 
2-1. models/common.py:22~23 -> import ddesinger_api
2-2. nn.Conv2d -> cnn.Conv2d
2-3. nn.ConvTranspose2d -> cnn.ConvTranspose2d
2-4. transform, pruning option integration
3. models/detect/bsnet-t.yaml: Run
###############################################
1. Add Dropout layer
2. Add bsnet-t-o.yaml
3. Integrate XWN argument
4. Add LAMB optimizer 
5. Display LR
6. Block fusing layers
7. add option in export.py:113 => training=torch.onnx.TrainingMode.TRAINING,
################################################
20250425
1. train.py
	492:
        	# last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        	last = Path(opt.weights if len(opt.weights) > 0 else get_latest_run())
	496:
    		workers = opt.workers 
	508:
        	opt.workers = workers
	

