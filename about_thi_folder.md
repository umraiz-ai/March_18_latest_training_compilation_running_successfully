In march I am creating this folder for a training from the scratch and use exactly the compilation pipeline which tachyrt suggsted. 
I made a conda environment named umz_second
All the packages details are provided in the file 
/home/contil/umraiz/yolov9_march/requirements_umz_second.txt
Activate the conda environment
conda activate umz_second


After you clone this folder 
run like this 
Training 1: 
torchrun --nproc_per_node=4 --master_port=29500 train.py   --workers 16   --device 0,1,2,3   --batch 64   --data /srv/DATA/DATASETS/NIPA_Data_2025_v9/data.yaml   --img 416   --cfg models/deeper-i/bsnet-t.yaml   --weights ''   --name bsnet-t   --hyp hyp.scratch.yaml   --min-items 0   --epochs 300   --close-mosaic 3   --optimize SGD

The model will be saved as a .pt file 

Training 2:
torchrun --nproc_per_node=4 --master_port=29500 train.py   --workers 16   --device 0,1,2,3   --batch 64   --data /srv/DATA/DATASETS/NIPA_Data_2025_v9/data.yaml   --img 416   --cfg models/deeper-i/bsnet-t-o.yaml   --weights /home/contil/umraiz/yolov9_march/runs/train/training2/weights/best.pt   --name t
raining   --hyp hyp.optimize.yaml   --min-items 0   --epochs 150   --close-mosaic 3  --optimize SGD

Another model will be saved as .pt file



Now to run this compile script after cloning 
run like this (Only once after cloning and not after that)

cp ./models ./models_compile -rf 
cp ./models_compile/yolo_compile.py ./models_compile/yolo.py -f
cd ./TACHY-Compiler/platform_converter/utils/yolov9/
ln -s ../../../../utils utils
ln -s ../../../../models_compile models


then run this for once
chmod +x TACHY-Compiler/compiler/utils/block_4bit.out

This version of the compile
B_DEFAULT_PAD_MODE="--default_pad_mode=dynamic"
was the actual cause of the fixed bounding boxes. 
Now it is solved in this version. 

Now that we have two .pt files, give their path in the python file compile in the same main folder 
/home/contil/umraiz/yolov9_march/compile

Run this file as 
Python compile 

This will save the .tachyrt file with the folder name as the current date and time. There must be .tachyrt file. 
This compile and its helping scripts are provided by the deeper-I . they use 
Remember that Deeper-I uses XWN for quantization, not the standard int4 or int8. 