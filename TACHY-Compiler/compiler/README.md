[DPI] TACHY-Compiler
======================================

# 1. File Handling
# 1.1. Command
* compile		: compile_frontend + compile_backend
* compile_frontend	: *.onnx -> *.tachy with optimization
* compile_backend	: Generate Instruction and Parameter for NPU
* convert_tachy		: *.onnx -> *.tachy for evaluation (without optimization)
* gen_prototxt		: Generate *.prototxt from *.onnx (*.onnx -> *.prototxt)

# 2. Examples
## 2.1. inference : Deep-learning application example
### 2.1.1. od
* Object Detection without pre and post processing
### 2.1.2. synergy
* Arrhythmia Classification without pre and post processing for SYAI 

## 2.2. extract : Extract information about target model
### 2.2.1. weights
* Extract weights of model


# 3. Modification & Verification
## 3.1. tachy_inference.py
### 3.1.1. predict
### 3.1.2. debug
### 3.1.3. dump

## 3.2. tachy_extract.py
### 3.2.1. weights

## 3.3. tachy_convert.py
### 3.3.1. translation_pu
### 3.3.2. split_concat
### 3.3.3. split_conv

## 3.4. tachy_prototxt.py
