1. Concept
* Transfer  : Transfer model to different format on same platform (ex. *.ckpt -> *.pb, *.pth -> *.pt, ...)
* Convert   : Cross between different platforms (ex. *.pb -> *.onnx, ...)
* Deploy    : Model transformation before and after optimization for compilation (ex. opt.* -> eval.* & comp.*)

2. Command
* deploy_model		: Depoly Model (Tensorflow, PyTorch)
* convert_pb2onnx	: Convert *.pb -> *.onnx
* convert_pt2onnx	: Convert *.pt -> *.onnx
* convertall_pytorch	: Deploy & Convert All For PyTorch
* convertall_tensorflow	: Deploy & Convert All For Tensorflow
