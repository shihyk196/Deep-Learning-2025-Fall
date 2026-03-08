Lab04 Model Compression: Pruning & Quantization

Dataset: CIFAR10
Model: ResNet20 (pretrained)
Tasks:
Task 1: unstructured pruning
Task 2-1: post training static quantization using Pytorch's FX graph mode
Task 2-2: manual post training static quantization

Task 1: Unstructured Pruning
Use unstructured pruning to prune the pretrained model to achieve >=50% sparsity while keeping test accuracy >=90%.

Task 2-1: Post Training Static Quantization Using Pytorch's FX Graph Mode
Pytorch FX graph mode: Prepare, calibrate, convert
Use at least five different amount of data for calibration andcompare the changes in accuracy of the quantized models on the test set, and document your observations in your report.

Task 2-2: Manually Quantizing the Model
Manually quantize the model.
The quantized model should have test accuracy >= 90.0%.
