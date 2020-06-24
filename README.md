# Infer on Videos 

This little repo is about deep learning video inference. It aims at plotting the prediction of a classification model on a video.

It uses mobileNetV2 model from https://pytorch.org/hub/pytorch_vision_mobilenet_v2 trained on ImageNet dataset but you can easily custom it with your own model.
### requirements
You need to have python installed with the following library:
- `torch`
- `torchvision`
- `numpy`
- `PILLOW`
- `matplotlib`
- `open-cv-pythoon`

### Infer on video with Mobilenet!

```
python plot_mobilenet_predictions_video
```

You can add parameters like 
- `--input = path_to_input_video` If you don't provide it, it will propose you to infer on one of the video in the current directory.
- `--out = path_to_otput_video` If you don't provide it, it defaults to 'output_video.avi'
- `--step= step_of_inference` default is 10: It infers once every 10 frames ~ 3 times per second

### Comments
CLASSES in the file [classes.json](classes.json) are the one of ImageNet. You can also custom it and put your own classes.
