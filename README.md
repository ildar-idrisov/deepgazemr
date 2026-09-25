# DeepGaze MR

This repository provides the video saliency model *DeepGaze MR* and the
meta-benchmark proposed in the following paper:

```
@InProceedings{tangemann2020,
  author = {Matthias Tangemann and Matthias Kümmerer and Thomas S.A. Wallis and Matthias Bethge},
  title = {Measuring the Importance of Temporal Features in Video Saliency},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month = {August},
  year = {2020}
}
```


## Usage
*DeepGaze MR* is available on PyTorch Hub. It can be easily used without having
to clone this repository. However, you have to make sure that all dependencies
listed in requirements.txt are satisfied.

```python
model = torch.hub.load('mtangemann/deepgazemr', 'DeepGazeMR', pretrained=True)
model.to(device)
```

When loading the model with `pretrained=True` (the default), you will get the
model with the weights and center bias from the LEDOV dataset. This is exactly
the model which has been used in the paper.

Input videos are expected as float tensors of shape *T x C x H x W*
in the range *[0.0,1.0]*. *DeepGaze MR* takes care of correctly normalizing the
features for the VGG network. The following example shows how to preprocess
a video when using [scikit-video](http://www.scikit-video.org):

```python
video = skvideo.io.vread('file.mp4')
video = torch.from_numpy(video).type(torch.float32)
video = video.permute(0, 3, 1, 2) / 255.0
video = video.to(device)
```

There are two ways how to use *DeepGaze MR* for predicting human gaze. The
`forward` method expects a single clip of 16 frames (*T x C x H x W*; batches
of clips are not supported) and returns the predicted **log-density** (natural
log) of human gaze on the last frame in the window. The prediction is
normalized, i.e. `prediction.exp()` is a probability distribution that sums to
1:

```python
clip = video[0:16]
log_density = model.forward(clip)
print(log_density.shape)  # e.g. [360, 640], matching the input resolution
density = log_density.exp()  # probability distribution over pixels
```

The `predict` method is used to predict gaze for full videos. This method is
optimized to not compute features for the same frame multiple times when
shifting the window. So it is much faster than naively using `forward` for all
windows. The `predict` method returns an iterator over all predictions
(log-densities, as for `forward`) for the input video. Due to the windowed approach, the predictions for the first 15
frames will be `None`.

```python
for i, prediction in enumerate(model.predict(video)):
  if prediction is not None:
    # do something with the prediction for frame i
```

The model is not scale invariant: the final gaussian blur has a fixed size of
about 16 pixels of the output (= input) resolution and the VGG features depend
on the input resolution as well. Predictions for the same video therefore
change with the resolution it is passed in. The bundled LEDOV center bias has
a resolution of 720 x 1280 and is rescaled to the input resolution.

When transferring *DeepGaze MR* to different datasets than LEDOV, you have to
provide the correct center bias for that dataset (even when using the
pretrained model). The center bias is expected to be a probability distribution
of shape *H x W*. If the resolution of the center bias does not match the
resolution of the input video, *DeepGaze MR* will scale it accordingly.

```python
center_bias = torch.load(...)  # normalized tensor of shape HxW

model = torch.hub.load('mtangemann/deepgazemr', 'DeepGazeMR', pretrained=True, center_bias=center_bias)
model.to(device)
```

Alternatively, the custom center bias can also be passed to the `forward` and
`predict` methods:

```python
prediction = model.forward(clip, center_bias=center_bias)

# or

iterator = model.predict(video, center_bias=center_bias)
```


## Tests
The tests run with `python -m pytest tests` (requires pytest) and use the
bundled checkpoint, no download needed.


## Meta-Benchmark
*DeepGaze MR* was used to create a meta benchmark based on the LEDOV and DIEM
datasets. The meta benchmark consists of those frames where at least 1bit of
explainable information was not explained by *DeepGaze MR*. The files in the
`meta-benchmark` directory contain the frame indices of the meta benchmark as
described in the paper.

Each CSV file contains 3 columns named `video`, `frame` and `mask` for the
video name, the frame number and whether the respective frame is included in
the meta-benchmark (True=included, False=not included). Please note that we
resampled the videos in the LEDOV dataset to 30Hz in our study. The given
frame numbers refer to the resampled videos.

To evaluate your model on this meta-benchmark, please first apply it to the
full videos of the respective dataset. Then calculate the performances using
regular image saliency metrics for those frames contained in the meta-benchmark.
Final results are then obtained by first averaging the scores over frames,
and then over videos.


## Contact
If you have any questions, please contact matthias.tangemann@bethgelab.org or
create an issue on GitHub.


## License
Licensed under the [MIT License](LICENSE).

If you use this model in your work, please cite the original paper mentioned
above.
