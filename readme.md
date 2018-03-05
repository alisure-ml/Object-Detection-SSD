## SSD


### Paper

[SSD: Single Shot MultiBox Detector](/paper/SSD%20-%20Single%20Shot%20MultiBox%20Detector.pdf)


### checkpoint

download [ssd_300_vgg.ckpt.zip](https://github.com/balancap/SSD-Tensorflow/tree/master/checkpoints)
unzip in `checkpoints`.


### Run

just run `RunnerSSDOneOrRealTime.py`

* run one image
```python
if __name__ == '__main__':
    runner = RunnerOneOrRealTime(ckpt_filename='checkpoints/ssd_300_vgg.ckpt')
    runner.run(image_name="demo/dog.jpg",  result_name="demo/dog_result.png")
```

* run camera
```python
if __name__ == '__main__':
    runner = RunnerOneOrRealTime(ckpt_filename='checkpoints/ssd_300_vgg.ckpt')
    runner.run(prop_id=0, size=(960, 840))
```

* run video
```python
if __name__ == '__main__':
    runner = RunnerOneOrRealTime(ckpt_filename='checkpoints/ssd_300_vgg.ckpt')
    runner.run(prop_id="demo/video1.mp4")
```


### Result

| 图片 | 结果 |
| --- | --- |
| ![car](demo/car.jpg) | ![car](demo/car_result.png) |
| ![car](demo/dog.jpg) | ![car](demo/dog_result.png) |


### Reference

* [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)

