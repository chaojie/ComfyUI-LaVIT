# ComfyUI-LaVIT

## workflow

### T2V

[T2V](https://github.com/chaojie/ComfyUI-LaVIT/blob/main/wf_t2v.json)

<img src="wf_t2v.png" raw=true>

### I2V

[I2V](https://github.com/chaojie/ComfyUI-LaVIT/blob/main/wf_i2v.json)

<img src="wf_i2v.png" raw=true>

### T2V Long Video

[T2V Long](https://github.com/chaojie/ComfyUI-LaVIT/blob/main/wf_t2v_long.json)

<img src="wf_t2v_long.png" raw=true>

### 1、Model Weights

```
 huggingface-cli download --resume-download rain1011/Video-LaVIT-v1 --local-dir ~/ComfyUI/models/diffusers/Video-LaVIT-v1 --local-dir-use-symlinks False
```

## [LaVIT](https://github.com/jy0205/LaVIT)