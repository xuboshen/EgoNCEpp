Download possible datasets ([Ego4D](https://ego4d-data.org/docs/start-here/), [Epic-Kitchens-100](https://academictorrents.com/details/d08f4591d1865bbe3436d1eb25ed55aae8b8f043), [EGTEA](https://cbs.ic.gatech.edu/fpv/), [CharadesEgo](https://prior.allenai.org/projects/charades-ego)) from their official websites and organize them as follows:
```
|–– BASE_PATH/
    |-- ego4d/down_scale/
        |-- 0a02a1ed-a327-4753-b270-e95298984b96.mp4
            |-- 0.mp4
            |-- 300.mp4
            |-- ...
    |-- CharadesEgo/
        |-- 0BIAJEGO.mp4
        |-- ...
    |-- egtea/
        |-- cropped_clips/cropped_clips/
            |-- OP01-R01-PastaSalad\
                |-- OP01-R01-PastaSalad-87815-90765-F002100-F002186.mp4
                |-- ...
    |-- EK100_256p/
        |-- P01/
            |-- P01_01/
                |-- frame_0000000001.jpg
                    |-- ...
            |-- P01_01.MP4
            |-- ...

```

## Ego4D
We chunk each video in Ego4D into 5-minute-long chunks and resize the size into 288×(>=288) following LaViLa.
Here we provide a multi-processing version for fastser processing.

1. Download [Ego4D videos](https://ego4d-data.org/docs/start-here/).

2. Process long videos into shorter chunks.

- For single-process scripts, see [single-process-chunk](../scripts/single_crop_and_resize_ego4d.sh).
- For parallel-processing scripts, see [parallel-process-chunk](../scripts/parallel_crop_and_resize_ego4d.sh)
