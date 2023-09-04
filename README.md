# RSFNet
The implementation for 'Region selective fusion network for robust RGBT tracking'


The pretrained backbone weights: [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)

Our pretrained weights for testing:

link：https://pan.baidu.com/s/1YdlXZNYUzLV752vLGxIxlg 
password：vsia`

testing RGBT234 examples:
```
python tracking/test.py rsfnet vitb_ep180 --dataset rgbt234 --threads 0 --num_gpus 1

```


## Acknowledgments

* The code is built based on [PyTracking](https://github.com/visionml/pytracking) library and some codes are borrowed from STARK and OSTRACK.
* Thanks for the [STARK](https://github.com/researchmm/Stark), Ostrack (https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library. 
>>>>>>> 6aa9f8f937d768ba7bcd2227ca35691c60407ef0



