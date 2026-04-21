import warnings
warnings.filterwarnings('ignore')
import torch, glob, tqdm
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info

if __name__ == '__main__':
    flops_dict = {}
    yaml_base_path = 'ultralytics/cfg/models/rt-detr'
    for yaml_path in tqdm.tqdm(glob.glob(f'{yaml_base_path}/*.yaml')):
        if 'DCN' in yaml_path:
            continue
        try:
            model = RTDETR(yaml_path)
            model.fuse()
            n_l, n_p, n_g, flops = model_info(model.model)
            flops_dict[yaml_path] = [flops, n_p]
        except:
            continue
    
    sorted_items = sorted(flops_dict.items(), key=lambda x: x[1][0])
    for key, value in sorted_items:
        print(f"{key}: {value[0]:.2f} GFLOPs {value[1]:,} Params")