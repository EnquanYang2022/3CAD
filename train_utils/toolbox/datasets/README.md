

## Download the datasets in the following link and extract them into corresponding folders.
| Dataset | Official | Our link | Task
|------------|------------------|------------------| ------------------|
| 3CAD    | [Official](https://drive.google.com/file/d/1zhCHL6oH8_IuEkU72F-9bltroiBHmmcH/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1zhCHL6oH8_IuEkU72F-9bltroiBHmmcH/view?usp=sharing) | Anomaly Detection & Localization |
| MVTec-AD    | [official](https://www.mvtec.com/company/research/datasets/mvtec-ad) | [Google Drive](https://drive.google.com/file/d/1qImSm9GFZEag67hJeTNyVhon8hVLnwyO/view?usp=sharing) | Anomaly Detection & Localization |
| DTD    | [official](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | [Google Drive](https://drive.google.com/file/d/171A3_RGjRsLxdqdY4g42Efecj3WzWNjI/view?usp=sharing) | Anomaly Detection & Localization |

## Dataset Directory Structure

After downloading and extracting, ensure that each dataset folder directly contains the category folders. For the convenience of research, the dataset structure of 3CAD is identical to that of MVTec-AD.

#### 3CAD and MVTec-AD Directory Structure
Both the **3CAD** and **MVTec-AD** datasets should be organized with the same folder structure:


```
└── 3CAD
    ├── Aluminum_Camera_Cover
    ├── Aluminum_Ipad
    ├── Aluminum_Middle_Frame
    ├── Aluminum_New_Ipad
    ├── Aluminum_New_Middle_Frame
    ├── Aluminum_Pc
    ├── Copper_Stator
    └── Iron_Stator
```

```
└── MVTec-AD
    ├── bottle
    ├── cable
    ├── capsule
    ├── carpet
    ├── grid
    ├── hazelnut
    ├── leather
    ├── metal_nut
    ├── pill
    ├── screw
    ├── tile
    ├── toothbrush
    ├── transistor
    ├── wood
    └── zipper
```
#### Prepare DTD dataset.
If you use **DTD** dataset for anomaly synthesis, Please download and extract it to your path. Modify the **dtd_path** in [get_dataset](../utils.py) to the path where you extracted it.
After organizing the datasets, each folder should contain the category subfolders. 
