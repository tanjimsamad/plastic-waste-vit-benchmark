# Dataset

The images are **not included** in this repository. Add your own copy using the layout below
(this is what `torchvision.datasets.ImageFolder` expects, and what the fine-tuning sections read):

```
data/
├── Background/     # 28 images    – empty scene, no object
├── Non_Plastic/    # 1,737 images – non-plastic objects
└── Plastic/        # 2,242 images – plastic waste / plastic objects
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Total: 4,007 images.

## CLIP zero-shot section

The first CLIP cell reads `Background` and `Plastic` straight from zip archives and `Non_Plastic`
from an already-extracted folder:

```
data/images_of_Background.zip
data/images_of_Plastic.zip
data/Non_Plastic/            # extracted from Non_Plastic.rar
```

If you already have the three extracted class folders, you can skip the zips and run the CLIP
cell on the folders instead. It writes `data/clip_inference_results.csv` (git-ignored).

## Source and licence

https://drive.google.com/drive/folders/1cJbOuRxLnhbwdAIW2exl6naP_LDNZ3bV?usp=sharing
