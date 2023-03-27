# CVPR2023 “Masked and Adaptive Transformer for Exemplar Based Image Translation" (MATEBIT) 

## Abstract

​	We present a novel framework for exemplar based image translation. Recent advanced methods for this task mainly focus on establishing cross-domain semantic correspondence, which sequentially dominates image generation in the manner of local style control. Unfortunately, cross-domain semantic matching is challenging; and matching errors ultimately degrade the quality of generated images. To overcome this challenge, we improve the accuracy of matching on the one hand, and diminish the role of matching in image generation on the other hand. To achieve the former, we propose a masked and adaptive transformer (MAT) for learning accurate cross-domain correspondence, and executing context-aware feature augmentation. To achieve the latter, we use source features of the input and global style codes of the exemplar, as supplementary information, for decoding an image. Besides, we devise a novel contrastive style learning method, for acquire quality-discriminative style representations, which in turn benefit high-quality image generation. 


## Sample Results

- **same results:**

![localFace1](https://github.com/AiArt-HDU/MATEBIT/blob/main/images/Metfaces.png)

![localFace2](https://github.com/AiArt-HDU/MATEBIT/blob/main/images/celeba.png)

![localFace3](https://github.com/AiArt-HDU/MATEBIT/blob/main/images/aahq.png)

![localFace4](https://github.com/AiArt-HDU/MATEBIT/blob/main/images/dishini.png)

![localFace5](https://github.com/AiArt-HDU/MATEBIT/blob/main/images/uk.png)

![localFace6](https://github.com/AiArt-HDU/MATEBIT/blob/main/images/deep.png)

- **More Results:**

We offer more results here: https://drive.google.com/drive/folders/1t2U82eDvqd1-xKigBs_GnnVRemdoYOpR
