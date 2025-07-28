# Hierarchical Part-based Generative Model for Realistic 3D Blood Vessel (MICCAI2025)

Welcome to the project! This repository implements a hierarchical part-based generative model focused on realistic 3D blood vessel.(MICCAI2025)
🧠 **[MICCAI 2025 arXiv](https://arxiv.org/pdf/2507.15223)** 
If you find this work helpful, please consider citing:

```bibtex
@article{chen2025hierarchical,
  title={Hierarchical Part-based Generative Model for Realistic 3D Blood Vessel},
  author={Chen, Siqi and Zhang, Guoqing and Lai, Jiahao and Shen, Bingzhi and Zhang, Sihong and Dong, Caixia and Chen, Xuejin and Li, Yang},
  journal={arXiv preprint arXiv:2507.15223},
  year={2025},
  url={https://arxiv.org/abs/2507.15223v1}
}
```

## Dependencies 📦

Install the required packages using:
    
    pip install -r requirements.txt
## Dataset 📊

You can download the three datasets from the following links:

1. **ImageCAS:** [GitHub - ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)

2. **Processed CoW:** [GitHub - vessel_diffuse](https://github.com/chinmay5/vessel_diffuse)

3. **Vascusynth:** [Vascusynth Data](https://vascusynth.cs.sfu.ca/Data.html)


## Usage ⚙️
Data Preprocessing:

    /data_preprocess
In the First Stage:

    python train_tree.py
In the Second Stage:

    python train_tree.py

In the  Third Stage:

     python assembly/aseemble.py

