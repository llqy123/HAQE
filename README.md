
<h2 align="center">
Hierarchy-Aware Quaternion Embedding for Knowledge Graph Completion
</h2>

<p align="center">
    <img src="https://img.shields.io/badge/version-1.0.1-blue">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
    <a href="https://2024.ieeewcci.org/"><img src="https://img.shields.io/badge/IJCNN-2024-%23bd9f65?labelColor=4aaaf1&color=4aaaf1"></a>
</p>

This repository is the official implementation of ["Hierarchy-Aware Quaternion Embedding for Knowledge Graph Completion"](https://ieeexplore.ieee.org/document/10650007) accepted by IJCNN 2024.

<!-- Run Locally -->
### :running: Reproduce the Results
```
    python datasets/process.py
    bash train_model/train_xx.sh
```
    
## Citation
    @INPROCEEDINGS{10650007,
        author={Liang, Qiuyu and Wang, Weihua and Yu, Jie and Bao, Feilong},
        booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
        title={Hierarchy-Aware Quaternion Embedding for Knowledge Graph Completion}, 
        year={2024},
        volume={},
        number={},
        pages={1-8},
        keywords={Analytical models;Quaternions;Neural networks;Knowledge graphs;Tail;Benchmark testing;Mathematical models;hyperbolic space;knowledge graph completion;quaternion space;rigid body transformation},
        doi={10.1109/IJCNN60899.2024.10650007}
    }


## Acknowledgement
Some of the code was forked from the original KGEmb implementation which can be found at [KGEmb](https://github.com/HazyResearch/KGEmb), thank for the excellent source code.





