# PIANO (Physical Invariant Attention Neural Operator)

This repository contains the code for the paper
- **Deciphering and integrating invariants for neural operator learning with various physical mechanisms**

**TLDR:** PIANO: a new operator learning framework that deciphers and incorporates invariants from the PDE series via self-supervised learning and attention technique, achieving superior performance in scenarios with various physical mechanisms.

## Abstract

Neural operators have been explored as surrogate models for simulating physical systems to overcome the limitations of traditional partial differential equation (PDE) solvers. However, most existing operator learning methods assume that the data originate from a single physical mechanism, limiting their applicability and performance in more realistic scenarios. To this end, we propose Physical Invariant Attention Neural Operator (PIANO) to decipher and integrate the physical invariants (PI) for operator learning from the PDE series with various physical mechanisms. PIANO employs self-supervised learning to extract physical knowledge and attention mechanisms to integrate them into dynamic convolutional layers. Compared to existing techniques, PIANO can reduce the relative error by 13.6\%-82.2\% on PDE forecasting tasks across varying coefficients, forces, or boundary conditions. Additionally, varied downstream tasks reveal that the PI embeddings deciphered by PIANO align well with the underlying invariants in the PDE systems, verifying the physical significance of PIANO.

## Methodology
![image](https://github.com/optray/PIANO/assets/42396587/6bc83555-9f94-44e4-9f2f-e8d64903e283)

(a). Given the PDE initial fields, PIANO first infers the physical invariant (PI) embedding via the PI encoder, then integrates it into the neural operator to obtain a personalized operator. After that, PIANO predicts the subsequent PDE fields with this personalized operator. (b). Training the PI encoder via self-supervised learning methods. (c). Integrate the PI representation with the dynamic convolution technique.

## Quick Start
For each task, you can run the following three main files sequentially:
1. Generate the train/test/validation data (generate_data.py in the data folder)
2. Train the PI encoder (main.py in the pretrain folder)
3. Train the neural operator (main.py in the train folder)

## Citation

This paper has been accepted by the _National Science Review_. If you find our work useful in your research, please consider citing:
```
@article{10.1093/nsr/nwad336,
    author = {Zhang, Rui and Meng, Qi and Ma, Zhi-Ming},
    title = "{Deciphering and integrating invariants for neural operator learning with various physical mechanisms}",
    journal = {National Science Review},
    pages = {nwad336},
    year = {2023},
    month = {12},
    issn = {2095-5138},
    doi = {10.1093/nsr/nwad336},
    url = {https://doi.org/10.1093/nsr/nwad336},
    eprint = {https://academic.oup.com/nsr/advance-article-pdf/doi/10.1093/nsr/nwad336/54922405/nwad336.pdf},
}

```

If you have any questions, please feel free to contact me via: rayzhang@amss.ac.cn
