# From Adults to Children: Benchmarking Foundation Models for Chest X-Ray Classification under Domain Shift

**Sara Inácio, Hugo Proença, João C. Neves**  
University of Beira Interior · Instituto de Telecomunicações · NOVA LINCS

This repository accompanies the paper **“From Adults to Children: Benchmarking Foundation Models for Chest X-Ray Classification under Domain Shift.”** It presents a benchmark to study how models trained on **adult chest X-rays** transfer to the **pediatric domain**, where anatomical differences, disease presentation, and population characteristics introduce a substantial domain shift.

> **Code availability:** The training, evaluation, and data-processing scripts will be added to this repository. Installation and reproduction instructions will be updated once the code is released.

## Overview

Foundation models have shown strong potential in medical imaging, but robust performance on adult data does not necessarily translate to pediatric populations. In this work, we benchmark **six foundation models** and **two conventional CNN baselines** under a common **five-label multi-label classification task**, using **adult data only** for downstream training and evaluating performance on both adult and pediatric test domains.

The benchmark is designed to address the following questions:

1. Which models transfer most effectively from adult to pediatric chest X-rays?
2. How do pretraining domain and supervision paradigm relate to cross-domain performance?
3. How does adaptation depth affect pediatric performance and robustness?
4. Does in-domain adult performance reliably reflect cross-domain robustness?

## Benchmark at a Glance

The figures below summarize the domain shift setting and the evaluation workflow used in the paper.

### Adult-to-Pediatric Domain Shift

![Adult-to-pediatric domain shift](figures/domain_shift_overview.png)

*Illustration of the source and target domains considered in this benchmark: VinDr-CXR as the adult source domain and PediCXR as the pediatric target domain.*

### Evaluation Workflow

![Benchmark workflow](figures/benchmark_workflow.png)

*Overview of the benchmark: models are adapted using adult chest X-rays and subsequently evaluated in both the adult in-domain setting and the pediatric cross-domain setting.*

> Replace the image paths above with the final file locations in your repository (for example, inside a `figures/` folder).

## Datasets

We use two chest X-ray datasets:

- **VinDr-CXR**: used as the **adult source domain**.
- **PediCXR**: used as the **pediatric target domain**.

To better separate the two populations, the benchmark includes an additional **age-based screening step** based on available DICOM metadata. Patients aged **0–18 years** are considered pediatric, and patients **older than 18 years** are considered adult. Pediatric examinations identified in the original VinDr-CXR test partition are reassigned to the pediatric evaluation pool, while adult examinations identified in PediCXR are excluded from the pediatric domain.

The two datasets are harmonized into a shared **five-label target space**:

- **No Finding**
- **Other Diseases**
- **Pneumonia**
- **Tuberculosis**
- **Lung Tumor**

In PediCXR, *Pneumonia*, *Broncho-pneumonia*, and *Pleuro-pneumonia* are merged into a single **Pneumonia** label. Labels without a counterpart across datasets are excluded from the shared target space.

For complete details on age-based screening, dataset composition, and label harmonization, please refer to the paper and supplementary material.

> **Dataset access:** This repository does not include the original medical images. Users must obtain the datasets from their official sources and comply with the respective licenses and usage conditions.

## Evaluated Models

The benchmark includes both conventional CNN baselines and foundation models spanning different pretraining domains and supervision strategies.

| Category | Models |
|---|---|
| CNN baselines | DenseNet-121, EfficientNet-B0 |
| General-purpose foundation models | CLIP, DINOv2 |
| Biomedical foundation models | BiomedCLIP |
| CXR-specific / radiology-oriented foundation models | MedCLIP, BioViL-T, RAD-DINO |

For all models, we use the **visual encoder** together with the same downstream classification head. For multimodal models, the **text encoder is not used** during downstream training or inference.

## Adaptation Strategies

We compare three levels of downstream adaptation:

- **Linear Probing (LP):** only the classification head is trained; the encoder remains frozen.
- **Partial Fine-Tuning (PFT):** the classification head and the final encoder block/stage are updated.
- **Full Fine-Tuning (FFT):** the full encoder and classification head are jointly optimized.

These three settings allow us to assess how much adaptation is needed for models to transfer from adult to pediatric chest X-rays.

## Evaluation

All models are trained or adapted **exclusively on adult data** and evaluated in two settings:

- **In-domain evaluation (A → A):** testing on adult chest X-rays.
- **Cross-domain evaluation (A → P):** testing on pediatric chest X-rays.

To quantify the impact of the adult-to-pediatric shift, we also report the **AUROC gap** between adult and pediatric evaluation.

## Main Findings

The benchmark reveals several consistent trends:

- All evaluated models suffer a **clear performance drop** when transferred from adult to pediatric data.
- **RAD-DINO** shows the most stable behavior across adaptation strategies and retains strong pediatric performance.
- The effect of adaptation depth is **model-dependent**: more fine-tuning can improve, preserve, or even reduce cross-domain performance depending on the pretrained representation.
- Strong adult in-domain performance does **not necessarily imply** stronger pediatric transfer.

Overall, the results show that evaluating models only on adult data gives an incomplete view of their robustness across patient populations.

## Paper and Supplementary Material

- **Paper:** *From Adults to Children: Benchmarking Foundation Models for Chest X-Ray Classification under Domain Shift*
- **Supplementary Material:** includes additional details on age-based population screening, dataset construction, splitting, and label composition.

Links to the final publication and supplementary files will be added when available.

## Citation

If you find this benchmark useful in your research, please cite the accompanying paper. The final BibTeX entry will be added once the publication details are available.

## Authors

**Sara Inácio**, **Hugo Proença**, and **João C. Neves**  
University of Beira Interior · Instituto de Telecomunicações · NOVA LINCS
