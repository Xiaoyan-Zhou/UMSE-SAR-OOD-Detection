# UMSE-SAR-OOD-Detection
Mitigating SAR Out-of-distribution Overconfidence based on Evidential Uncertainty

### Our paper will be submitted to IEEE Geoscience and Remote Sensing Letters; the trained model weights will be released soon.
## Abstract
Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) is extensively applied in both military and civilian sectors. However, in the open world, test and training data distribution may differ. Therefore, SAR Out-of-Distribution (OOD) detection is important because it enhances the reliability and adaptability of SAR systems. However, most OOD detection models are based on maximum likelihood estimation and overlook the impact of data uncertainty, leading to overconfidence output for both in-distribution (ID) and OOD data. To address this issue, we consider the effect of data uncertainty on prediction probabilities, treating these probabilities as random variables and modeling them using Dirichlet distribution. Building on this, we propose an Evidential Uncertainty aware Mean Squared Error (UMSE) loss function to guide the model in learning highly distinguishable output between ID and OOD data. Furthermore, to comprehensively evaluate OOD detection performance, we have compiled and organized publicly available data, which includes MSTAR, SAMPLE, SAR-ACD, and FUSAR-ship, and constructed a SAR OOD detection dataset that lays the groundwork for future research in this field. Experimental results on our constructed OOD detection dataset demonstrate that the UMSE approach achieves state-of-the-art performance, reducing the average FPR95 by up to 64.8\%

## Usage
### train

```sh
python 0-train.py --loss_option UMSE --model resnet18
```

### test

```sh
python 1-ood_test.py --loss_option UMSE --ood_method MaxLogit
```