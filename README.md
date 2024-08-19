# UMSE-SAR-OOD-Detection
## Mitigating SAR Out-of-distribution Overconfidence based on Evidential Uncertainty

## Abstract
Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) is extensively applied in both military and civilian sectors. Nevertheless, test and training data distribution may differ in the open world. Therefore, SAR Out-of-Distribution (OOD) detection is important because it enhances the reliability and adaptability of SAR systems. However, most OOD detection models are based on maximum likelihood estimation and overlook the impact of data uncertainty, leading to overconfidence output for both in-distribution (ID) and OOD data. To address this issue, we consider the effect of data uncertainty on prediction probabilities, treating these probabilities as random variables and modeling them using Dirichlet distribution. Building on this, we propose an Evidential Uncertainty aware Mean Squared Error (UMSE) loss function to guide the model in learning highly distinguishable output between ID and OOD data. Furthermore, to comprehensively evaluate OOD detection performance, we have compiled and organized some publicly available data and constructed a new SAR OOD detection dataset named SAR-OOD. Experimental results on SAR-OOD demonstrate that the UMSE approach achieves state-of-the-art performance. 

## Usage
### train

```sh
python 0-train.py --loss_option UMSE --model resnet18
```

### test

```sh
python 1-ood_test.py --loss_option UMSE --ood_method MaxLogit
```

## SOME THINGS NEED TO PAY ATTENTION
The distribution results in our paper are from results2.rar, and you can get the results by modify 1-ood_test.py as follows:
plot_distribution([score_id, score_sample, score_airplane, score_ship], ['MSTAR', 'SAMPLE', 'SAR-ACD', 'FUSAR-ship'], savepath=os.path.join('./results2/', opt.fig_name + '_'+ opt.ood_method+'.png'))

In draw_reults.py, since we use Seaborn's kdeplot to draw the density plot, the x-axis range might extend beyond 1 even though all data values are less than or equal to 1. This occurs because the kernel density estimation (KDE) can generate small tails near the data boundaries, attempting to smooth the overall distribution. Therefore, use plt.xlim(0, 1) in the function of plot_distribution when you use MSP method and want to show the distribution result.


