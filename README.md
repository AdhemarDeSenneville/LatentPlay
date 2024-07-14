
# Latent Play - VST

## -> [**Youtube Demo**](./code/Experiments_WaveShaper.ipynb) <-

[**Overview**](./code/Experiments_WaveShaper.ipynb)
| [**Tutorial**](#Tutorial)
| [**Presentation**](DDSP_Presentation.pptx)

## Presentation

Latent Play is a demo VST pluging for sample generation. It uses neural synthesise to allow infinit comtrol over your sample packs. This repository includes [Kaggle](https://www.kaggle.com/code/adhmardesenneville/latentplay/output?scriptVersionId=188128696) code using [Pytorch Lightning ](https://lightning.ai/docs/pytorch/stable/) and [WandB](https://wandb.ai/site) for the training pippline, and [Pyqt5](https://pypi.org/project/PyQt5/) for the GUI.


![](./fig/VST_view.png)

### How does it works
- 1) Train a model on on / several of your favorite sample packs. 
- 2) Load the sample of your choise
- Play with the PCA and Custom controls
- Select the sample of your choise


# Tutorial

### Dataset
```
Dataset
├── Pack 1
│   ├── Audio_1.wav
│   └── Audio_e.wav
├── Pack 2
│   └── SubPack 1
│       └── Audio_r.wav
│       └── Audio_X.wav
│   └── SubPack 2
│       └── Audio_F.wav
├── Pack 3
...
```
  
Having your sample separated in different pack allow clear and where each pack is sample in the latent space

###
In the code you can change, sample duration, auto encoder duration
model deep, training hardware, training time ect...


### Training
Launch the script and monitor the training via WandB API. You can have accest also to audio sample generated by your model during training to monitor audio quality.

![](./fig/training_loss.png)
![](./fig/training_audio.png)
![](./fig/training_spect.png)

### Run the VST
Put the output file from the run in the models folder
Put your dataset in the Dataset folder

```
python run GUI/main.py
```


# Implementation details

### Latent Space Modification

this part of the project has been puporsly done without any look at the literatur nor popular technics to alow mor creativ thinking 
This could be the case that thos technics are either trivial or inexistant in the literatur

### Latent Space Control

Let $z$ represent the latent space. The objective is to find an application in this latent space z -> z' in a meaningful way. We aim to achieve specific modifications to control 5 parameters

1. $\lambda'_1 = \langle z', w_{(1)} \rangle$ : The projection of $ z $ in the Frirst Principal conponents

2. $\lambda'_2 = \langle z', w_{(2)} \rangle$ : The projection of $z$ in the Frirst Principal conponents

3. $f' = \langle z',\theta_{\text{f}} \rangle$ : Estimated Audio Frequency from Latent Space

4. $\alpha' = \langle z',\theta_{\alpha} \rangle$ : Estimated Audio Attack from Latent Space

5. $\beta' = \langle z',\theta_{\beta} \rangle$ : Estimated Audio Release from Latent Space



Those constrain then can  be conviniently expressed as a linear matrix equality $Az' = b$, where:
$$
A = 
\begin{bmatrix}
 & - - - & w_1 & - - - & \\
 & - - - & w_2 & - - - & \\
 & - - - & \theta_{\rho} & - - - & \\
 & - - - & \theta_{\alpha} & - - - & \\
 & - - - & \theta_{\beta} & - - - & \\
\end{bmatrix}, b' = \begin{bmatrix}
 & \lambda'_1& \\
 & \lambda'_2& \\
 & f' & \\
 & \alpha'& \\
 & \beta' & \\
\end{bmatrix}
$$

### Now 
Now we also want that z' to be 'close' to z in the latent space, our ditance metric is l2 norm between $z$  and $z'$ : $\|z - z'\|^2$


#### 
our new point z' must solve
$$
\begin{aligned}
    & \underset{z'}{\text{minimize}}
    & & \|z - z'\|^2 \\
    & \text{subject to}
    & & Az' = b'
\end{aligned}
$$

This is a classical convex optimisation problem.
Analiitical solution is :

The solution to that 
$$
z' = z + A^T (A A^T)^{-1} (A z - b) = z + A^T (A A^T)^{-1} (b - b') = 
$$
# Training

Training is done on Kaggle, using 
Pytorch lightning for the pipeline
WandB for the monitoring

# Model

Model is Auto encoder constitute of a ResNet of 1d convolution and a dense layer. 
The model is not a variational auto encoder, because of fiew resons.
It alows to load directly the samples from your sample pack in the vst plugging by picking the corresponding point in the latent space
It alows better reconstruction quality

# Loss

First experiment using mse temporal loss had high frequency ...
Using the time frequency multy rsolution loss, the results were much better
Also an other default of neural synthesis of the cick, the attack (0.05 first second) is really important to the final impression for us. This issue was improoved using a weighted loss that has hight ponderation at the begining. This was achieved using a multiplier anvlop parametrised as $env(x) = 1 + Ke^{-\frac{x}{\tau}}$
$\alpha = 100, K = 5, \tau = 0.1$

#

- Tonal and saturated, 
- subi / no sub
- Long tale - short transient
- Punch at the start
- top clip
- High ends
- Transient part - boby part - tail
- Sine sweep
- Dry (ne reverb) vs wet
- Round vs dirty



echo "# LatentPlay" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/AdhemarDeSenneville/LatentPlay.git
git push -u origin main

git submodule update --init --recursive

cd Simple_EQ_VST_Juce/SimpleEQ/

git branch -r

git checkout <branch-name>

cd ../../..

git add .

git commit -m "Your commit message"
