# DCGAN

The implementation of DCGAN (Deep Convolutional Generative Adversarial Networks) studied from https://arxiv.org/abs/1511.06434 has been done in PyTorch. I have trained a Generative Adversarial Network (GAN) and generated cat faces after showing it original (real) cat faces. The document covers the information about GAN, DCGAN, and its implementation to generate cat faces.

## Generative Adversarial Networks (GAN)

In simple terms, GANs are used to train a DL model to capture training data distribution so as to generate new data from the same distribution. In 2014, Ian Goodfellow et. al. came up with GAN for their work (https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf). GAN consists of two models: Generator and Discriminator. The job of the generator is to generate fake images. The job of the discriminator is to discriminate between fake images and real images. During the training phase, the generator tries to generate better and better fake images to outsmart the discriminator.

## Deep Convolutional Generative Adversarial Network (DCGAN)

* The Discriminator model comprises of strided convolutional layers, Batch Normalization layers, and LeakyReLU activations. The input for a discriminator is a 3x64x64 image and the output is the probability that the input is real data distribution.
* The Generator model comprises of convolutional-transpose layers, Batch Normalization laters, and ReLU activations. The input for a generator is a latent vector space, z, that is obtained from the standard normal distribution and the output is a generated 3x64x64 image.


** Implementation steps **
1. Download the cats-faces dataset from https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models
2. Paste the downloaded data into the “dataset” folder
3. Create an empty directory named “generated_images” which will contain the output of our DCGAN
4. Run the notebook (CatGAN.ipynb)
