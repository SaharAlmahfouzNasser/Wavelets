# Introduction
The wavelet-based deep neural networks proved their efficiency in the domain of single image super-
resolution and image denoising. The core idea of these architectures is about replacing the pooling layers
with wavelet transform (WT). On the one hand, pooling layers are used in deep neural networks to avoid
overfitting and reduce the computation burden. On the other hand, these layers cause loss of information. In
2017, Bae et al showed that replacing the pooling layers with (DWT) improves the reconstruction results,
see figure 10. In the same year Guo et al proposed deep wavelet super-resolution network. This network
predicts the missing details of the wavelet coefficients of the low-resolution images to obtain the coefficients
of the SR image which serve as input to IDWT to generate the SR image.
In this project: I implemented U-Net, SRResNet, DCED networks. I edited the original networks by adding wavelets blocks.
