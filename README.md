# RC-SPAN

Low Complexity RCAN  By Channel Splitting Madule for Single-Image Super Resolution (RC-SPAN) The code is based on the EDSR (Enhanced Deep Super-Resolution) and RCAN (Residual Channel Attention Network) methods. https://github.com/sanghyun-son/EDSR-PyTorch https://github.com/yulunzhang/RCAN This implementation takes the foundations of those previous models and builds upon them.
this code is an implementation of the paper titled "Low Complexity RCAN  By Channel Splitting Madule for  Single-Image Super Resolution (RC-SPAN) " which was recently submitted.
Deep learning has made significant strides in improving Single Image Super-Resolution (SISR) tasks, although it often comes with high computational demands. Achieving decreased computational complexity in deep learning involves reducing the parameter count of networks. Models with fewer parameters not only lead to faster inference times, reduced memory needs, and enhanced generalization capabilities. This paper presents an innovative approach to reduce the parameters in SISR models without significantly compromising performance. By splitting input channels, processing them independently, and then recombining them, the overall parameter count is reduced while maintaining input dimensionality. This paper contributes to developing resource-efficient SISR models that strike a balance between accuracy and computational requirement.
The proposed method was evaluated on five benchmark datasets across three different scaling factors. The results were assessed using performance metrics and visual evaluations. The proposed method reduces the number of parameters and number of FLOPs by 47.5% without significantly reducing the visual and qualitative results. In fact, it creates an acceptable trade-off between the number of parameters and the accuracy of the results.
We select the Residual Channel Attention Network (RCAN) as the baseline model for the single image super-resolution task [7]. The architecture of this model, depicted in (Fig.1), comprises six key components: a shallow feature extraction module, multiple residual groups, residual blocks, a channel attention mechanism, an up-scaling module, and a reconstruction part.
 The shallow feature extractor acts as the first layer, extracting basic features from the input image. These features are then passed through a series of residual groups, each containing multiple residual blocks. The residual blocks use skip connections to improve gradient flow during the training process. 
 
Figure 1: Structure of RCAN Model
we aim to calculate the total number of parameters in the RCAN architecture. Referring to Fig.1, the first layer is the shallow feature extraction module. Since this module operates on 3 input channels with C=64 filters and employs 3x3 convolutional kernels, the number of parameters in this layer can be determined as follows:
Number of parameters in shallow feature extraction = 3 (input channels) x 64 (filters) x 9 (kernel size 3x3) = 1,728 parameters.
Next, the number of parameters in each Residual Channel Attention Block (RCAB) is determined as follows. Before the channel attention module within an RCAB, there are two convolutional layers( ConvA & ConvB), each having 64 input channels, 64 output channels, and 3x3 kernel sizes. The number of parameters for these two layers is: 2 x (64 input channels x 64 output channels x 9 for 3x3 kernel) = 73,728 parameters 
In the channel attention layer of the RCAB, there is a channel downscaling convolution with 64 input channels and 4 output channels (C/r where r=16), followed by a channel upscaling convolution with 4 input channels and 64 output channels. The number of parameters for channel downscaling and channel upsampling is 64 * 4 = 256  and 4 * 64 = 256 parameters respectively. So the total number of parameters in the channel attention module is 256 + 256 = 512.
In the RCAN, there are 200 Residual Channel Attention Blocks (RCAB)  sequentially ((G=10)*(b=20)), so the total number of parameters in this module is 14,848,000, (200×(73728+512). In Fig.1, after the 20 Residual Channel Attention Blocks (RCABs), there is a convolution layer with dimensions 64*64*9, resulting in a total of 36,864 parameters. This layer is repeated 10 times, so the number of parameters in this layer  is 368,640. Furthermore, there is a convolution layer following the 10 residual groups with dimensions 64*64*9 which has 36,864 parameters. 
Subsequently, there is an upscale module where setting the scaling size to 4 results in 64*256*9 = 147,456 parameters. Lastly, there is one convolution layer with 64*3*9 = 1,728 parameters. Summing up the number of parameters for each layer without considering the bias, the total number of network parameters in the RCAN  is 15,303,040. 
For comparison the Number of parameter between the proposed method and the original RCAN model, Table 1 is provided, The number of parameters of each layer is given separately in this table. Careful examination of Table 1 reveals that by reducing the parameter count of the convolutional layers a and b through the proposed technique, the overall number of trainable parameters in the network has decreased by approximately 47.5%, going from around 15.3m parameters down to about 8m parameters. Fig 5 illustrates the components of the Channel Splitting Attention Module. We've incorporated this module into the RCAN architecture to develop a novel approach called the Residual Channel-Splitting Attention Network (RC-SPAN).
 
Figure 5: Channel Splitting Attention Madule (Proposed method)

4.1 Datasets and Evaluation Metrics
In the field of Single Image Super-Resolution, there are six benchmark datasets: DIV2K[20], Set5 [21], Set14[22], BSD100 [23], Urban100 [24] and Manga109 [25]. Recent studies primarily utilized the DIV2K dataset for model training because of its vast scale and diverse images. The performance of the models was assessed using SET5, SET14, BSD100, Urban100, and Manga109 datasets. To ensure a fair comparison, we followed the same approach. Our model was trained on the DIV2K dataset for image super-resolution, while the other benchmark datasets were used exclusively for evaluating performance.
To evaluate our proposed image compression technique against existing alternatives, we employ a comprehensive assessment strategy. Our evaluation framework incorporates multiple metrics to provide a holistic performance analysis[4, 13]:
	Peak Signal-to-Noise Ratio (PSNR) serves as our primary quantitative benchmark, offering a numerical measure of image quality.
	Structural Similarity Index (SSIM) is utilized to evaluate the perceived quality of compressed images.
	Network parameter count is considered to assess the model's complexity and potential for real-world deployment.
	Perception Index (PI) is included to capture subjective aspects of image quality from a human viewer's perspective.
	Floating Point Operations (FLOP) count is measured to evaluate computational complexity.
	The number of multiply-add operations is calculated to estimate processing efficiency.
This multi-faceted approach integrates objective mathematical metrics, subjective human perception factors, and computational efficiency measures. By doing so, we aim to provide a comprehensive evaluation of our compression method's performance across technical, practical, and computational dimensions. The inclusion of FLOP count and multiply-add operations provides valuable insights into the method's computational requirements and potential processing speed. These metrics are particularly relevant for real-world applications, especially in scenarios involving resource-constrained environments or the processing of large-scale datasets.
4.2 Implementation Details
The proposed methodology was executed on a high-capacity computing system. This workstation featured 64 GB of RAM and an Intel Core i7 processor, operating on a 64-bit Windows 10 platform. Graphics processing was handled by an NVIDIA GeForce RTX 3060 with 12 GB of dedicated memory. The software stack included PyTorch for deep learning operations and Visual Studio Code for development tasks.
The model architecture comprised 10 residual groups, each containing 20 residual blocks. Every convolutional layer utilized 64 channels, with a batch normalization size of 16 and a reduction ratio of 16. For optimization, the Adam algorithm was employed with an initial learning rate of 10-4. This rate was decreased by half every 200 update steps over a total of 1000 training epochs.



![image](https://github.com/user-attachments/assets/b0c2dd56-7c55-4e1f-802a-a93bf71f9846)


 




