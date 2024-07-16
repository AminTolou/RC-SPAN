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

Databases  	Set 5 [21]
Set14 [22]
BSD100 [23]
Urban100 [24]
Manga109 [25]

Methods	Scale	PSNR	SSIM	PSNR	SSIM	PSNR	SSIM	PSNR	SSIM	PSNR	SSIM
Bicubic[2]
×2	33.66	0.9299	30.24	0.8688	29.56	0.8431	26.88	0.8403	30.80	0.9339
SRCNN[5]
	36.66	0.9542	32.45	0.9067	31.36	0.8879	29.50	0.8946	35.60	0.9663
VDSR [6]
	37.53	0.9590	33.05	0.9130	31.90	0.8960	30.77	0.9140	37.22	0.9750
SeaNet[12]
	38.08	0.9609	33.75	0.9190	32.27	0.9008	32.50	0.9318	38.76	0.9774
RNAN[10]
	38.17	0.9611	33.87	0.9207	32.32	0.9014	32.73	0.9340	39.23	0.9785
EDSR[8]
	38.11	0.9602	33.92	0.9195	32.32	0.9013	32.93	0.9351	39.10	0.9773
RDN[9]
	38.24	0.9614	34.01	0.9212	32.34	0.9017	32.89	0.9353	39.18	0.9780
RCAN[11]
	38.27	0.9614	34.12	0.9216	32.41	0.9027	33.34	0.9384	39.44	0.9786
RC-SPAN		38.25	0.9614	34.04	0.921	32.36	0.902	33.06	0.9366	39.26	0.9782
RC-SPAN+		38.31	0.9617	34.11	0.9217	32.403	0.9024	33.273	0.9383	39.45	0.9786
Bicubic[2]
×3	30.39	0.8682	27.55	0.7742	27.21	0.7385	24.46	0.7349	26.95	0.8556
SRCNN[5]
	32.75	0.9090	29.30	0.8215	28.41	0.7863	26.24	0.7989	30.48	0.9117
VDSR [6]
	33.67	0.9210	29.78	0.8320	28.83	0.7990	27.14	0.8290	32.01	0.9340
SeaNet[12]
	34.55	0.9282	30.42	0.8445	29.17	0.8071	28.50	0.8594	33.73	0.9463
RNAN[10]
	---	---	---	---	---	---	---	---	---	---
EDSR[8]
	34.65	0.9280	30.52	0.8462	29.25	0.8093	28.80	0.8653	34.17	0.9476
RDN[9]
	34.71	0.9296	30.57	0.8468	29.26	0.8093	28.80	0.8653	34.13	0.9484
RCAN[11]
	34.74	0.9299	30.65	0.8482	29.32	0.8111	29.09	0.8702	34.44	0.9499
 RC-SPAN		34.71	0.9298	30.55	0.8465	29.23	0.8089	28.74	0.8644	34.02	0.9478
RC-SPAN+		34.78	0.9302	30.638	0.8478	29.31	0.8103	29.04	0.8687	34.403	0.9497
Bicubic[2]
×4	28.42	0.8104	26.00	0.7027	25.96	0.6675	23.14	0.6577	24.89	0.7866
SRCNN[5]
	30.48	0.8628	27.50	0.7513	26.90	0.7101	24.52	0.7221	27.58	0.8555
VDSR [6]
	31.35	0.8830	28.02	0.7680	27.29	0.0726	25.18	0.7540	28.83	0.8870
SeaNet[12]
	32.33	0.8970	28.72	0.7855	27.65	0.7388	26.32	0.7942	30.74	0.9129
RNAN[10]
	32.49	0.8982	28.83	0.7878	27.72	0.7421	26.61	0.8023	31.09	0.9149
EDSR[8]
	32.46	0.8968	28.80	0.7876	27.71	0.7420	26.64	0.8033	31.02	0.9148
RDN[9]
	32.47	0.8990	28.81	32.47	27.72	0.7419	26.61	0.8028	31.00	0.9151
RCAN[11]
	32.63	0.9002	28.87	0.7889	27.77	0.7436	26.82	0.8087	31.22	0.9173
 RC-SPAN		32.47	0.8985	28.8	0.7872	27.72	0.7417	26.70	0.8046	31.02	0.9158
RC-SPAN+		32.59	0.9001	28.9	0.7891	27.79	0.7433	26.91	0.8090	31.40	0.9190


![image](https://github.com/user-attachments/assets/b0c2dd56-7c55-4e1f-802a-a93bf71f9846)


 	 	 	 	 	 	 	 
Urban 100-img092-X4	HR	Bicubic
16.58/0.4377
/(0.3570)	EDSR
19.04/0.6609
/(0.4892)	SeaNET
19.11/0.6725
/(0.4956)	RCAN
19.640.6964
/(0.5119)	RC-SPAN
19.57/0.6930
/(0.5096)	RC-SPAN+
19.72/0.6959
/(0.5123)
 	 	 	 	 	 	 	 
Urban 100-img076-X4	HR	Bicubic
21.57/0.6288
/(0.4942)	EDSR
23.07/0.7367
/(0.5606)	SeaNET
23.13/0.7439
/(0.5647)	RCAN
24.30/0.7896
/(0.5973)	RC-SPAN
24.07/0.7889
/(0.5901)	RC-SPAN+
24.38/0.7912
/(0.5988)





