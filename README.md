CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

## SIREESHA PUTCHA 
	
* <img src= "img/Logos/linkedin.png" alt = "LinkedIn" height = "30" width = "30">   [ LinkedIn ](https://www.linkedin.com/in/sireesha-putcha/)

* <img src= "img/Logos/facebook.png" alt = "Fb" height = "30" width = "30">  [ Facebook ](https://www.facebook.com/sireesha.putcha98/)

* <img src= "img/Logos/chat.png" alt = "Portfolio" height = "30" width = "30">   [ Portfolio ](https://sites.google.com/view/sireeshaputcha/home)

* <img src= "img/Logos/mail.png" alt = "Mail" height = "30" width = "30">  [ Mail ](sireesha@seas.upenn.edu)


* Tested on personal computer - Microsoft Windows 10 Pro, 
Processor : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 2601 Mhz, 6 Core(s), 12 Logical Processor(s)
 
GPU : NVIDIA GeForce RTX 2060

## Denoiser 
In this project, I have implemented a pathtracing denoiser that uses geometry buffers (G-buffers) to guide a smoothing filter.
The technique I have used is based on the paper "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering," by Dammertz, Sewtz, Hanika, and Lensch. 
You can find the paper here: https://jo.dreggn.org/home/2010_atrous.pdf

Denoisers can help produce a smoother appearance in a pathtraced image with fewer samples-per-pixel/iterations, although the actual improvement often varies from scene-to-scene. 
Smoothing an image can be accomplished by blurring pixels - a simple pixel-by-pixel blur filter may sample the color from a pixel's neighbors in the image, weight them by distance, 
and write the result back into the pixel.

However, just running a simple blur filter on an image often reduces the amount of detail, smoothing sharp edges. This can get worse as the blur filter gets larger, or with more blurring passes. 
Fortunately in a 3D scene, we can use per-pixel metrics to help the filter detect and preserve edges.

Per-pixel metrics can include scene geometry information (hence G-buffer), such as per-pixel normals and per-pixel positions, as well as surface color or albedo for preserving detail in mapped or procedural textures.

## Edge Avoiding À-Trous Filter Algorithm 

This algorithm takes the ray traced image, position and normal buffers as input and produces a denoised image as output. In our approach, we store the position and normals after the first bounce
in G-Buffers. We also use Ping pong buffers for the output since reading and writing to the same buffer can cause race conditions.
<img src="/img/denoise/algo.png"/>

Each iteration of the À-Trous algorithm (with increasing step width) corresponds to computing one more level of the wavelet analysis which also corresponds to doubling the filter size.

1. At level i = 0 we start with the input signal c0(p)
2. ci+1(p) = ci(p) ∗ hi
, where ∗ is the discrete convolution.
The distance between the entries in the filter hi
is 2i
.
3. di(p) = ci+1(p)−ci(p),
where di are the detail or wavelet coefficients of level i.
4. if i < N (number of levels to compute):
increment i, go to step 2
5. {d0,d1,...,dN−1, cN} is the wavelet transform of c.
The reconstruction is given by

<img src="/img/denoise/reconstructiom.png"/>

## Performance Analysis 

- Influence of denoising on number of iterations needed to get a smooth result

The images below show the result after 5 iterations. We can see that the denoised image on the right corner looks way smoother than the noisy image on the left. We can conclude that
we can get a satisfactory result even with lesser number of iterations with denoising. 

<p float="left">
 <img src="/img/denoise/noise.png" height = "270" width = "270" />
 <img src="/img/denoise/blur.png" height = "270" width = "270" />
 <img src="/img/denoise/gbuf.png" height = "270" width = "270" />
</p>

- Impact on denoising on runtime

The image below shows the time taken for denoising the image in the last iteration. 

<img src="/img/denoise/timing_denoise.png" height = "400" width = "400" /> 

- Effectiveness with different material types

The denoiser doesn't take into account the material type. It only deals with the color of the pixel after the final iteration using monte carlo path tracing. We do account for material 
type during path tracing so we observe that smoothing is unaffected by material type. 

- Denoising with different filter sizes 

The images below show the outputs at filter sizes 3, 4 and 5 respectively. For my denoiser, a filter size of 4 works decently. 

<p float="left">
 <img src="/img/denoise/comp_filter_3.png" height = "270" width = "270" />
 <img src="/img/denoise/comp_filter_4.png" height = "270" width = "270" />
 <img src="/img/denoise/comp_filter_5.png" height = "270" width = "270" />
</p>

