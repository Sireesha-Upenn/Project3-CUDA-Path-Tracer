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

## Performance Analysis 

- Influence of denoising on number of iterations needed to get a smooth result

The images below show the result after 5 iterations. We can see that the denoised image on the right corner looks way smoother than the noisy image on the left. We can conclude that
we can get a satisfactory result even with lesser number of iterations with denoising. 

<p float="left">
 <img src="/img/denoise/noise.png" height = "250" width = "250" />
 <img src="/img/denoise/blur.png" height = "250" width = "250" />
 <img src="/img/denoise/gbuf.png" height = "250" width = "250" />
</p>

- Impact on denoising on runtime

The image below shows the time taken for denoising the image in the last iteration. 

<img src="/img/denoise/timing_denoise.png" height = "270" width = "270" /> 

- Effectiveness with different material types

The denoiser doesn't take into account the material type. It only deals with the color of the pixel after the final iteration using monte carlo path tracing. We do account for material 
type during path tracing so we observe that smoothing is unaffected by material type. 

- Denoising with different filter sizes 

The images below show the outputs at filter sizes 3, 4 and 5 respectively. For my denoiser, a filter size of 4 works decently. 

<p float="left">
 <img src="/img/denoise/comp_filter_3.png" height = "250" width = "250" />
 <img src="/img/denoise/comp_filter_4.png" height = "250" width = "250" />
 <img src="/img/denoise/comp_filter_5.png" height = "250" width = "250" />
</p>

