# 16-726 Final Project - GAN Photo Editing
### Akhil Eppa (aeppa@andrew.cmu.edu), Roshini Rajesh Kannan (rrajeshk@andrew.cmu.edu), Sanjana Moudgalya (smoudgal@andrew.cmu.edu)


## Introduction
Image restoration has seen a great deal of progress, with the creation of contemporary photographs using 
filters like denoising, colorization, and super resolution on old, grainy, black-and-white photos. 
In addition to this, the creators of Time-Travel Rephotography employ StyleGAN2 to transpose outdated 
images into a contemporary high-resolution image space. In order to imitate the characteristics of 
vintage cameras and the aging process of film, they use a physically based film deterioration operator. 
They eventually create the model output image using contextual loss and color transfer loss. 
The process of converting ancient photos to a modern versions gives the audience a perspective 
of how someone would have looked during the time and helps revisualize the aspect of color as well.

## Note
The code base that we used in this project is referenced from [Deoldify](https://github.com/jantic/DeOldify) and [Time-Travel-Photography](https://github.com/Time-Travel-Rephotography/Time-Travel-Rephotography.github.io). The baseline implementation was taken from these repositories and modifications were made in order to make improvements. 
