from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import ImageEnhance
import os
import shutil
import numpy as np
import random
import cv2
import albumentations as A

from albumentations import (
### Pixel-level transforms
### simple color transform
	ChannelShuffle, HueSaturationValue, InvertImg, RGBShift, ToGray, ToSepia, ChannelDropout,
### complex_color_transform
	Equalize, Posterize, CLAHE, IAASharpen, RandomGamma, FancyPCA, IAAEmboss,
	RandomBrightness, RandomContrast, RandomBrightnessContrast,
### special efficacy transform	
	RandomFog, RandomRain, RandomShadow, RandomSnow, RandomSunFlare, Solarize, 
### blur and noise transform	
	Blur, GaussianBlur, GlassBlur, GaussNoise, MedianBlur, MotionBlur, IAAAdditiveGaussianNoise, 
	### hole in grid transform
	CoarseDropout,
### destroy transform
	Downscale, ImageCompression, JpegCompression,
### ---------------------------
	Cutout, IAASuperpixels, ISONoise, MultiplicativeNoise, 
	ToFloat, FromFloat, Normalize, 
### Spatial-level transforms
	### rigid transform
	ShiftScaleRotate, RandomSizedCrop, RandomResizedCrop, CropNonEmptyMaskIfExists, Rotate, RandomRotate90, 
	Flip, HorizontalFlip, Transpose, VerticalFlip, IAAFliplr, IAAFlipud,
	### basic rigid transform
	RandomScale, Resize, LongestMaxSize, SmallestMaxSize, 
	Crop, CenterCrop, RandomCrop, IAACropAndPad,
	### deform transform
	ElasticTransform, GridDistortion, IAAAffine, OpticalDistortion, IAAPiecewiseAffine, IAAPerspective,
	### hole in grid transform
	GridDropout,
### ---------------------------
	MaskDropout, NoOp, PadIfNeeded,
	RandomCropNearBBox, RandomGridShuffle, RandomSizedBBoxSafeCrop, 
	Lambda,
### Core
	BboxParams, Compose, KeypointParams, OneOf, OneOrOther
)

border_mode = [
	cv2.BORDER_CONSTANT,	# 0
	cv2.BORDER_REPLICATE,	# 1
	cv2.BORDER_REFLECT,		# 2
	cv2.BORDER_WRAP,		# 3
	cv2.BORDER_REFLECT_101	# 4, default
]

interpolation = [
	cv2.INTER_NEAREST,	# 0
	cv2.INTER_LINEAR,	# 1, default
	cv2.INTER_CUBIC,	# 2
	cv2.INTER_AREA,		# 3
	cv2.INTER_LANCZOS4	# 4
]

### 
batch_size = 128    
target_size = (512, 512)
# target_size = (280, 280)
# target_size = (360, 360)
interpolation = 'bilinear' #"nearest", "bilinear", "nearest" 為 SAIAP 內部預設

source_dir=r"D:\NCKU\Class In NCKU\DeepLearning\AOI\aoi_data\train_images" # 圖片來源
output_dir=r"D:\NCKU\Class In NCKU\DeepLearning\AOI\aoi_data\train_images_gen" # 輸出目錄

###
gen_id_for_this_run = True # 對於程式每次執行隨機產生一個 0~9999 的亂數編名


def round_clip_0_1(x, **kwargs):
	return x.round().clip(0, 1)

def strong_aug(p=1.0, target_size=(360,360)):
	rigid_transform=[
		### Randomly apply affine transforms: translate, scale and rotate the input.
		A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, 
			value=None, mask_value=None, always_apply=False, p=0.5),
		### Crop a random part of the input and rescale it to some size.
		A.RandomSizedCrop(min_max_height=(int(target_size[0]//1.2),target_size[0]), height=target_size[0], width=target_size[1],
			w2h_ratio=1.0, interpolation=1, always_apply=False, p=0.5),
		### Torchvision’s variant of crop a random part of the input and rescale it to some size.
		A.RandomResizedCrop(height=target_size[0], width=target_size[1],
			scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=0.5),
		### Rotate the input by an angle selected randomly from the uniform distribution.	
		A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
		### [B] Crop area with mask if mask is non-empty, else make random crop.
	#	A.CropNonEmptyMaskIfExists(height=target_size[0], width=target_size[1], ignore_values=None, 
	#		ignore_channels=None, always_apply=False, p=0.5),
		### Flip the input either horizontally, vertically or both horizontally and vertically.
		A.Flip(always_apply=False, p=0.5),
		### Flip the input horizontally around the y-axis.
		A.HorizontalFlip(always_apply=False, p=0.5),
		A.IAAFliplr(always_apply=False, p=0.5),
		### Flip the input vertically around the x-axis.
		A.VerticalFlip(always_apply=False, p=0.5),
		A.IAAFlipud(always_apply=False, p=0.5),
		### Transpose the input by swapping rows and columns.
	#	Transpose(always_apply=False, p=0.5),
		A.NoOp(always_apply=False, p=0.5),
	]
	
	basic_rigid_transform=[
		### [B] Resize the input to the given height and width.
		A.Resize(height=target_size[0], width=target_size[1], interpolation=1, always_apply=False, p=0.5),
		### [B] Randomly resize the input. Output image size is different from the input image size.
		A.RandomScale(scale_limit=0.1, interpolation=1, always_apply=False, p=0.5),
		### [B] Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
		A.LongestMaxSize(max_size=max(target_size[0],target_size[1]), interpolation=1, always_apply=False, p=0.5),
		### [B] Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
		A.SmallestMaxSize(max_size=max(target_size[0],target_size[1]), interpolation=1, always_apply=False, p=0.5),
		### [B] Crop region from image.
		A.Crop(x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=0.5),
		### [B] Crop the central part of the input.
		A.CenterCrop(height=target_size[0], width=target_size[1], always_apply=False, p=0.5),
	#	IAACropAndPad(px=None, percent=None, pad_mode='constant', pad_cval=0, keep_size=True, always_apply=False, p=1.0),
		### [B] Crop a random part of the input.
		A.RandomCrop(height=target_size[0], width=target_size[1], always_apply=False, p=0.5),
		### Randomly rotate the input by 90 degrees zero or more times.
		A.RandomRotate90(always_apply=False, p=0.5),
		A.NoOp(always_apply=False, p=0.5),
	]
	
	simple_color_transform=[
		### Randomly rearrange channels of the input RGB image.
		A.ChannelShuffle(p=0.5),
		### Randomly change hue, saturation and value of the input image.
		A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
		### Invert the input image by subtracting pixel values from 255.
	#	InvertImg(p=0.5),
		### Randomly shift values for each channel of the input RGB image.
		A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
		### Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.
		A.ToGray(p=0.5),
		### Applies sepia filter to the input RGB image
		A.ToSepia(p=0.5),
		### Randomly Drop Channels in the input Image.
		A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
		A.NoOp(always_apply=False, p=0.5),
	]
	
	distortion_transform=[
		### Elastic deformation of images
		A.ElasticTransform(
			alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, 
			value=None, mask_value=None, always_apply=False, approximate=False, p=0.5),
		### [R] 
		A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
			value=None, mask_value=None, always_apply=False, p=0.5),
		### [R]
		A.OpticalDistortion(
			distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4,
			value=None, mask_value=None, always_apply=False, p=0.5),
		### Place a regular grid of points on the input and randomly move the neighbourhood of these point around via affine transformations.
		A.IAAPiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', always_apply=False, p=0.5),
		### Perform a random four point perspective transform of the input.
		A.IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5),
		A.NoOp(always_apply=False, p=0.5),
	]
	
	complex_color_transform=[
		### Equalize the image histogram
		A.Equalize(mode='pil', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5),
		### Reduce the number of bits for each color channel.
		A.Posterize(num_bits=4, always_apply=False, p=0.5),
		### Apply Contrast Limited Adaptive Histogram Equalization to the input image.
		A.CLAHE(clip_limit=(1,4), tile_grid_size=(8,8), p=0.5),
		### Sharpen the input image and overlays the result with the original image.
		A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
		A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
		### Randomly change brightness and contrast of the input image.
		A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
		### Augment RGB image using FancyPCA from Krizhevsky's paper "ImageNet Classification with Deep Convolutional Neural Networks"
		A.FancyPCA(alpha=0.1, always_apply=False, p=0.5),
		### Emboss the input image and overlays the result with the original image.
		A.IAAEmboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5),
		### No transform
		A.NoOp(always_apply=False, p=0.5),
	]
	
	destroy_transform=[
		### Decreases image quality by downscaling and upscaling back.
		A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5),
		### Decrease Jpeg, WebP compression of an image.
		A.ImageCompression(quality_lower=99, quality_upper=100, compression_type=ImageCompression.ImageCompressionType.JPEG, always_apply=False, p=0.5),
		### Decrease Jpeg compression of an image.
		A.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
		### No transform
		A.NoOp(always_apply=False, p=0.5),
	]
	 
	noise_transform=[
		### Apply gaussian noise to the input image.
		A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
		### Add gaussian noise to the input image.
		A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003, 12.75), per_channel=False, always_apply=False, p=0.5),
		### No transform
		A.NoOp(always_apply=False, p=0.5),
	]

	blur_transform=[
		### Blur the input image using a random-sized kernel. 
		A.Blur(blur_limit=7, always_apply=False, p=0.5),
		### Blur the input image using using a Gaussian filter with a random kernel size
		A.GaussianBlur(blur_limit=(3, 7), always_apply=False, p=0.5),
		### Apply glass noise to the input image.
		A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5),
		### Apply motion blur to the input image using a random-sized kernel.
		A.MotionBlur(blur_limit=(3,7), always_apply=False, p=0.5),
		### Blur the input image using using a median filter with a random aperture linear size.
		A.MedianBlur(blur_limit=7, always_apply=False, p=0.5),
		### Add gaussian noise to the input image.
		A.NoOp(always_apply=False, p=0.5),
	]

	grid_hole_transform=[
		### [R] GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
		A.GridDropout(ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None,
				shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),		
		### CoarseDropout of the rectangular regions in the image.
		A.CoarseDropout(
			max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None,
			fill_value=[255,0,0], always_apply=False, p=1.0),
		### No transform
		A.NoOp(always_apply=False, p=0.5),
	]
	
	special_efficacy_transform=[
		### Adds rain effects.
		A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, 
			drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5),
		### Bleach out some pixel values simulating snow.
		A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5),
		### Simulates Sun Flare for the image
		A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, 
			num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5),
		### Simulates shadows for the image	
		A.RandomShadow(p=0.5, num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1)),
		### Simulates fog for the image
		A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5),
		### Invert all pixel values above a threshold.
		A.Solarize(threshold=128, always_apply=False, p=0.5),
		### No transform
		A.NoOp(always_apply=False, p=0.5),
	]
	
	no_use=[
		### Take an input array where all values should lie in the range [0, 1.0], 
		### multiply them by max_value and then cast the resulted value to a type specified by dtype.
		### If max_value is None the transform will try to infer the maximum value for the data type from the dtype argument.
		A.FromFloat(dtype='uint16', max_value=None, always_apply=False, p=1.0),
	]

	return Compose([
		A.OneOf(rigid_transform,p=1.0),
		A.OneOf(basic_rigid_transform,1.0),
		A.OneOf(simple_color_transform, p=1.0),
		A.OneOf(distortion_transform, p=1.0),
		A.OneOf(complex_color_transform, p=1.0),
		A.OneOf(destroy_transform,p=1.0),
		A.OneOf(blur_transform,p=1.0),
		A.OneOf(noise_transform,p=1.0),
		A.OneOf(grid_hole_transform,p=1.0),
		A.OneOf(special_efficacy_transform,p=1.0),
	], p=p)

# Use Albumentations library for data augmentation (bool)
opt_DA_ADVANCED_AUGMENTATION = True
# probability of applying the augmentation. (float:0.0~1.0)
opt_DA_ADV_AUGMENTATION_PROBABILITY = 1.0
# Dimension of Input of Model (H,W)
opt_PREDICTOR_INPUT = (512, 512)
# Range for random zoom. float or [lower, upper].  ex. [0.8, 1.2]
# If a float, [lower, upper] = [1 - DA_ZOOM_RANGE, 1 + DA_ZOOM_RANGE].
opt_DA_ZOOM_RANGE = [0.98, 1.02]
# Degree range for random rotations. (float)
opt_DA_ROTATION_RANGE = 3
# Randomly Shift image in horizontial direction
# float: fraction of total height, if < 1, or pixels if >= 1.
# With DA_WIDTH_SHIFT_RANGE=2 possible values are integers [-1, 0, +1], 
# while with DA_WIDTH_SHIFT_RANGE=1.0 possible values are floats in the interval [-1.0, +1.0).
opt_DA_WIDTH_SHIFT_RANGE = 5
# Randomly shift image in vertical direction
# float: fraction of total height, if < 1, or pixels if >= 1.
# With DA_HEIGHT_SHIFT_RANGE=2 possible values are integers [-1, 0, +1], 
# while with DA_HEIGHT_SHIFT_RANGE=1.0 possible values are floats in the interval [-1.0, +1.0).
opt_DA_HEIGHT_SHIFT_RANGE = 5
# Randomly flip inputs horizontally (bool)
opt_DA_HORIZONTAL_FLIP = True
# Randomly flip inputs vertically. (bool)
opt_DA_VERTICAL_FLIP = True
# Randomly shift values for each channel of the input RGB image.
# Range for random channel shifts. ex. 10.0 (float)
opt_DA_CHANNEL_SHIFT_RANGE = 0.0
# Randomly change brightness of the input image. Tuple or list of two floats.
# Range for picking a brightness shift value from. ex. [0.8,1.2](list or tuple)
# 1.0 represent the original brightness
opt_DA_BRIGHTNESS_RANGE = [0.99, 1.01]
# Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
opt_DA_SHEAR_RANGE = None
# Random crop the input image within the range of pixels [180,224]
opt_DA_ADV_RANDOM_CROP = None
# Crop image 4 sides with fraction of image width and height, and pad with black
opt_DA_ADV_CENTER_CROP = None #0.225
# Horizontal Offset of Center Crop image (pixel)
opt_DA_ADV_CENTER_CROP_SH = 0
# Vertical Offset of Center Crop image (pixel)
opt_DA_ADV_CENTER_CROP_SV = 0
# Randomly change hue, saturation and value of the input image. (bool or list ex. [20. 30, 20])
opt_DA_ADV_HSV_SHIFT = None #[20, 30, 20]
# Randomly rotate the input by 90 degrees zero or more times. (bool)
opt_DA_ADV_ROTATION90 = True
# Randomly change contrast of the input image. Tuple or list of two floats.
# Range for picking a contrast shift value from. ex. [0.8,1.2](list or tuple)
opt_DA_ADV_CONTRAST_RANGE = [0.99, 1.01]
#opt_DA_ADV_CONTRAST_RANGE = None
# Apply Grid Distortion/Optical Distortion/Piecewise Affine Transform/Elastic Transform to the input image
opt_DA_ADV_DISTORTION_ALL = False
# Distort the input image just like optical distortion of imperfect lens. (bool) 
opt_DA_ADV_DISTORTION_OPTICAL_DISTORTION = False
# Apply grid distortion to input image (bool)
opt_DA_ADV_DISTORTION_GRID_DISTORTION = False
# Elastic deformation of images as described in [Simard2003]_ (with modifications). 
# Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5 (bool)
opt_DA_ADV_DISTORTION_ELASTICTRANSFORM = False
# This augmenter places a regular grid of points on an image and randomly moves the neighbourhood of these point around via affine transformations.
# This leads to local distortions.
opt_DA_ADV_DISTORTION_PIECEWISE_AFFINE = False
# Perform a random four point perspective transform of the input. (bool)
opt_DA_ADV_DISTORTION_PERSPECTIVE = False
# Apply Gaussian Noise to the input image (bool)
opt_DA_ADV_NOISE = False
# Reduce the number of bits for each color channel. (bool)
opt_DA_ADV_POSTERIZE = False
# Blur the input image using a random-sized kernel (int or list/tuple)
opt_DA_ADV_BLUR = False#[3, 5]
# Motion Blur the input image using a random-sized kernel (int or list/tuple)
opt_DA_ADV_MOTION_BLUR = False#[3, 5]
# Blur the input image using a median filter with a random aperture linear size. 
opt_DA_ADV_MEDIAN_BLUR = False#[3, 5]
# Apply glass noise to the input image (bool).
opt_DA_ADV_GLASS_BLUR = False
# Apply Contrast Limited Adaptive Histogram Equalization to the input image.
opt_DA_ADV_CLAHE = False
# Equalize the image histogram
opt_DA_ADV_EQUALIZE = False
# Sharpen the input image and overlays the result with the original image.
# range to choose the visibility of the sharpened image. 
# At 0, only the original image is visible, at 1.0 only its sharpened version is visible. Default: ex.(0.2, 0.5)
opt_DA_ADV_SHARPEN_ALPHA_RANGE = True
# Sharpen the input image and overlays the result with the original image.
# range to choose the lightness of the sharpened image. Default: ex. (0.5, 1.0).
opt_DA_ADV_SHARPEN_LIGHTNESS_RANGE = True
# Convert the input RGB image to grayscale. (bool)
# If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.
opt_DA_ADV_TO_GRAY = False
# Applies sepia filter to the input RGB image (bool)
opt_DA_ADV_TO_SEPIA = False
# Randomly Drop Channels in the input Image. (bool)
opt_DA_ADV_CHANNEL_DROPOUT = False
# Emboss the input image and overlays the result with the original image. (bool)
opt_DA_ADV_EMBOSS = False
# Decreases image quality (bool)
opt_DA_ADV_DESTROY_QUALITY = False

def get_shift(x,width):
	x_pos=0.0625
	x_neg=-0.0625
	if isinstance(x,int) and x>=1:
		x_pos=x/width/2.0
		x_neg=-x/width/2.0
	elif isinstance(x,float) and x>1:
		x_pos=x/width/2.0
		x_neg=-x/width/2.0
	elif isinstance(x,float) and x<=1:
		x_pos=x
		x_neg=-x
	elif isinstance(x,(tuple,list)):
		if isinstance(x[1],int) and x[1]>=1:
			x_pos=x[1]/width
			x_neg=x[0]/width
		elif isinstance(x[1],float) and x[1]>1:
			x_pos=x[1]/width
			x_neg=x[0]/width
		elif isinstance(x[1],float) and x[1]<=1:
			x_pos=x[1]
			x_neg=x[0]
	return [x_neg,x_pos]


def kaug_to_alaug(params):
	if isinstance(params,(tuple,list)):
		ret_params=[x-1.0 for x in params]
	else:
		ret_params=params
	return ret_params

# define heavy augmentations for classification
def get_classification_augmentation(
#	opt,
	p=1.0):
	rigid_transform=[]
	rigid_transform_other=[]
	simple_color_transform=[]
	complex_color_transform=[]
	distortion_transform=[]
	noise_transform=[]
	blur_transform=[]
	destroy_transform=[]
	
	target_size=opt_PREDICTOR_INPUT
	### Basic image Crop if needed
	if opt_DA_ADV_CENTER_CROP and isinstance(opt_DA_ADV_CENTER_CROP,float) and opt_DA_ADV_CENTER_CROP < 1.0:
		x_min = int(target_size[1]*opt_DA_ADV_CENTER_CROP) + opt_DA_ADV_CENTER_CROP_SH
		y_min = int(target_size[0]*opt_DA_ADV_CENTER_CROP) + opt_DA_ADV_CENTER_CROP_SV
		x_max = target_size[1] - int(target_size[1] * opt_DA_ADV_CENTER_CROP) + opt_DA_ADV_CENTER_CROP_SH
		y_max = target_size[0] - int(target_size[1] * opt_DA_ADV_CENTER_CROP) + opt_DA_ADV_CENTER_CROP_SV
		train_transform = [
		A.Crop(
			x_min=x_min, y_min=y_min,
			x_max=x_max, y_max=y_max,
			always_apply=True, p=1.0),
		A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], always_apply=True, border_mode=0),		
		]
	else:
		train_transform = [
		]
	
	### Scale、Shift、Rotate Transform ----------------------
	if opt_DA_ZOOM_RANGE or opt_DA_WIDTH_SHIFT_RANGE or opt_DA_HEIGHT_SHIFT_RANGE:
		if opt_DA_WIDTH_SHIFT_RANGE:
			shift_limit_al=get_shift(opt_DA_WIDTH_SHIFT_RANGE,target_size[1])
		elif opt_DA_HEIGHT_SHIFT_RANGE:
			shift_limit_al=get_shift(opt_DA_HEIGHT_SHIFT_RANGE,target_size[0])
		else:
			shift_limit_al=0.0
		if opt_DA_ZOOM_RANGE:
			scale_limit_al=kaug_to_alaug(opt_DA_ZOOM_RANGE)
		else:
			scale_limit_al=0.0
		if opt_DA_ROTATION_RANGE:
			rotate_limit_al=kaug_to_alaug(opt_DA_ROTATION_RANGE)
		else:
			rotate_limit_al=0.0
		rigid_transform.append(A.ShiftScaleRotate(
			shift_limit=shift_limit_al, scale_limit=scale_limit_al, rotate_limit=rotate_limit_al,
			interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5))
	elif opt_DA_ROTATION_RANGE:
		rigid_transform.append(A.Rotate(limit=kaug_to_alaug(opt_DA_ROTATION_RANGE), always_apply=False, p=0.5))
	if opt_DA_ADV_RANDOM_CROP:
		if isinstance(opt_DA_ADV_RANDOM_CROP, (list,tuple)):
			if opt_DA_ADV_RANDOM_CROP[0] > target_size[0]:
				min_height=int(target_size[0]/1.2)
			else:
				min_height=int(opt_DA_ADV_RANDOM_CROP[0])
			if opt_DA_ADV_RANDOM_CROP[1] > target_size[0]:
				max_height=int(target_size[0])
			else:
				max_height=int(opt_DA_ADV_RANDOM_CROP[1])
		else:
			min_height=int(target_size[0]/1.2)
			max_height=int(target_size[0])
		w2h_ratio=target_size[1]/target_size[0]
		rigid_transform.append(A.RandomSizedCrop(min_max_height=[min_height, max_height], height=target_size[0], width=target_size[1],
			w2h_ratio=w2h_ratio, interpolation=1, always_apply=False, p=0.5))
	
	### rigid transform other ------------------------------------------
	if opt_DA_HORIZONTAL_FLIP and opt_DA_VERTICAL_FLIP:
		rigid_transform_other.append(A.Flip(always_apply=False, p=0.5))
	elif opt_DA_HORIZONTAL_FLIP:
		rigid_transform_other.append(A.HorizontalFlip(always_apply=False, p=0.5))
	elif opt_DA_VERTICAL_FLIP:
		rigid_transform_other.append(A.VerticalFlip(always_apply=False, p=0.5))
	
	if opt_DA_ADV_ROTATION90:
		rigid_transform_other.append(A.RandomRotate90(p=0.5))
	
	### simple color transform ------------------------------------------
	if opt_DA_CHANNEL_SHIFT_RANGE:
		if isinstance(opt_DA_CHANNEL_SHIFT_RANGE,(list,tuple)) and len(opt_DA_CHANNEL_SHIFT_RANGE)==3:
			simple_color_transform.append(A.RGBShift(
				r_shift_limit=opt_DA_CHANNEL_SHIFT_RANGE[0],
				g_shift_limit=opt_DA_CHANNEL_SHIFT_RANGE[1],
				b_shift_limit=opt_DA_CHANNEL_SHIFT_RANGE[2], always_apply=False, p=0.5))
		elif isinstance(opt_DA_CHANNEL_SHIFT_RANGE,(int,float)):
			simple_color_transform.append(A.RGBShift(
				r_shift_limit=int(opt_DA_CHANNEL_SHIFT_RANGE),
				g_shift_limit=int(opt_DA_CHANNEL_SHIFT_RANGE),
				b_shift_limit=int(opt_DA_CHANNEL_SHIFT_RANGE), always_apply=False, p=0.5))
		else:
			simple_color_transform.append(A.RGBShift(
				r_shift_limit=20,
				g_shift_limit=20,
				b_shift_limit=20, always_apply=False, p=0.5))
	if opt_DA_ADV_HSV_SHIFT:
		if isinstance(opt_DA_ADV_HSV_SHIFT,(list,tuple)) and len(opt_DA_ADV_HSV_SHIFT)==3:
			simple_color_transform.append(A.HueSaturationValue(
				hue_shift_limit=opt_DA_ADV_HSV_SHIFT[0],
				sat_shift_limit=opt_DA_ADV_HSV_SHIFT[1],
				val_shift_limit=opt_DA_ADV_HSV_SHIFT[2], always_apply=False, p=0.5))
		elif isinstance(opt_DA_CHANNEL_SHIFT_RANGE,(int,float)):
			simple_color_transform.append(A.HueSaturationValue(
				hue_shift_limit=int(opt_DA_ADV_HSV_SHIFT),
				sat_shift_limit=int(opt_DA_ADV_HSV_SHIFT),
				val_shift_limit=int(opt_DA_ADV_HSV_SHIFT), always_apply=False, p=0.5))		
		else:
			simple_color_transform.append(A.HueSaturationValue(
				hue_shift_limit=20,
				sat_shift_limit=30,
				val_shift_limit=20, p=0.5))
	if opt_DA_ADV_POSTERIZE:
		simple_color_transform.append(A.Posterize(num_bits=4, always_apply=False, p=0.5))
	if opt_DA_ADV_TO_GRAY:
		simple_color_transform.append(A.ToGray(p=0.5))
	if opt_DA_ADV_TO_SEPIA:
		simple_color_transform.append(A.ToSepia(p=0.5))
	if opt_DA_ADV_CHANNEL_DROPOUT:
		### Randomly Drop Channels in the input Image.
		simple_color_transform.append(A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5))
	
	### complex color transform ---------------------------------------
	if opt_DA_BRIGHTNESS_RANGE and opt_DA_ADV_CONTRAST_RANGE:
		complex_color_transform.append(A.RandomBrightnessContrast(
			brightness_limit=kaug_to_alaug(opt_DA_BRIGHTNESS_RANGE),
			contrast_limit=kaug_to_alaug(opt_DA_ADV_CONTRAST_RANGE),
			brightness_by_max=True, always_apply=False, p=0.5))
	elif opt_DA_ADV_CONTRAST_RANGE:
		complex_color_transform.append(A.RandomContrast(limit=kaug_to_alaug(opt_DA_ADV_CONTRAST_RANGE), p=0.5))
	elif opt_DA_BRIGHTNESS_RANGE:
		complex_color_transform.append(A.RandomBrightness(limit=kaug_to_alaug(opt_DA_BRIGHTNESS_RANGE), p=0.5))
	if opt_DA_ADV_CLAHE:
		complex_color_transform.append(A.CLAHE(clip_limit=(1,4), tile_grid_size=(8,8), p=0.5))
	if opt_DA_ADV_EQUALIZE:
		complex_color_transform.append(A.Equalize(mode='pil', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5))
	if opt_DA_ADV_EMBOSS:
		### Emboss the input image and overlays the result with the original image.
		complex_color_transform.append(A.IAAEmboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5))	
	
	### noise transform -----------------------------------------------
	if opt_DA_ADV_NOISE:
		noise_transform.extend([
			### Apply gaussian noise to the input image.
			A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
			### Add gaussian noise to the input image.
			A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003, 12.75), per_channel=False, always_apply=False, p=0.5),
			### No transform
			#A.NoOp(always_apply=False, p=0.5),
		])
	
	### distortion transform ------------------------------------------
	if opt_DA_SHEAR_RANGE:
		distortion_transform.extend([
			### Place a regular grid of points on the input and randomly move the neighbourhood of these point around via affine transformations.
			A.IAAAffine(shear=opt_DA_SHEAR_RANGE, p=0.5),
		])	
	if opt_DA_ADV_DISTORTION_ALL:
		distortion_transform.extend([
			A.OpticalDistortion(
				distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4,
				value=None, mask_value=None, always_apply=False, p=0.5),
			A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
				value=None, mask_value=None, always_apply=False, p=0.5),
			A.IAAPiecewiseAffine(
				scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', always_apply=False, p=0.5),
			A.ElasticTransform(
				alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, 
				value=None, mask_value=None, always_apply=False, approximate=False, p=0.5),
		])
	else:
		if opt_DA_ADV_DISTORTION_OPTICAL_DISTORTION:
			distortion_transform.extend([
				### distort thie input image just like optical distortion of imperfect lens.
				A.OpticalDistortion(
					distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4,
					value=None, mask_value=None, always_apply=False, p=0.5),
		])
		if opt_DA_ADV_DISTORTION_GRID_DISTORTION:
			distortion_transform.extend([
				###  
				A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
					value=None, mask_value=None, always_apply=False, p=0.5),		
			])
		if opt_DA_ADV_DISTORTION_ELASTICTRANSFORM:
			distortion_transform.extend([
				### Elastic deformation of images as described in [Simard2003]_ (with modifications). 
				### Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
				A.ElasticTransform(
					alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, 
					value=None, mask_value=None, always_apply=False, approximate=False, p=0.5),
			])
		if opt_DA_ADV_DISTORTION_PIECEWISE_AFFINE:
			distortion_transform.extend([
				### This augmenter places a regular grid of points on an image and randomly moves the neighbourhood of these point around via affine transformations.
				### This leads to local distortions.
				A.IAAPiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', always_apply=False, p=0.5),
			])
		if opt_DA_ADV_DISTORTION_PERSPECTIVE:
			distortion_transform.extend([
				### Perform a random four point perspective transform of the input.
				A.IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5),
			])
	
	### Blur transform -----------------------------------------------
	if opt_DA_ADV_SHARPEN_ALPHA_RANGE or opt_DA_ADV_SHARPEN_LIGHTNESS_RANGE:
		if opt_DA_ADV_SHARPEN_ALPHA_RANGE and isinstance(opt_DA_ADV_SHARPEN_ALPHA_RANGE,(list,tuple)) and len(opt_DA_ADV_SHARPEN_ALPHA_RANGE)==2:
			alpha_range=opt_DA_ADV_SHARPEN_ALPHA_RANGE
		else:
			alpha_range=(0.2, 0.5)
		if opt_DA_ADV_SHARPEN_LIGHTNESS_RANGE and isinstance(opt_DA_ADV_SHARPEN_LIGHTNESS_RANGE,(list,tuple)) and len(opt_DA_ADV_SHARPEN_LIGHTNESS_RANGE)==2:
			lightness_range=opt_DA_ADV_SHARPEN_LIGHTNESS_RANGE
		else:
			lightness_range=(0.5, 1.0)
		blur_transform.append(A.IAASharpen(
			alpha=alpha_range, lightness=lightness_range, always_apply=False, p=0.5))
	if opt_DA_ADV_BLUR and isinstance(opt_DA_ADV_BLUR,(list,tuple)) and len(opt_DA_ADV_BLUR)==2:
		if int(opt_DA_ADV_BLUR[0])<3:
			blur_limit_kernel=[3,int(opt_DA_ADV_BLUR[1])]
		else:
			blur_limit_kernel=[int(opt_DA_ADV_BLUR[0]),int(opt_DA_ADV_BLUR[1])]
		blur_transform.extend([
			### Blur the input image using a random-sized kernel. 
			A.Blur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5),
			### Blur the input image using using a Gaussian filter with a random kernel size
			A.GaussianBlur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5),
		])
	elif opt_DA_ADV_BLUR and not isinstance(opt_DA_ADV_BLUR, bool):
		if isinstance(opt_DA_ADV_BLUR, (int,float)) and int(opt_DA_ADV_BLUR)<3:
			blur_limit_kernel=3
		else:
			blur_limit_kernel=int(opt_DA_ADV_BLUR)
		blur_transform.extend([
			### Blur the input image using a random-sized kernel. 
			A.Blur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5),
			### Blur the input image using using a Gaussian filter with a random kernel size
			A.GaussianBlur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5),
		])
	elif opt_DA_ADV_BLUR:
		blur_transform.extend([
			### Blur the input image using a random-sized kernel. 
			A.Blur(always_apply=False, p=0.5),
			A.GaussianBlur(always_apply=False, p=0.5),
		])

	if opt_DA_ADV_MOTION_BLUR and isinstance(opt_DA_ADV_MOTION_BLUR,(list,tuple)) and len(opt_DA_ADV_MOTION_BLUR)==2:
		if int(opt_DA_ADV_MOTION_BLUR[0])<3:
			blur_limit_kernel=[3,int(opt_DA_ADV_MOTION_BLUR[1])]
		else:
			blur_limit_kernel=[int(opt_DA_ADV_MOTION_BLUR[0]),int(opt_DA_ADV_MOTION_BLUR[1])]
		blur_transform.append(
			### Apply motion blur to the input image using a random-sized kernel.
			A.MotionBlur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5)
		)
	elif opt_DA_ADV_MOTION_BLUR and not isinstance(opt_DA_ADV_MOTION_BLUR, bool):
		if isinstance(opt_DA_ADV_MOTION_BLUR, (int,float)) and int(opt_DA_ADV_MOTION_BLUR)<3:
			blur_limit_kernel=3
		else:
			blur_limit_kernel=int(opt_DA_ADV_MOTION_BLUR)
		blur_transform.append(
			### Apply motion blur to the input image using a random-sized kernel.
			A.MotionBlur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5)
		)
	elif opt_DA_ADV_MOTION_BLUR:
		blur_transform.append(
			### Apply motion blur to the input image using a random-sized kernel.
			A.MotionBlur(always_apply=False, p=0.5)
		)

	if opt_DA_ADV_MEDIAN_BLUR and isinstance(opt_DA_ADV_MEDIAN_BLUR,(list,tuple)) and len(opt_DA_ADV_MEDIAN_BLUR)==2:
		if int(opt_DA_ADV_MEDIAN_BLUR[0])<3:
			
			blur_limit_kernel=[3,(int(opt_DA_ADV_MEDIAN_BLUR[1])//2)*2+1]
		else:
			blur_limit_kernel=[(int(opt_DA_ADV_MEDIAN_BLUR[0])//2)*2+1,(int(opt_DA_ADV_MEDIAN_BLUR[1])//2)*2+1]
		blur_transform.append(
			### Blur the input image using using a median filter with a random aperture linear size.
			A.MedianBlur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5)
		)
	elif opt_DA_ADV_MEDIAN_BLUR and not isinstance(opt_DA_ADV_MEDIAN_BLUR, bool):
		if isinstance(opt_DA_ADV_MEDIAN_BLUR, (int,float)) and int(opt_DA_ADV_MEDIAN_BLUR)<3:
			blur_limit_kernel=3
		else:
			blur_limit_kernel=(int(opt_DA_ADV_MEDIAN_BLUR)//2)*2+1
		blur_transform.append(
			### Blur the input image using using a median filter with a random aperture linear size.
			A.MedianBlur(blur_limit=blur_limit_kernel, always_apply=False, p=0.5),
		)
	elif opt_DA_ADV_MEDIAN_BLUR:
		blur_transform.append(
			### Blur the input image using using a median filter with a random aperture linear size.
			A.MedianBlur(always_apply=False, p=0.5)
		)
	
	if opt_DA_ADV_GLASS_BLUR:
		blur_transform.append(
		### Apply glass noise to the input image.
			A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5)
		)
	
	
	### Destroy transform
	if opt_DA_ADV_DESTROY_QUALITY:
		destroy_transform.extend([
			### Decreases image quality by downscaling and upscaling back.
			A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5),
			### Decrease Jpeg, WebP compression of an image.
			A.ImageCompression(quality_lower=99, quality_upper=100,
				compression_type=A.ImageCompression.ImageCompressionType.JPEG, always_apply=False, p=0.5),
			### Decrease Jpeg compression of an image.
			#A.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
		])
	
	if rigid_transform:
#		rigid_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(rigid_transform, p=1.0))
	if rigid_transform_other:
#		rigid_transform_other.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(rigid_transform_other, p=1.0))
	if simple_color_transform:
#		simple_color_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(simple_color_transform, p=1.0))
	if complex_color_transform:
#		complex_color_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(complex_color_transform, p=1.0))
	if noise_transform:
#		noise_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(noise_transform, p=1.0))
	if blur_transform:
#		blur_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(blur_transform, p=1.0))
	if distortion_transform:
#		distortion_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(distortion_transform, p=1.0))
	if destroy_transform:
#		destroy_transform.append(A.NoOp(always_apply=False, p=0.5))
		train_transform.append(A.OneOf(destroy_transform, p=1.0))
	return A.Compose(train_transform)

############################################

# if gen_id_for_this_run:
# 	randgen_id=str(random.randint(0,9999))
# else:
# 	randgen_id=''
# os.makedirs(output_dir, exist_ok=True)
# source_dir=os.path.abspath(source_dir)

def saveimg_by_pil(f,img, scale=False):
	"""save numpy array of RGB 3 channels of shape (H,W,C) to image file by PIL.

	Args:
		f (str): The image filename to save to .
		img (array): The RGB 3 channels numpy array of image (H,W,C).
	
	Returns:
		
	"""
	image.save_img(
		path=f,
		x=img,
		data_format='channels_last',
		scale=scale)
    
gen_number = 10
for i in range(gen_number):
    
    if gen_id_for_this_run:
        randgen_id=str(random.randint(0,9999))
    else:
    	randgen_id=''
    os.makedirs(output_dir, exist_ok=True)
    source_dir=os.path.abspath(source_dir)
    
    
    
    
    
    train_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
    	source_dir,
    	shuffle = False,
    	batch_size = batch_size,
    	class_mode = 'categorical',
    	target_size = target_size,
    	interpolation = interpolation,
    )
    
    whatever_data = "my name"
    #augmentation = strong_aug(p=1.0,target_size=target_size)
    augmentation = get_classification_augmentation(p=opt_DA_ADV_AUGMENTATION_PROBABILITY)

 
    train_generator.reset()
    findex = 0
  
    for x in range(len(train_generator)):
    	input_list,output_label=train_generator.next()
    	for f in range(len(input_list)):
    		filepath = train_generator.filenames[findex]
    		dirname = os.path.dirname(filepath)
    		basename = os.path.basename(filepath)
    		filename, file_extension = os.path.splitext(basename)
    		output_subdir = os.path.join(output_dir,dirname)
    		os.makedirs(os.path.abspath(output_subdir), exist_ok = True)
    		image_original =  input_list[f].astype('uint8')
    		outfile = os.path.join(output_dir,dirname, filename+'_aug'+randgen_id+file_extension)
    		augmented  =  augmentation(image = image_original)
    		image_aug  =  augmented["image"]
    		saveimg_by_pil(outfile, image_aug)
    		findex = findex + 1
