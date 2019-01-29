import imgaug as ia
from imgaug import augmenters as iaa
sometimes_5 = lambda aug: iaa.Sometimes(0.5, aug)
sometimes_1 = lambda aug: iaa.Sometimes(0.1, aug)
augmentations=iaa.Sequential([
            sometimes_1(iaa.CropAndPad(
                percent=(-0.02, 0.02),
                pad_mode=["edge"],
                pad_cval=(0, 255)
            )),
            #This inverts completely inverts the text ( make black white etc.)
            sometimes_1(iaa.Invert(1, per_channel=True)), 

            #This does some affine transformations
            sometimes_1(iaa.Affine(
                scale={"x": (0.8, 1), "y": (0.8, 1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0, 0), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-2, 2), # rotate by -45 to +45 degrees
                shear=(-2, 2), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=["edge"]# use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            sometimes_1(iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
                #iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
            ])),
            sometimes_1(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            sometimes_1(iaa.AddToHueAndSaturation((-20, 20))), # change hue and saturation
            sometimes_1(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
            sometimes_1(iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.2)), # move pixels locally around (with random strengths)
            sometimes_1(iaa.PiecewiseAffine(scale=(0.001, 0.005))), 

        ])