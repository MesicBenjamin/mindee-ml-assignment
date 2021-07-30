# ML assignment - OCR

## Introduction

My solution to https://github.com/mindee/ml-assignment.


## Setup

Based on good old cv2 :-)
```
pip install -r requirements.txt
```


## Prediction

```
python predict.py --draw True --name_img data/0A1NOPtu573ak.jpg
```


## How it works ?

The whole procedure is explained below as block of code followed by an image showing corresponding result.


### QR and Bar code detection


The first step is to convert to grayscale image and then invert the colors.
```
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
img_gray = 255 - img_gray # invert grayscale
```
![Step 1](data/steps/step1.jpg)

To highlight QR and suppress surrounding, two filters were applied.
As it can be seen below, QR is now much more visible than other features.
```
X = 30
kernel = np.ones((X,1),np.float32)/X
dst = cv2.filter2D(img_gray,-1,kernel)
kernel = np.ones((1,X),np.float32)/X
dst = cv2.filter2D(dst,-1,kernel)
```
![Step 2](data/steps/step2.jpg)


Finally, QR is simply extracted from the image by 2 lines below.
```
img_gray[dst<50] = 0
mask = np.where(img_gray>0)
```
![Step 3](data/steps/step3.jpg)

The procedure explained above can be used to select both QR and BAR. In order to classify them,
smearing filter is applied. Result of smearing is represented with histograms for both cases.
One can clearly see the difference in distribution of colors and this difference, i.e. variance value of histograms has been used to classify QR and BAR code.

```
# Difference between QR and BAR
kernel = np.ones((int((y2-y1)/2),1),np.float32)/(y2-y1)/2
dst = cv2.filter2D(img_gray[y1:y2, x1:x2],-1,kernel) 

# Bar code
if np.var(dst.flatten()) > 350:
    code_type = 'bar_code'

# QR code
else:
    code_type = 'qr_code'
```
Histogram of QR code after applying 2D filter.   
![qr](data/steps/qr.png)

Histogram of BAR code after applying 2D filter.  
![bar](data/steps/bar.png)


### Text detection

```
X = 15
kernel = np.ones((X, X),np.float32)
img_gray = cv2.filter2D(img_gray,-1,kernel)
```

![Step 4](data/steps/step4.jpg)


## Results

This exercise is an example where classical CV approach can give very good results. It only takes a little bit of parameter tuning. For more realistic dataset, one would need to use different approach with DL.
 
Summary:
- QR and BAR code detection/recognition works perfectly without a single fail on the whole dataset.  
- Text detection works very good as well. However, there is a minor difference between
ground truth bounding boxes and the predicted ones. The reason for that lies in the definition
of the word separator, as it can be seen on the images below. 

An example where predicted bounding box contains two words (SVA934c0 and Zm9) with sufficient space but it failed to separate them. Additional tuning can solve this.   
![Example 2][data/results/LVO4IATd5toqoi.jpg]

An example where according to the ground truth labels, two words (BSQU8ev29 and Cj2) are separated. However, there is insufficient space for word separation and it is not possible to separate this case, not even by human intervention.  
![Example 1][data/results/K3XNE2Cr2rq7s2.jpg]