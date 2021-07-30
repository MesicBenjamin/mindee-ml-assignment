import argparse

import cv2 
import numpy as np
import matplotlib.pyplot as plt

def get_code_candidate(img):
    
    '''For given img gives code candidate'''

    # QR and BAR code case
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = 255 - img_gray # invert grayscale

    # cv2.imwrite('step1.jpg', img_gray)

    # Filter everything but code
    X = 30
    kernel = np.ones((X,1),np.float32)/X
    dst = cv2.filter2D(img_gray,-1,kernel)
    kernel = np.ones((1,X),np.float32)/X
    dst = cv2.filter2D(dst,-1,kernel)

    # cv2.imwrite('step2.jpg', dst)

    img_gray[dst<50] = 0
    mask = np.where(img_gray>0)

    # cv2.imwrite('step3.jpg', img_gray)

    if len(mask[0])==0:
        return 

    # Find corners of bounding box
    y1 = np.min(mask[0])
    x1 = np.min(mask[1])
    y2 = np.max(mask[0])
    x2 = np.max(mask[1])

    # # cv2.rectangle(img,(x1,y1),(x2,y2),(0,0, 255),2)
    # cv2.imshow("Show",img_gray)
    # cv2.waitKey()

    # Difference between qr and bar
    kernel = np.ones((int((y2-y1)/2),1),np.float32)/(y2-y1)/2
    dst = cv2.filter2D(img_gray[y1:y2, x1:x2],-1,kernel) 

    # print(np.var(dst.flatten()))
    # plt.hist(dst.flatten(), bins=25)
    # plt.show()

    # Bar code
    if np.var(dst.flatten()) > 350:
        code_type = 'bar_code'

    # QR code
    else:
        code_type = 'qr_code'

    candidate = {
        'type': code_type,
        'geometry': [x1, y1, x2, y2],
    }

    return [candidate]

def get_text_candidates(img):

    '''For given img gives list of text candidates'''
    candidates = []

    # QR and BAR code case
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = 255 - img_gray # invert grayscale

    img_gray[900:,:] = 0

    # Filter code
    X = 30
    kernel = np.ones((X,1),np.float32)/X
    dst = cv2.filter2D(img_gray,-1,kernel)
    kernel = np.ones((1,X),np.float32)/X
    dst = cv2.filter2D(dst,-1,kernel)

    img_code = img_gray.copy()
    img_code[dst<50] = 0
    mask = np.where(img_code>50)

    if not len(mask[0])==0:

        # Find corners of bounding box
        y1 = np.min(mask[0])
        x1 = np.min(mask[1])
        y2 = np.max(mask[0])
        x2 = np.max(mask[1])

        img_gray[y1-2:y2+2, x1-2:x2+2] = 0

    _, img_gray = cv2.threshold(img_gray, 25, 255, 0)

    # Blur text for extracting contours
    X = 15
    kernel = np.ones((X, X),np.float32)
    img_gray = cv2.filter2D(img_gray,-1,kernel)

    # cv2.imshow('res', img_gray) 
    # cv2.waitKey()

    # cv2.imwrite('step4.jpg', img_gray)

    # -------------------------        
    # Find label candidates
    contours, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for n,c in enumerate(contours):

        x,y,w,h = cv2.boundingRect(c)

        candidate = {
            'type': 'text',
            'geometry': [x, y, x+w, y+h],
        }

        candidates.append(candidate)

        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.rectangle(img_gray,(x,y),(x+w,y+h),255,2)

    # cv2.imshow('res', img_gray) 
    # cv2.waitKey()

    return candidates

def get_candidates(img):
    
    '''For given img gives list of label candidates'''

    candidates = []

    code_canidate = get_code_candidate(img)

    if code_canidate is not None:

        candidates += code_canidate

    text_candidates = get_text_candidates(img)

    if text_candidates is not None:

        candidates += text_candidates

    return candidates

if __name__ == '__main__':

    # ----------------------------------------
    parser = argparse.ArgumentParser(description='Label prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name_img', type=str, help='image name', default='data/0A1NOPtu573ak.jpg')
    parser.add_argument('--draw', type=bool, help='Draw and save image', default=False)

    args = parser.parse_args()

    # ------------------

    img = cv2.imread(args.name_img)

    # ------------------

    # Candidates is the final list of resulting bounding boxes as required by exercise
    candidates = get_candidates(img)

    print(candidates)

    # ------------------
    # Draw
    if args.draw:

        colors = {'text':(0,255,0), 'qr_code':(255, 0, 0), 'bar_code':(0,255, 255)}

        for c in candidates:

            c_type = str(c['type'])

            x1,y1,x2,y2= c['geometry']
            cv2.rectangle(img,(x1,y1),(x2,y2),colors[c_type],2)
            cv2.putText(img, c_type, (x2+10,y2),0,0.3, colors[c_type])

        # cv2.imshow("Show",img)
        # cv2.waitKey()  
        cv2.imwrite('result.jpg', img)