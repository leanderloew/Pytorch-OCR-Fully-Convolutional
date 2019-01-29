import torch
import numpy
import matplotlib.pyplot as plt

    
def wer(r, h):
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def preds_to_integer(preds):
    preds=torch.argmax(preds,dim=1).detach().cpu().numpy()
    preds=preds.tolist()

    out=[]
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != preds[i - 1]:
            out.append(preds[i])
    return out 

def wer_eval(preds,labels):
    preds=preds_to_integer(preds)
    we=wer(preds,labels)
    return we/len(preds)

def show(img):
    npimg = img.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    
