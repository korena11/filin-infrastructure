import GVF
import numpy as np

if __name__ == '__main__':
    A = np.array([[
         1,  2,  3,  11],
         [4,  5,  6 , 12],
         [7,  8,  9  ,13]])

    b = GVF.boundMirrorExpand(A)
    c = GVF.boundMirrorEnsure(b)
    a = GVF.boundMirrorShrink(c)