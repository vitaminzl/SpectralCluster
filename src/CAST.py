# import numpy as np
# import ROSC
# import preprocess as prep
# import postprocess as postp
#
#
# def getCAST_Z(X, W, alpha1, alpha2):
#     Z = X
#     return Z
#
#
# def CAST(S, k, alpha1, alpha2):
#     W_tknn = ROSC.getTKNN_W(S, K=k)
#     W = np.diag(np.sum(S, axis=0)) @ S
#     X = prep.PIC_k(W, k=k)
#     X = prep.whiten(X)
#     X = prep.norm(X)
#     Z = getCAST_Z(X, W, alpha1, alpha2)
#     Z = (np.abs(Z) + np.abs(Z.T)) / 2
#     C = postp.ncuts(Z, k)
#
#
# def main():
#     data = np.loadtxt("dataset/Syn.txt")
#     label = np.loadtxt("dataset/SynLabel.txt")
#     S = prep.getSimilarMatrix2(data=data)
#     ROSC(S, k=7)
#
#
# if __name__ == "__main__":
#     main()