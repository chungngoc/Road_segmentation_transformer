import numpy as np

class IOUEval:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.num_classes, dtype=np.float32)
        self.per_class_iu = np.zeros(self.num_classes, dtype=np.float32)
        self.mIoU = 0
        self.batchCount = 1
    
    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.num_classes)
        c = self.num_classes * a[k].astype(int) + b[k]
        return np.bincount(c, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
    
    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        epsilon = 1e-8
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIoU = np.nanmean(per_class_iu)

        self.overall_acc += overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIoU += mIoU
        self.batchCount += 1

    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        mIoU = self.mIoU / self.batchCount

        return overall_acc, per_class_acc, per_class_iu, mIoU