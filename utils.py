import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def list2str(lst, decimals, separator=','):
    """
    convert a list of float into string with defined decimals and separator
    """
    return separator.join(map(lambda x: '{:.{}f}'.format(x, decimals), lst))


def one_hot(x, num_ratings=None):
    if num_ratings is None:
        num_ratings = int(max(x)) + 1
    return np.eye(num_ratings)[x]


def _fast_mat(label_true, label_pred, n_class=2):
    mat = np.bincount((n_class * label_true.astype(np.int) + label_pred.astype(np.int)).flatten(),
                       minlength=n_class ** 2).reshape(n_class, n_class)
    return mat



class Measurement(object):
    @staticmethod
    def conf_mats(labels, preds):
        labels = np.array(labels)
        preds = np.array(preds)

        assert preds.shape == labels.shape
        n_class = labels.shape[1]
        mats = np.zeros((n_class, 2, 2))
        for i in range(n_class):
            mats[i] = _fast_mat(labels[:, i].flatten(), preds[:, i].flatten())
        return mats

    @staticmethod
    def conf_mat_based_measurements(mats, e=1e-10):
        n_class = mats.shape[0]
        precisions = np.zeros(n_class)
        recalls = np.zeros(n_class)
        fs = np.zeros(n_class)
        specificities = np.zeros(n_class)
        for i in range(n_class):
            mat = mats[i]
            tp = mat[1, 1]
            tn = mat[0, 0]
            fp = mat[0, 1]
            fn = mat[1, 0]

            precisions[i] = tp / (tp + fp + e)
            recalls[i] = tp / (tp + fn + e)
            fs[i] = 2 * tp / (2 * tp + fp + fn + 2 * e)
            specificities[i] = tn / (tn + fp + e)

        return precisions, recalls, fs, specificities

    @staticmethod
    def rank_based_measurements(scores, labels):
        assert scores.shape == labels.shape
        im_num, label_num = scores.shape

        aps = np.zeros(label_num)
        iaps = np.zeros(im_num)
        aucs = np.zeros(label_num)

        for i in range(label_num):
            score = scores[:, i]
            target = labels[:, i]
            if (target == 0).all() or (target == 1).all():
                ap = np.nan
                auc = np.nan
            else:
                ap = average_precision_score(target, score)
                auc = roc_auc_score(target, score)
            aps[i] = ap
            aucs[i] = auc
        for i in range(im_num):
            score = scores[i, :]
            target = labels[i, :]
            if (target == 0).all() or (target == 1).all():
                ap = np.nan
            else:
                ap = average_precision_score(target, score)
            iaps[i] = ap

        return aps, iaps, aucs


class Evaluater(Measurement):
    def evaluate(self, scores, labels, thre=0):
        preds = scores.copy()
        preds[preds>thre] = 1
        preds[preds<=thre] = 0
        preds = preds.astype(np.int)
        labels = labels.astype(np.int)

        mat = self.conf_mats(labels, preds)
        precisions, recalls, fs, specificities = self.conf_mat_based_measurements(mat)
        aps, iaps, aucs = self.rank_based_measurements(scores, labels)

        return mat, precisions, recalls, fs, specificities, aps, iaps, aucs

    @staticmethod
    def best_f1(scores, labels):
        n_class = scores.shape[1]

        senss = []
        specs = []
        thress = []
        f1s = []
        
        for i in range(n_class):
            b_spec = -1
            b_sens = -1
            b_f1 = -1
            b_thre = -1
            fprs, tprs, thres = roc_curve(labels[:, i], scores[:, i])

            for fpr, tpr, thre in zip(fprs, tprs, thres):
                sens = tpr
                spec = 1 - fpr
                f1 = 2 * sens * spec / (sens + spec + 1e-10)
                if f1 > b_f1:
                    b_f1 = f1
                    b_sens = sens
                    b_spec = spec
                    b_thre = thre
            senss.append(b_sens)
            specs.append(b_spec)
            f1s.append(b_f1)
            thress.append(b_thre)
        return senss, specs, thress, f1s