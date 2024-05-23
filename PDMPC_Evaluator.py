# -*- coding: utf-8 -*-
"""
@author: jkoo
"""


def evaluator(F, I_QSP, QIN, QTO, QSP, QT, RWL, TWL_N, TWL_L, max_INF):

    MT = 264  # the maximun discharge via turbines
    FWL = 80.0   # Flood water level

    def compl():
        penalty = 0
        for i in range(F):
            if QSP[i] > 0 and QT[i] < MT - 1:
                penalty += 500000
        return penalty

    def num_C():
        num_C_ls = []
        for j in range(1, F):
            if abs(QSP[j - 1] - I_QSP[j]) > 1:
                num_C_ls.append(1)
            else:
                num_C_ls.append(0)
        w__ = [1, 0.5, 0.3, 0.2, 0] + [0 for i in range(25)]
        penalty = sum([num_C_ls[k] * w__[k] for k in range(len(num_C_ls))]) / 2
        return penalty, num_C_ls


    def num_C_inc():
        num_C_ls = []
        for j in range(1, F):
            if QSP[j - 1] - I_QSP[j] > 1:
                num_C_ls.append(1)
            else:
                num_C_ls.append(0)
        w__ = [1, 0.5, 0.3, 0.2, 0] + [0 for i in range(25)]
        penalty = sum([num_C_ls[k] * w__[k] for k in range(len(num_C_ls))]) / 2
        return penalty, num_C_ls


    def num_C_inside():
        num_C_ls = []
        for j in range(F-1):
            if QSP[j+1] - QSP[j] > 1:
                num_C_ls.append(1)
            else:
                num_C_ls.append(0)
        w__ = [1, 0.5, 0.3, 0.2, 0] + [0 for i in range(25)]
        penalty = sum([num_C_ls[k] * w__[k] for k in range(len(num_C_ls))])
        return penalty


    def Max_QTO_():
        penalty = round(2 ** (max(QSP) / 1000), 1) * 0.9 + round(2 ** (sum(QSP) / F / 1000), 1) * 0.1
        if max(QTO) > max_INF:
            penalty += 1000
        return penalty


    def Max_RWL_():
        penalty = round(2 ** max(max(RWL) - TWL_N, 0), 1) * 1 + round(2 ** max(RWL[-1] - RWL[0], 0), 1) * 1 + \
                  round(2 ** max(TWL_L - min(RWL), 0), 1) * 2
        if max(RWL) > FWL - 0.1:
            penalty += 1000
        return penalty


    def num_C_oc():
        num_C_o = 0
        num_C_c = 0
        C_oc = []

        for i in range(1, F):
            QSP_ = int(QSP[i]-0.1)
            try:
                Bi = QSP_/QSP_
            except:
                Bi = 0
            C_oc.append(Bi)

        for j in range(1, len(C_oc)):
            num_C_c += abs(min(C_oc[j] - C_oc[j - 1], 0))
            num_C_o += abs(max(C_oc[j] - C_oc[j - 1], 0))
        penalty = (2 ** (num_C_o) + 2 ** (num_C_c)) / 2
        return penalty

    penalty_1 = compl()
    penalty_2, _ = num_C_inc()
    penalty_2_in = num_C_inside()
    penalty_3 = Max_QTO_()
    penalty_4 = Max_RWL_()
    penalty_5 = num_C_oc()

    penaltys = penalty_1 + penalty_2 + penalty_2_in*0.1 + penalty_3 + penalty_4 + penalty_5

    return penaltys