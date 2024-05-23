# -*- coding: utf-8 -*-
"""
@author: jkoo
"""

from pyomo.environ import (AbstractModel, Constraint, Reals, NonNegativeIntegers, Boolean, Param, Var,
                           NonNegativeReals, Objective, Set, minimize, Binary)

# ================================================================================

MT = 264  # the maximun discharge via turbines
MSP = int(11680)  # the maximum discharge via spillway gates
LWS = 385583000 # storage at LWL
FWS = 1408538000 # storage at FWL


def Model_(I_QT, I_QSP, I_RWS, weights):
    model = AbstractModel()

    model.t = Set()
    model.QIN = Param(model.t)  # inflow input
    model.Md = Param(model.t)  # demand input

    model.QTO = Var(model.t, domain=NonNegativeReals, bounds=(0, (MT + MSP)), initialize=(I_QT[1] + I_QSP[1]))
    model.QT = Var(model.t, domain=NonNegativeReals, bounds=(0, MT), initialize=I_QT[1])
    model.QSP = Var(model.t, domain=NonNegativeReals, bounds=(0, MSP), initialize=I_QSP[1])
    model.S = Var(model.t, domain=NonNegativeReals, bounds=(LWS, FWS), initialize=I_RWS)
    model.d_S1 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.d_S2 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.d_S3 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.d_QTO1 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.d_QTO2 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.d_QTO3 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.d_QTO4 = Var(model.t, within=NonNegativeReals, initialize=0)
    model.max_QO = Var(within=NonNegativeReals, initialize=I_QSP[1])
    model.Obj_v = Var(model.t, within=NonNegativeReals, initialize=0)

    # # ======================================= essential parts for all MPC formulations ==================

    weight_1 = weights[0]
    weight_2 = weights[1]
    weight_3 = weights[2]
    weight_4 = weights[3]
    weight_5 = weights[4]
    weight_6 = weights[5]
    weight_7 = weights[6]
    weight_8 = weights[7]
    weight_9 = weights[8]
    TWS_N = weights[9]
    TWS_L = weights[10]
    TWS_F = weights[11]
    F = len(model.QIN)

    # # ======================================= system dynamic and default constraints ==================
    # System dynamics
    def _S1(model, t):
        if t == 0:
            return model.S[t] - I_RWS - (model.QIN[t] - model.QTO[t]) * 3600 == 0
        return model.S[t] - model.S[t - 1] - (model.QIN[t] - model.QTO[t]) * 3600 == 0

    model.storage1 = Constraint(model.t, rule=_S1)

    # Hard constraint on RWL to be between LWS and FWS (same as the bound)
    def _S2(model, t):
        return (LWS, model.S[t], FWS)

    model.storage2 = Constraint(model.t, rule=_S2)

    # #definition of QT
    def QT_(model, t):
        return model.QTO[t] - model.QSP[t] - model.QT[t] == 0

    model.QT_ = Constraint(model.t, rule=QT_)

    # Hard constraint that turbine and spill flow must be above demand:
    def Demand_meet(model, t):
        return model.QT[t] - model.Md[t] >= 0

    model.demanddown = Constraint(model.t, rule=Demand_meet)

    # Hard constraint to stick previously determined QSP and QT
    def Stick_first_QSP(model, t):
        if t==0:
            return model.QSP[0] - I_QSP[1] == 0
        else:
            return model.QSP[t] >= 0

    model.stick_first_qsp_ = Constraint(model.t, rule=Stick_first_QSP)

    def Stick_first_QT(model, t):
        if t==0:
            return model.QT[0] - I_QT[1] == 0
        else:
            return model.QT[t] >= 0

    model.stick_first_qt_ = Constraint(model.t, rule=Stick_first_QT)


    # # ======================================= different parts for each MPC formulation ==================

    # # ============  relating to RWL

    # keep RWL following TWL as musch as possible
    def Gap_S_1(model, t):
        return model.S[t] - TWS_N - model.d_S1[t] <= 0

    model.Gap_S_1 = Constraint(model.t, rule=Gap_S_1)

    def Gap_S_2(model, t):
        return TWS_L - model.S[t] - model.d_S2[t] <= 0

    model.Gap_S_2 = Constraint(model.t, rule=Gap_S_2)

    def Gap_S_3(model, t):  # min model.d_S1[t]
        return model.S[t] - TWS_F - model.d_S3[t] <= 0

    model.Gap_S_3 = Constraint(model.t, rule=Gap_S_3)


    # # ============  relating to changes in outflows

    # minimize changes in outflow schedules
    def QTO_1(model, t):  # min model.d_QSP1[t]
        if t == 0:
            return model.d_QTO1[t] + model.d_QTO2[t] == 0
        elif t < 4:
            return (model.QTO[t] - I_QSP[int(t) + 1] - I_QT[int(t) + 1]) / int(t) + model.d_QTO1[t] - model.d_QTO2[t] == 0
        elif t < F-1:
            return (model.QTO[t] - I_QSP[int(t) + 1] - I_QT[int(t) + 1]) / (int(t)*2) + model.d_QTO1[t] - model.d_QTO2[t] == 0
        else:
            return model.d_QTO1[t] + model.d_QTO2[t] == 0

    model.QTO_1 = Constraint(model.t, rule=QTO_1)


    # minimize changes in outflows in a prediction horizon
    def QTO_3(model, t):  # min model.d_QSP3[t]
        if t == 0:
            return model.d_QTO3[t] + model.d_QTO4[t] == 0
        else:
            return ((model.QTO[t] - model.QTO[t - 1])/(int(t)*3)) + model.d_QTO3[t] - model.d_QTO4[t] == 0

    model.QTO_3 = Constraint(model.t, rule=QTO_3)


    # # ============  relating to the peak outflow

    # minimize the peak spillway outflow
    def QO_max(model, t):  # min model.max_QO
        return model.QSP[t] - model.max_QO <= 0

    model.QO_max = Constraint(model.t, rule=QO_max)

    # # ======================================= The formulation of the objective function ==================

    # defining objective variable by combining every variable
    def Obj_var(model, t):
        return model.Obj_v[t] == model.d_S1[t] * weight_1 + model.d_S2[t] * weight_2 + model.d_S3[t] * weight_3 \
                + model.d_QTO1[t] * weight_4 + model.d_QTO2[t] * weight_5 \
                + model.d_QTO3[t] * weight_6 + model.d_QTO4[t] * weight_7\
                + model.max_QO * weight_8 + model.QSP[t] * weight_9

    model.Obj_var = Constraint(model.t, rule=Obj_var)

    def obj_f(model, t):
        return sum(model.Obj_v[i] for i in model.t)

    model.obj = Objective(rule=obj_f, sense=minimize)

    return model