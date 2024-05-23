# -*- coding: utf-8 -*-
"""
@author: jkoo
"""

import pandas as pd
import numpy as np
from pyomo.environ import (Reals, NonNegativeIntegers, Boolean, NonNegativeReals, Binary, SolverFactory, value)
from pyomo.opt import SolverStatus, TerminationCondition
import PDMPC_Evaluator as EV
from PDMPC_formulations import Model_
import pygad

input_path  = "./Inflow_data/"
LV_curve = pd.read_csv(r"LV_curve.csv")

def LtoS(lvl):
    sto = np.interp(lvl, LV_curve['0'], LV_curve['1'])
    return sto

def StoL(sto):
    lvl = np.interp(sto, LV_curve['1'], LV_curve['0'])
    return lvl


# ================================================================================

MT = 264  # the maximun discharge via turbines
MSP = int(11680)  # the maximum discharge via spillway gates
LWL = 60.0 # Low water level
FWL = 80.0 # Flood water level
LWS = int(LtoS(LWL)) # Storage at LWL
FWS = int(LtoS(FWL)) # Storage at FWL

IWL = 76.5  # initial water level
IWS = int(LtoS(IWL)) # initial water storage
TWL = 76.5  # Target water level at the end of the episode
TWL_N = 76.5
TWL_L = 76.0
TWS = int(LtoS(TWL))
TWS_L = int(LtoS(TWL_L))
TWS_N = int(LtoS(TWL_N))

I_QT = 150
I_QSP = 0

# ================================================================================

def _solver(model, data_h):
    inst = model.create_instance(data_h)
    solver = SolverFactory('glpk')
    results = solver.solve(inst)
    return inst, results


def to_weight(solution, F):
    if solution[7] == 0:
        TWS_FF = int(LtoS(78.5))
    elif solution[7] == 1:
        TWS_FF = int(LtoS(79.0))
    elif solution[7] == 2:
        TWS_FF = int(LtoS(79.5))
    elif solution[7] == 3:
        TWS_FF = int(LtoS(79.8))

    w_ = [solution[0] * 20 / (FWS) / F,
          solution[0] * 20 * 2 / (FWS) / F,
          solution[0] * 20 * 20 / (FWS) / F,
          solution[1] * 20 / MSP / F,
          solution[2] * 20 / MSP / F,
          solution[3] * 20 / MSP / F,
          solution[4] * 20 / MSP / F,
          solution[5] * 20 / MSP / F,
          solution[6] * 2 / MSP / F]

    weights = w_ + [TWS_N, TWS_L, TWS_FF]  # decision variables
    return weights


def MPC_main(F, event_nm, Perfect_prediction):

    if Perfect_prediction:
        PP = 'PT'
    else:
        PP = 'PF'

    # load inflow data
    total_data_h_ls = np.load(input_path + "total_data_F{}_E{}_{}.npy".format(F, event_nm, PP), allow_pickle=True)
    total_real_QIN = np.load(input_path + "total_real_QIN_F{}_E{}.npy".format(F, event_nm, PP), allow_pickle=True)
    Qpd = list(total_real_QIN[:, 0]) + list(total_real_QIN[-1][1:])

    # the number of MPC iteration
    max_step = len(total_data_h_ls) - F - 1

    # set the initial state
    I_RWS_t = IWS
    I_QT_t_ls = [I_QT for i in range(0, F)]
    I_QSP_t_ls = [I_QSP for i in range(0, F)]

    ## ====================== start MPC iterations ================================================================


    for k in range(0, max_step):
        data_h = total_data_h_ls[k]
        if k == 0:
            max_INF = MT
        else:
            max_INF = max(max(Qpd[:k]), MT)

        # get evalutor results
        def opt_ga(solution):

            weights = to_weight(solution, F)
            model = Model_(I_QT_t_ls, I_QSP_t_ls, I_RWS_t, weights)

            penalty = 0
            inst, results = _solver(model, data_h)

            if (results.solver.status != SolverStatus.ok):
                penalty += -10000000
            elif results.solver.termination_condition != TerminationCondition.optimal:
                penalty += -10000000

            QTO = [round(value(inst.QTO[i])) for i in inst.t]
            QSP = [round(value(inst.QSP[i])) for i in inst.t]
            QT = [round(value(inst.QT[i])) for i in inst.t]
            RWL = [StoL(round(value(inst.S[i]))) for i in inst.t]

            qin_ = total_real_QIN[k]
            penaltys = EV.evaluator(F, I_QSP_t_ls, qin_, QTO, QSP, QT, RWL, TWL_N, TWL_L, max_INF)
            penalty += penaltys * -1

            return penalty


        fitness_function = opt_ga
        num_generations = 20000
        sol_per_pop = 1000
        function_inputs = [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0], [5, 1, 1, 1, 1, 1, 1, 0], [7, 1, 1, 1, 1, 1, 1, 0],
                            [10, 1, 1, 1, 1, 1, 1, 0], [15, 1, 1, 1, 1, 1, 1, 0], [20, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1],
                            [5, 1, 1, 1, 1, 1, 1, 1], [7, 1, 1, 1, 1, 1, 1, 1], [10, 1, 1, 1, 1, 1, 1, 1], [15, 1, 1, 1, 1, 1, 1, 1],
                            [20, 1, 1, 1, 1, 1, 1, 1], [1, 1, 10, 1, 1, 1, 1, 2], [5, 1, 10, 1, 1, 1, 1, 2], [7, 1, 10, 1, 1, 1, 1, 2],
                            [10, 1, 10, 1, 1, 1, 1, 2], [15, 1, 10, 1, 1, 1, 1, 2], [20, 1, 10, 1, 1, 1, 1, 2]]
        num_genes = len(function_inputs[0])

        if k == 0:
            function_inputs_ = function_inputs
        else:
            function_inputs_ = next_init_input

        initial_population_ = []
        j = 0
        while len(initial_population_) != sol_per_pop:
            initial_population_.append(function_inputs_[j])
            j += 1
            if j >= len(function_inputs_):
                j = 0
        initial_population_ = np.array(initial_population_)
        init_range_low = 1
        init_range_high = 3
        num_parents_mating = int(sol_per_pop/4)
        parent_selection_type = "rws"
        keep_parents = max(int(sol_per_pop/10), 1)
        crossover_type = "single_point"
        mutation_type = "random"
        mutation_percent_genes = 25
        gene_space = [np.linspace(1, 20, 20),
                      np.linspace(1, 20, 20),
                      np.linspace(1, 20, 20),
                      np.linspace(0, 19, 20),
                      np.linspace(0, 19, 20),
                      np.linspace(0, 19, 20),
                      np.linspace(0, 2, 3),
                      np.linspace(0, 2, 3)]

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               initial_population=initial_population_,
                               init_range_low=init_range_low,
                               init_range_high=init_range_high,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               gene_space=gene_space,
                               gene_type=int,
                               stop_criteria=["reach_0", "saturate_10"])

        ga_instance.run()

        # get the best weights and parameter set: solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        next_init_input = ga_instance.population

        # proceed 1 time step
        weights = to_weight(solution, F)
        model = Model_(I_QT_t_ls, I_QSP_t_ls, I_RWS_t, weights)
        inst, results = _solver(model, data_h)

        m_QTO = [round(value(inst.QTO[i])) for i in inst.t]
        m_QSP = [round(value(inst.QSP[i])) for i in inst.t]
        m_QT = [round(value(inst.QT[i])) for i in inst.t]

        next_RWS = I_RWS_t + (total_real_QIN[k][0] - m_QTO[0]) * 3600

        # new state for the next iteration
        I_RWS_t = next_RWS
        I_QT_t_ls = m_QT
        I_QSP_t_ls = m_QSP