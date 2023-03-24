import tpot2.evolutionary_algorithms.helpers as helpers
import numpy as np
import tpot2
from tpot2.evolutionary_algorithms.eaGeneric import eaGeneric_Evolver
from tpot2.evolutionary_algorithms import survival_select_NSGA2
from tpot2.evolutionary_algorithms.parent_selectors import TournamentSelection, TournamentSelection_Dominated

class eaNSGA2_Evolver(eaGeneric_Evolver):
    def __init__(self, 
                        survival_selector = survival_select_NSGA2,
                        parent_selector = TournamentSelection_Dominated,
                        **kwargs,
                        ):

        super().__init__(survival_selector = survival_selector,
                        parent_selector = parent_selector,
                         **kwargs)

