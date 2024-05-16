from .lexicase_selection import lexicase_selection
from .max_weighted_average_selector import max_weighted_average_selector
from .random_selector import random_selector
from .tournament_selection import tournament_selection
from .tournament_selection_dominated import tournament_selection_dominated
from .nsgaii import nondominated_sorting, crowding_distance, dominates, survival_select_NSGA2
from .map_elites_selection import map_elites_survival_selector, map_elites_parent_selector


SELECTORS =     {"lexicase":lexicase_selection,
                "max_weighted_average":max_weighted_average_selector,
                "random":random_selector,
                "tournament":tournament_selection,
                "tournament_dominated":tournament_selection_dominated,
                "nsgaii":survival_select_NSGA2,
                "map_elites_survival":map_elites_survival_selector,
                "map_elites_parent":map_elites_parent_selector,
                }