from .average_path_length_objective import average_path_length_objective
from .number_of_nodes_objective import number_of_nodes_objective
from .number_of_leaves_scorer import number_of_leaves_scorer, number_of_leaves_objective
from .complexity_objective import complexity_scorer


#these scorers are calculated per fold of CV on the fitted pipeline for that fold
SCORERS =       {   "number_of_leaves": number_of_leaves_scorer,
                    "complexity": complexity_scorer
                }

#these objectives are calculated once on unfitted models as secondary objectives
OBJECTIVES =    {  "average_path_length": average_path_length_objective,
                    "number_of_nodes": number_of_nodes_objective,
                }