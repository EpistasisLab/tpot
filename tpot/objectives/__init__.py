from .average_path_length import average_path_length_objective
from .number_of_nodes import number_of_nodes_objective
from .number_of_leaves import number_of_leaves_scorer, number_of_leaves_objective
from .complexity import complexity_scorer


#these scorers are calculated per fold of CV on the fitted pipeline for that fold
SCORERS =       {   
                    "complexity_scorer": complexity_scorer
                }

#these objectives are calculated once on unfitted models as secondary objectives
OBJECTIVES =    {  "average_path_length_objective": average_path_length_objective,
                    "number_of_nodes_objective": number_of_nodes_objective,
                    "number_of_leaves_objective": number_of_leaves_objective
                }