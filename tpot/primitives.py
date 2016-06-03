from deap import (
    tools
)
from pandas import (
    np,
)


def get_fitness_attr(x): return x.fitness.values[1]


def _div(num1, num2):
    return float(num1) / float(num2) if num2 != 0. else 0.


def _combined_selection_operator(individuals, k):
    return tools.selNSGA2(
        individuals, int(k / 5.)
    ) * 5


def _combine_dfs(input_df1, input_df2):
    return input_df1.join(input_df2)


def _zero_count(input_df):
    modified_df = input_df.copy()
    modified_df['non_zero'] = modified_df.apply(
        np.count_nonzero, axis=1
    ).astype(np.float64)
    modified_df['zero_col'] = len(modified_df)-modified_df['non_zero']
    return modified_df.copy()
