from deap import (
    tools
)
from pandas import (
    np, concat
)


def get_fitness_attr(x): return x.fitness.values[1]


def _div(num1, num2):
    return float(num1) / float(num2) if num2 != 0. else 0.


def _combined_selection_operator(individuals, k):
    return tools.selNSGA2(
        individuals, int(k / 5.)
    ) * 5


def _combine_dfs(input_df1, input_df2):
    if input_df1.equals(input_df2):
        return input_df1
    return concat(
            [input_df1, input_df2], axis=1
        ).set_index(
            input_df1.index
        )

def _zero_count(df):
    df['non_zero'] = df.apply(
        np.count_nonzero, axis=1
    ).astype(np.float64)
    df['zero_col'] = len(df)-df['non_zero']
    return df
