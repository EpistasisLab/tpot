from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal



#MDR
MDR_configspace = ConfigurationSpace(
    space = {
        'tie_break': Categorical('tie_break', [0,1]),
        'default_label': Categorical('default_label', [0,1]),
    }
)




def get_skrebate_ReliefF_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
            'n_neighbors': Integer('n_neighbors', bounds=(2,500), log=True),
        }
    )


def get_skrebate_SURF_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
        }
)


def get_skrebate_SURFstar_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
        }
)
def get_skrebate_MultiSURF_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
        }
)
