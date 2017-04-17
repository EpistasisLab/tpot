import numpy as np

classifier_config_dict = {

    # Classifiers
    
    'mdr.MDR': {
    'tie_break': tie_break % 2
    'default_label': default_label % 2
    },
    
    # Feature Selectors
    
    'skrebate.ReliefF': {
        'n_features_to_select': max(1, n_features_to_select)
    },
        
    'skrebate.SURF': {
        'n_features_to_select': max(1, n_features_to_select)
    },
        
    'skrebate.SURFstar': {
        'n_features_to_select': max(1, n_features_to_select)
    },
        
    'skrebate.MultiSURF': {
        'n_features_to_select': max(1, n_features_to_select)
    }

}
