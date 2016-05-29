{% extends source.py %}

{% block imports %}
import numpy as np
import pandas as pd

{% for module_name in pipeline_imports.keys() %}
from {{module_name}} import {{pipeline_imports[module_name] | join(', ')}}
{% endfor %}

{% raw %}
# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv("{{data_source | default('PATH/TO/DATA/FILE')}}", sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data[{{class_column|default('class')}}].values, train_size={{train_size|default(0.75)}}, test_size={{1-train_size|default(0.25)}})
{% endraw %}
{% endblock %}
