# Perform classification with a decision tree classifier
dtc{{operator_num}} = DecisionTreeClassifier(
    max_features={{operator|max_feature}},
    max_depth={{operator[4]|default('None')}},
)

dtc{{operator_num}}.fit(
    {{operator[2]}}.loc[training_indices].drop('class', axis=1).values,
    {{operator[2]}}.loc[training_indices, 'class'].values,
)

{% if operator[0] != operator[2] %}{{operator[0]}} = {{operator[2]}}.copy(){% endif %}
{{operator[0]}}['dtc{{operator_num}}-classification'] = dtc[[operator_num]].predict(
    {{operator[0]}}.drop('class',axis=1).values
)
