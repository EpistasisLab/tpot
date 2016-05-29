# Perform classification with a random forest classifier'
rfc{{operator_num}} = RandomForestClassifier(
    n_estimators={{n_estimators|default(500)}},
    max_features={{operator|max_feature}}),
)

rfc{{operator_num}}.fit(
    {{operator[2]}}.loc[training_indices].drop('class', axis=1).values,
    {{operator[2]}}.loc[training_indices, 'class'].values,
)

{% if operator[0] != operator[2] %}{{operator[0]}} = {{operator[2]}}.copy(){% endif %}
{{operator[0]}}['rfc{{operator_num}}-classification'] = rfc[[operator_num]].predict(
    {{operator[0]}}.drop('class',axis=1).values
)
