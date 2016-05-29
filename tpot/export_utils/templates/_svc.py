# Perform classification with a logistic regression classifier
svc{{operator_num}} = LogisticRegression(C={{operator[3] if not operator[3] <= 0. else .0001}})

svc{{operator_num}}.fit(
    {{operator[2]}}.loc[training_indices].drop('class', axis=1).values,
    {{operator[2]}}.loc[training_indices, 'class'].values,
)

{% if operator[0] != operator[2] %}{{operator[0]}} = {{operator[2]}}.copy(){% endif %}
{{operator[0]}}['svc{{operator_num}}-classification'] = svc[[operator_num]].predict(
    {{operator[0]}}.drop('class',axis=1).values
)
