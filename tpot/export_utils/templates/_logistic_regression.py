C = float(operator[3])
if C <= 0.:
    C = 0.0001

# Perform classification with a logistic regression classifier
operator_text += '\nlrc{OPERATOR_NUM} = LogisticRegression(C={C})\n'.format(OPERATOR_NUM=operator_num, C=C)
operator_text += ('''lrc{OPERATOR_NUM}.fit({INPUT_DF}.loc[training_indices].drop('class', axis=1).values, '''
                  '''{INPUT_DF}.loc[training_indices, 'class'].values)\n''').format(OPERATOR_NUM=operator_num, INPUT_DF=operator[2])
if result_name != operator[2]:
    operator_text += '{OUTPUT_DF} = {INPUT_DF}.copy()\n'.format(OUTPUT_DF=result_name, INPUT_DF=operator[2])
operator_text += ('''{OUTPUT_DF}['lrc{OPERATOR_NUM}-classification'] = lrc{OPERATOR_NUM}.predict('''
                  '''{OUTPUT_DF}.drop('class', axis=1).values)\n''').format(OUTPUT_DF=result_name,
                                                                            OPERATOR_NUM=operator_num)

# Perform classification with a decision tree classifier
lrc{{operator_num}} = LogisticRegression(C={{operator[3] if not operator[3] <= 0. else .0001}})

lrc{{operator_num}}.fit(
    {{operator[2]}}.loc[training_indices].drop('class', axis=1).values,
    {{operator[2]}}.loc[training_indices, 'class'].values,
)

{% if operator[0] != operator[2] %}{{operator[0]}} = {{operator[2]}}.copy(){% endif %}
{{operator[0]}}['lrc{{operator_num}}-classification'] = lrc[[operator_num]].predict(
    {{operator[0]}}.drop('class',axis=1).values
)
