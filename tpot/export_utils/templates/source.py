{% block "imports" scoped %}{% endblock %}

{% for operator in pipeline_list %}
{% set operator_num = int(operator[0].strip('result')) %}
{% for i in [2,3] %}{% if operator[i] in ['ARG0'] %}
result{{operator_num}} = tpot.copy()
{% endif %}{% endfor %}
{% include operator[1]+'.py' %}
{% endfor %}
