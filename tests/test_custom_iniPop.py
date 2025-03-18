from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

individual_str1 = 'MultinomialNB(input_matrix, MultinomialNB__alpha=0.1, MultinomialNB__fit_prior=True)'
individual_str2 = 'GaussianNB(DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=entropy, DecisionTreeClassifier__max_depth=4, DecisionTreeClassifier__min_samples_leaf=17, DecisionTreeClassifier__min_samples_split=13))'
individual_str3 = 'GaussianNB(SelectFwe(CombineDFs(input_matrix, ZeroCount(input_matrix))))'

est = TPOTClassifier(generations=3, population_size=5, verbosity=2, random_state=42, config_dict=None,
                     customized_initial_population=[individual_str1, individual_str2, individual_str3],
                      )
est.fit(X_train, y_train)
print(est.score(X_test, y_test))
