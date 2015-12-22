import hashlib

class BasicOperator(object):
    def __init__(self, operation_object, intypes, outtype, import_code, callable_code):
        self.operation_object     = operation_object
        self.intypes       = intypes
        self.outtype       = outtype
        self.import_code   = import_code
        self._callable_code = callable_code
    def object_alias(self, operator_num):
        return self.__class__.__name__.lower() + str(operator_num)
    def callable_lhs(self, operator_num, operator, result_name):
        """Generates the left hand side of the 'callable_code' chunk"""
        return self.object_alias(operator_num)
    def callable_code(self, operator_num, operator, result_name):
        operator_text = self.codeblock_comment
        operator_text += "{} = ".format(self.callable_lhs(operator_num, operator, result_name))
        if len(self.intypes) > 1:
            # 0th argument will be the input dataframe, all subsequent args are input params
            operator_text += self._callable_code.format(*operator[2:]) 
        else:
            operator_text += self._callable_code
        operator_text +='\n'
        return operator_text
    def preprocess_arguments(self, input_df, *args, **kargs):
        return input_df, args, kargs
    def evaluate_operator(self, input_df, *args, **kargs):  
        input_df, args, kargs = self.preprocess_arguments(input_df, args, kargs)
        return self.operation_object(input_df, *args, **kargs)
    @property
    def codeblock_comment(self):
        return '\n\n# Run {}\n'.format(self.__class__.__name__)
        
        
class LearnerOperator(BasicOperator):
    def evaluate_operator(self, input_df, *args, **kwargs):
        return self._train_model_and_predict(input_df, *args, **kwargs)
    @property
    def codeblock_comment(self):
        return '\n# Run prediction step with a {} model\n'.format(self.__class__.__name__)
    def callable_code(self, operator_num, operator, result_name):
        """
        NB: The first line of the output will be the evaluated `callable_code` 
        string passed in when the class was initialized. The format operation
        expands the `operator` input from the second argument out, such that
        if see __init__(callable_code='{0},{1}'), the first line of the output 
        will be along the lines of 'input_df_name, parameter0'
        """
        operator_text = super(LearnerOperator, self).callable_code(operator_num, operator, result_name)
        operator_text += '''{0}.fit({1}.loc[training_indices].drop('class', axis=1).values, {1}.loc[training_indices, 'class'].values)\n'''.format(self.object_alias(operator_num), operator[2])
        if result_name != operator[2]:
            operator_text += '{} = {}\n'.format(result_name, operator[2]) # Should this be '{} = {}.copy()\n' ?
        operator_text += '''{0}['{1}-classification'] = {1}.predict({0}.drop('class', axis=1).values)\n'''.format(result_name, self.object_alias(operator_num))
        return operator_text
    # _train_model_and_predict scavenged en toto from current tpot.TPOT implementation
    def _train_model_and_predict(self, input_df, *args, **kwargs):
        """Fits an arbitrary sklearn classifier model with a set of keyword parameters
        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the k-neares
        model: sklearn classifier
            Input model to fit and predict on input_df
        kwargs: unpacked parameters
            Input parameters to pass to the model's constructor, does not need to be a dictionary
        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.
        """
        model = self.operation_object
        
        #Validate input
        #If there are no features left (i.e., only 'class', 'group', and 'guess' remain in the DF), then there is nothing to do
        if len(input_df.columns) == 3:
            return input_df

        input_df = input_df.copy()

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1).values
        training_classes = input_df.loc[input_df['group'] == 'training', 'class'].values
       
        # Try to seed the random_state parameter if the model accepts it.
        try:
            clf = model(random_state=42,**kwargs)
            clf.fit(training_features, training_classes)
        except TypeError:
            clf = model(**kwargs)
            clf.fit(training_features, training_classes)
        
        all_features = input_df.drop(['class', 'group', 'guess'], axis=1).values
        input_df.loc[:, 'guess'] = clf.predict(all_features)
        
        # Also store the guesses as a synthetic feature
        #sf_hash = '-'.join(sorted(input_df.columns.values))
        sf_hash = '-'.join(str(c) for c in sorted(input_df.columns.values)) # not clear to me why this is necessary running operators_test.py
        #Use the classifier object's class name in the synthetic feature
        sf_hash += '{}'.format(clf.__class__)
        sf_hash += '-'.join(kwargs)
        sf_identifier = 'SyntheticFeature-{}'.format(hashlib.sha224(sf_hash.encode('UTF-8')).hexdigest())
        input_df.loc[:, sf_identifier] = input_df['guess'].values

        return input_df