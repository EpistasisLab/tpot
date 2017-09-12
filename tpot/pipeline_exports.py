import uuid
import re
def build_pipeline_js(step,storage,func_dict):
    if type(step).__name__ == 'list':
        for item in step:
            build_pipeline_js(item,storage,func_dict)
    else:
        func_name = step[1].__class__.__name__
        lib = step[1].__module__ + '.' + func_name
        obj = {'name':func_name, 'obj_type':'list', 'items':[], 'func':'', 'lib':lib}
        if func_name == 'FeatureUnion':
            storage.append(obj)
            build_pipeline_js(step[1].transformer_list,obj['items'],func_dict)

        elif func_name == 'Pipeline':
            storage.append(obj)
            build_pipeline_js(step[1].steps,obj['items'],func_dict)
        else:
            uid = uuid.uuid4().hex[:6].upper()
            func_name = step[0] + '_' + uid
            param_list = step[1].get_params()

            #clean any func objects in params
            for param in param_list:
                if 'func' in param:
                    tmp = str(param_list[param])
                    tmp = re.sub(" at+\s+\w+>","",tmp)
                    tmp = re.sub("<function ","",tmp)
                    param_list[param] = tmp

            obj = {'name':step[0], 'obj_type':'algorithm', 'items':[], 'func':func_name, 'lib':lib}
            #look for estimator and format as tuple
            if 'estimator' in param_list.keys():
                estimator_name = "estimator_" + uid
                tup = (estimator_name,param_list['estimator'])
                build_pipeline_js(tup,obj['items'],func_dict)
                #add func key to parent object
                estimator_func = obj['items'][0]['func']
                param_list['estimator'] = estimator_func
                func_dict[func_name] = {'name':step[1].__class__.__name__,'params':param_list}
            else:
                func_dict[func_name] = {'name':step[1].__class__.__name__,'params':param_list}

            storage.append(obj)

            return storage

def serialize_to_js(steps,storage,func_dict):
    obj = {'name':'Pipeline', 'obj_type':'list', 'items':[], 'func':'','lib':'sklearn.pipeline.Pipeline'}
    storage.append(obj)
    build_pipeline_js(steps,obj['items'],func_dict)
    return storage

def collect_feature_list(pipeline,features,target):
    feature_list = []
    for step in pipeline:
        if step[1].__class__.__name__ == 'FeatureUnion':
            transformer_list = step[1].transformer_list
            for transformer in transformer_list:
                if "get_support" in dir(transformer[1]):
                    fit_step = transformer[1].fit(features,target)
                    feature_list.append(fit_step.get_support().tolist())

        if "get_support" in dir(step[1]):
            fit_step = step[1].fit(features,target)
            feature_list.append(fit_step.get_support().tolist())
    return feature_list
