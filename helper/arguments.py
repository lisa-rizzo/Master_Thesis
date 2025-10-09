
#defines utility functions for retrieving configuration parameters from a dictionary (spec) 

#reads a parameter named attr from the spec directory
def get_parameter(spec: dict, attr: str, default, dtype: type):
    if spec is not None and attr_in_spec(spec, attr):
        attr = get_actual_attr(spec, attr)
        parameter = spec[attr]
        if dtype == bool:
            if isinstance(parameter, str):
                parameter = parameter in ["true", "True", "1"]
            return dtype(parameter)
        else:
            if isinstance(parameter, dtype):
                return parameter
            else:
                return default

    # attr does not exist, return default
    return default

#checks if the attribute attr exists as a key in spec
def attr_in_spec(spec, attr):
    attr = attr.lower()
    return any(k.lower() == attr for k in spec)


#finds and returns the aactual dictionary key from spec matching attr
def get_actual_attr(spec, attr):
    lower_attr = attr.lower()
    for k in spec:
        if k.lower() == lower_attr:
            return k  
    return None  