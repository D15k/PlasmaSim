
# Contains generic utility functions for the PlasmaSim package.

'''
def copy_attrs(target, source, exclude_prefix="__"):
    for attr in dir(source):
        if not attr.startswith(exclude_prefix) and not callable²(getattr(source, attr)):
            setattr(target, attr, getattr(source, attr))
'''


def copy_attrs(source, target, exclude_prefix='__'):
    '''
    Copy the attributes from source object to target object, excluding attributes starting with the given prefixes.
    
    Args:
        source: The source object to copy the attributes from.
        target: The target object to copy the attributes to.
        exclude_prefix: The prefixes of the attributes to exclude from the copy. Default is '__'.
    '''
    
    if isinstance(exclude_prefix, str):
        exclude_prefix = [exclude_prefix]
        
    for key in dir(source):
        if not any(key.startswith(prefix) for prefix in exclude_prefix):
            setattr(target, key, getattr(source, key))