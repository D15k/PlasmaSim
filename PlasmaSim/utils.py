import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Contains generic utility functions for the PlasmaSim package.


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


def extract_attrs(source, attribute):
    '''
    Extract a specific attribute from the source object(s) and return the values as a jax array.

    Args:
        source: The source object(s) to extract the attribute from (can be a single object or list of objects).
        attribute: The name of the attribute to extract from each object.
    '''
    
    if not isinstance(source, list):
        source = [source]
        
    attribute_values = jnp.array([getattr(item, attribute) for item in source])
    
    return attribute_values