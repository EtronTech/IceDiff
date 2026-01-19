import os
import yaml

class Handler():
    '''
     Notice:
        1. yaml format is utilized as foundational method to orgnize configurations.
        2. Nested layers for configurations should be as concise as possible.
        3. Supported types of bottom layer (configure value) are as follows:
            List, String, Value (float or integer), Bool
           Dictionary type is forbbiden since '.keys()' is used to end recursive loop.
        4. When merging (or partially update) configurations, one should not set the uptated value to 'None'
           otherwise the recursive process from bottom to top will erase attributes rather than update them.
    '''
    def __init__(self,file_name=None):
        self.file = None
        if file_name is None:
            self.__config = None
        else:
            self.load_config(file_name)
            self.file = file_name


    def load_config(self,file_name=None, encoding='utf-8'):
        if file_name is None and self.file is None:
            raise('Empty Input Path!')
        
        if file_name is not None:
            file = file_name
        
        if os.path.exists(file):
            with open(file,encoding=encoding) as f:
                # Realization of Config Member Call
                self.__config = self.__generate_attributes(yaml.load(f,yaml.FullLoader)) 
        else:
            raise('Non-exist File Path!')
        
    def print_current_config_path(self):
        print(self.file)
    

    @property
    def get_data(self):
        return self.__config
    

    
    def merge_from_config(self,file_name=None, encoding='utf-8'):
        if file_name is not None and os.path.exists(file_name):
            with open(file_name,encoding=encoding) as f:
                updated_config = self.__generate_attributes(yaml.load(f,yaml.FullLoader))
                self.__merge_attributes(self.__config, updated_config)
        else:
            raise('Non-exist File Path!')
        
    
    def __generate_attributes(self, cfgObj):
        if cfgObj is not None:
            # Recursively Traverse Config 'Tree'
            return self.__parse_config(cfgObj)

    
    def __parse_config(self, obj):
        '''
         Since we are using recuresive building process, 
         the nested config files are advised to be written in a concise manner.
        ''' 
        # Get Keys
        layer = AttributeClass()
        try:
            # Dictionary Data Type is NOT Allowed in Config Files
            keys = obj.keys() 

        except:
            # Last Layer
            # Extract Value
            # Supported Data Type: List, Value, String
            value = obj
            return value
        
        else:
            for k in keys:
                setattr(layer, k, self.__parse_config(obj[k]))
            return layer
        
    
    def __merge_attributes(self,obj_orig, obj_uptates):
        try:
            orig_vars = vars(obj_orig)
            update_vars = vars(obj_uptates)
        except:
            # Last layer
            # Extract Value
            value = obj_uptates
            return value
        else:
            for var in update_vars:
                if var in orig_vars:
                    update_value = self.__merge_attributes(getattr(obj_orig,var), getattr(obj_uptates,var))
                    if update_value is not None:
                        # update attribute value to 'None' is not support 
                        setattr(obj_orig,var,update_value)

class AttributeClass():
    '''
     Auxiliary Class for Creation of the Parsed Configuration.
    '''
    def __init__(self):
        pass
 