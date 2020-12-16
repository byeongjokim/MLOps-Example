# class Yaml_Args():
#     def __init__(self, yaml_filename):
#         self.set_args()

class Yaml_Args():
    def __init__(self):
        pass
    
    def append(self, k, v):
        setattr(self, k, v)
