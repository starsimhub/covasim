'''
Test the speeds of different implementations
'''


import sciris as sc

n = int(10e4)
m = int(10e3)

class FlexDict(dict):
    '''
    A dict that allows more flexible element access: in addition to obj['a'],
    also allow obj[0] and obj.a. Lightweight implementation of the Sciris
    objdict class.
    '''

    def __getattribute__(self, attr):
        try: # First, try to get the attribute as an attribute
            output = dict.__getattribute__(self, attr)
            return output
        except Exception as E: # If that fails, try to get it as a dict item
            try:
                output = dict.__getitem__(self, attr)
                return output
            except: # If that fails, raise the original exception
                raise E

    def __getitem__(self, key):
        ''' Lightweight odict -- allow indexing by number, with low performance '''
        try:
            return super().__getitem__(key)
        except KeyError as KE:
            try: # Assume it's an integer
                dictkey = list(self.keys())[key]
                return self[dictkey]
            except:
                raise KE # This is the original errors

    def keys(self):
        return list(dict.keys(self))

    def values(self):
        return list(dict.values(self))

    def items(self):
        return list(dict.items(self))




d1 = {f'k{v}':v for v in range(n)}
d2 = FlexDict(d1)
d3 = sc.objdict(d1)


sc.tic()
for k in d1.keys():
    d1[k] = d1[k] + 1
sc.toc() # Elapsed time: 0.0129 s

sc.tic()
for k in d2.keys():
    d2[k] = d2[k] + 1
sc.toc() # Elapsed time: 0.0326 s

sc.tic()
for k in d3.keys():
    d3[k] = d3[k] + 1
sc.toc() # Elapsed time: 0.0600 s
