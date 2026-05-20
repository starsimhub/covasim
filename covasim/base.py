'''
Base classes for Covasim. These classes handle a lot of the boilerplate of the
People and Sim classes (e.g. loading, saving, key lookups, etc.), so those classes
can be focused on the disease-specific functionality.
'''

import numpy as np
import pandas as pd
import sciris as sc
import datetime as dt
from . import version as cvv
from . import utils as cvu
from . import misc as cvm
from . import defaults as cvd
from . import parameters as cvpar

# Specify all externally visible classes this file defines
__all__ = ['ParsObj', 'Result', 'BaseSim', 'BasePeople', 'Person', 'FlexDict', 'Contacts', 'Layer']


#%% Define simulation classes

class FlexPretty(sc.prettyobj):
    '''
    A class that supports multiple different display options: namely obj.brief()
    for a one-line description and obj.disp() for a full description.
    '''

    def __repr__(self):
        ''' Use brief repr by default '''
        try:
            string = self._brief()
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, something went wrong printing object:\n{str(E)}'
        return string

    def _disp(self):
        ''' Verbose output -- use Sciris' pretty repr by default '''
        return sc.prepr(self)

    def disp(self, output=False):
        ''' Print or output verbose representation of the object '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string

    def _brief(self):
        ''' Brief output -- use a one-line output, a la Python's default '''
        return sc.objectid(self)

    def brief(self, output=False):
        ''' Print or output a brief representation of the object '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string


class ParsObj(FlexPretty):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars):
        self.update_pars(pars, create=True)
        return


    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        try:
            return self.pars[key]
        except:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)


    def __setitem__(self, key, value):
        ''' Ditto '''
        if key in self.pars:
            self.pars[key] = value
        else:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return


    def update_pars(self, pars=None, create=False):
        '''
        Update internal dict with new pars.

        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
        '''
        if pars is not None:
            if not isinstance(pars, dict):
                raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')
            if not hasattr(self, 'pars'):
                self.pars = pars
            if not create:
                available_keys = list(self.pars.keys())
                mismatches = [key for key in pars.keys() if key not in available_keys]
                if len(mismatches):
                    errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
                    raise sc.KeyNotFoundError(errormsg)
            self.pars.update(pars)
        return


class Result(object):
    '''
    Stores a single result -- by default, acts like an array.

    Args:
        name (str): name of this result, e.g. new_infections
        npts (int): if values is None, precreate it to be of this length
        scale (bool): whether or not the value scales by population scale factor
        color (str/arr): default color for plotting (hex or RGB notation)
        n_variants (int): the number of variants the result is for (0 for results not by variant)

    **Example**::

        import covasim as cv
        r1 = cv.Result(name='test1', npts=10)
        r1[:5] = 20
        print(r1.values)
    '''

    def __init__(self, name=None, npts=None, scale=True, color=None, n_variants=0):
        self.name =  name  # Name of this result
        self.scale = scale # Whether or not to scale the result by the scale factor
        if color is None:
            color = cvd.get_default_colors()['default']
        self.color = color # Default color
        if npts is None:
            npts = 0
        npts = int(npts)

        if n_variants>0:
            self.values = np.zeros((n_variants, npts), dtype=cvd.result_float)
        else:
            self.values = np.zeros(npts, dtype=cvd.result_float)

        self.low  = None
        self.high = None
        return


    def __repr__(self):
        ''' Use pretty repr, like sc.prettyobj, but displaying full values '''
        output  = sc.prepr(self, skip=['values', 'low', 'high'], use_repr=False)
        output += 'values:\n' + repr(self.values)
        if self.low is not None:
            output += '\nlow:\n' + repr(self.low)
        if self.high is not None:
            output += '\nhigh:\n' + repr(self.high)
        return output


    def __getitem__(self, key):
        ''' To allow e.g. result['high'] instead of result.high, and result[5] instead of result.values[5] '''
        if isinstance(key, str):
            output = getattr(self, key)
        else:
            output = self.values.__getitem__(key)
        return output


    def __setitem__(self, key, value):
        ''' To allow e.g. result[:] = 1 instead of result.values[:] = 1 '''
        if isinstance(key, str):
            setattr(self, key, value)
        else:
            self.values.__setitem__(key, value)
        return


    def __len__(self):
        ''' To allow len(result) instead of len(result.values) '''
        return len(self.values)


    @property
    def npts(self):
        return len(self.values)


def set_metadata(obj, frame=4, **kwargs):
    ''' Set standard metadata for an object '''
    obj.created = kwargs.get('created', sc.now())
    obj.version = kwargs.get('version', cvv.__version__)
    obj.git_info = kwargs.get('git_info', cvm.git_info(frame=frame)) # 4 = 2 (default) + base + caller
    return


class BaseSim(ParsObj):
    '''
    The BaseSim class stores various methods useful for the Sim that are not directly
    related to simulating the epidemic. It is not used outside of the Sim object,
    so the separation of methods into the BaseSim and Sim classes is purely to keep
    each one of manageable size.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Initialize and set the parameters as attributes
        return


    def _disp(self):
        '''
        Print a verbose display of the sim object. Used by repr(). See sim.disp()
        for the user version. Equivalent to sc.prettyobj().
        '''
        return sc.prepr(self)


    def _brief(self):
        '''
        Return a one-line description of a sim -- used internally and by repr();
        see sim.brief() for the user version.
        '''
        # Try to get a detailed description of the sim...
        try:
            if self.results_ready:
                infections = self.summary['cum_infections']
                deaths = self.summary['cum_deaths']
                results = f'{infections:n}⚙, {deaths:n}☠'
            else:
                results = 'not run'

            # Set label string
            labelstr = f'"{self.label}"' if self.label else '<no label>'

            start = sc.date(self['start_day'], as_date=False)
            if self['end_day']:
                end = sc.date(self['end_day'], as_date=False)
            else:
                end = sc.date(self['n_days'], start_date=start)

            pop_size = self['pop_size']
            pop_type = self['pop_type']
            string   = f'Sim({labelstr}; {start} to {end}; pop: {pop_size:n} {pop_type}; epi: {results})'

        # ...but if anything goes wrong, return the default with a warning
        except Exception as E: # pragma: no cover
            string = sc.objectid(self)
            string += f'Warning, sim appears to be malformed; use sim.disp() for details:\n{str(E)}'

        return string


    def update_pars(self, pars=None, create=False, **kwargs):
        ''' Ensure that metaparameters get used properly before being updated '''

        # Merge everything together
        pars = sc.mergedicts(pars, kwargs)
        if pars:

            # Define aliases
            mapping = dict(
                n_agents = 'pop_size',
                init_infected = 'pop_infected',
            )
            for key1,key2 in mapping.items():
                if key1 in pars:
                    pars[key2] = pars.pop(key1)

            # Handle other special parameters
            if pars.get('pop_type'):
                cvpar.reset_layer_pars(pars, force=False)
            if pars.get('prog_by_age'):
                pars['prognoses'] = cvpar.get_prognoses(by_age=pars['prog_by_age'], version=self._default_ver) # Reset prognoses

            # Call update_pars() for ParsObj
            super().update_pars(pars=pars, create=create)

        return


    def set_metadata(self, simfile):
        ''' Set the metadata for the simulation -- creation time and filename '''
        set_metadata(self, frame=5)
        if simfile is None:
            self.simfile = 'covasim.sim'
        return


    def set_seed(self, seed=-1):
        '''
        Set the seed for the random number stream from the stored or supplied value

        Args:
            seed (None or int): if no argument, use current seed; if None, randomize; otherwise, use and store supplied seed
        '''
        # Unless no seed is supplied, reset it
        if seed != -1:
            self['rand_seed'] = seed
        cvu.set_seed(self['rand_seed'])
        return

    @property
    def n(self):
        ''' Count the number of people -- if it fails, assume none '''
        try: # By default, the length of the people dict
            return len(self.people)
        except:  # pragma: no cover # If it's None or missing
            return 0

    @property
    def scaled_pop_size(self):
        ''' Get the total population size, i.e. the number of agents times the scale factor -- if it fails, assume none '''
        try:
            return self['pop_size']*self['pop_scale']
        except:  # pragma: no cover # If it's None or missing
            return 0

    @property
    def npts(self):
        ''' Count the number of time points '''
        try:
            return int(self['n_days'] + 1)
        except: # pragma: no cover
            return 0

    @property
    def tvec(self):
        ''' Create a time vector '''
        try:
            return np.arange(self.npts)
        except: # pragma: no cover
            return np.array([])

    @property
    def datevec(self):
        '''
        Create a vector of dates

        Returns:
            Array of `datetime` instances containing the date associated with each
            simulation time step

        '''
        try:
            return self['start_day'] + self.tvec * dt.timedelta(days=1)
        except: # pragma: no cover
            return np.array([])


    def day(self, day, *args):
        '''
        Convert a string, date/datetime object, or int to a day (int).

        Args:
            day (str, date, int, or list): convert any of these objects to a day relative to the simulation's start day

        Returns:
            days (int or str): the day(s) in simulation time

        **Example**::

            sim.day('2020-04-05') # Returns 35
        '''
        return sc.day(day, *args, start_date=self['start_day'])


    def date(self, ind, *args, dateformat=None, as_date=False):
        '''
        Convert one or more integer days of simulation time to a date/list of dates --
        by default returns a string, or returns a datetime Date object if as_date is True.
        See also cv.date(), which provides a partly overlapping set of date conversion
        features.

        Args:
            ind (int, list, or array): the index day(s) in simulation time (NB: strings and date objects are accepted, and will be passed unchanged)
            args (list): additional day(s)
            dateformat (str): the format to return the date in
            as_date (bool): whether to return as a datetime date instead of a string

        Returns:
            dates (str, Date, or list): the date(s) corresponding to the simulation day(s)

        **Examples**::

            sim = cv.Sim()
            sim.date(34) # Returns '2020-04-04'
            sim.date([34, 54]) # Returns ['2020-04-04', '2020-04-24']
            sim.date([34, '2020-04-24']) # Returns ['2020-04-04', '2020-04-24']
            sim.date(34, 54, as_date=True) # Returns [datetime.date(2020, 4, 4), datetime.date(2020, 4, 24)]
        '''

        # Handle inputs
        if not isinstance(ind, list): # If it's a number, string, or dateobj, convert it to a list
            ind = sc.tolist(ind)
        ind.extend(args)
        if dateformat is None:
            dateformat = '%Y-%m-%d'

        # Do the conversion
        dates = []
        for raw in ind:
            if sc.isnumber(raw):
                date_obj = sc.date(self['start_day'], as_date=True) + dt.timedelta(days=int(raw))
            else:
                date_obj = sc.date(raw, as_date=True)
            if as_date:
                dates.append(date_obj)
            else:
                dates.append(date_obj.strftime(dateformat))

        # Return a string rather than a list if only one provided
        if len(ind)==1:
            dates = dates[0]

        return dates


    def result_keys(self, which='main'):
        '''
        Get the actual results objects, not other things stored in sim.results.

        If which is 'main', return only the main results keys. If 'variant', return
        only variant keys. If 'all', return all keys.

        '''
        keys = []
        choices = ['main', 'variant', 'all']
        if which in ['main', 'all']:
            keys += [key for key,res in self.results.items() if isinstance(res, Result)]
        if which in ['variant', 'all'] and 'variant' in self.results:
            keys += [key for key,res in self.results['variant'].items() if isinstance(res, Result)]
        if which not in choices: # pragma: no cover
            errormsg = f'Choice "which" not available; choices are: {sc.strjoin(choices)}'
            raise ValueError(errormsg)
        return keys


    def copy(self):
        ''' Returns a deep copy of the sim '''
        return sc.dcp(self)


    def export_results(self, for_json=True, filename=None, indent=2, *args, **kwargs):
        '''
        Convert results to dict -- see also to_json().

        The results written to Excel must have a regular table shape, whereas
        for the JSON output, arbitrary data shapes are supported.

        Args:
            for_json (bool): if False, only data associated with Result objects will be included in the converted output
            filename (str): filename to save to; if None, do not save
            indent (int): indent (int): if writing to file, how many indents to use per nested level
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            resdict (dict): dictionary representation of the results

        '''

        if not self.results_ready: # pragma: no cover
            errormsg = 'Please run the sim before exporting the results'
            raise RuntimeError(errormsg)

        resdict = {}
        resdict['t'] = self.results['t'] # Assume that there is a key for time

        if for_json:
            resdict['timeseries_keys'] = self.result_keys()
        for key,res in self.results.items():
            if isinstance(res, Result):
                resdict[key] = res.values
                if res.low is not None:
                    resdict[key+'_low'] = res.low
                if res.high is not None:
                    resdict[key+'_high'] = res.high
            elif for_json:
                if key == 'date':
                    resdict[key] = [str(d) for d in res] # Convert dates to strings
                else:
                    resdict[key] = res
        if filename is not None:
            sc.savejson(filename=filename, obj=resdict, indent=indent, *args, **kwargs)
        return resdict


    def export_pars(self, filename=None, indent=2, *args, **kwargs):
        '''
        Return parameters for JSON export -- see also to_json().

        This method is required so that interventions can specify
        their JSON-friendly representation.

        Args:
            filename (str): filename to save to; if None, do not save
            indent (int): indent (int): if writing to file, how many indents to use per nested level
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            pardict (dict): a dictionary containing all the parameter values
        '''
        pardict = {}
        for key in self.pars.keys():
            if key == 'interventions':
                pardict[key] = [intervention.to_json() for intervention in self.pars[key]]
            elif key == 'start_day':
                pardict[key] = str(self.pars[key])
            else:
                pardict[key] = self.pars[key]
        if filename is not None:
            sc.savejson(filename=filename, obj=pardict, indent=indent, *args, **kwargs)
        return pardict


    def to_json(self, filename=None, keys=None, tostring=False, indent=2, verbose=False, *args, **kwargs):
        '''
        Export results and parameters as JSON.

        Args:
            filename (str): if None, return string; else, write to file
            keys (str or list): attributes to write to json (default: results, parameters, and summary)
            tostring (bool): if not writing to file, whether to write to string (alternative is sanitized dictionary)
            indent (int): if writing to file, how many indents to use per nested level
            verbose (bool): detail to print
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        **Examples**::

            json = sim.to_json()
            sim.to_json('results.json')
            sim.to_json('summary.json', keys='summary')
        '''

        # Handle keys
        if keys is None:
            keys = ['results', 'pars', 'summary']
        keys = sc.tolist(keys)

        # Convert to JSON-compatible format
        d = {}
        for key in keys:
            if key == 'results':
                resdict = self.export_results(for_json=True)
                d['results'] = resdict
            elif key in ['pars', 'parameters']:
                pardict = self.export_pars()
                d['parameters'] = pardict
            elif key == 'summary':
                d['summary'] = dict(sc.dcp(self.summary))
            else: # pragma: no cover
                try:
                    d[key] = sc.sanitizejson(getattr(self, key))
                except Exception as E:
                    errormsg = f'Could not convert "{key}" to JSON: {str(E)}; continuing...'
                    print(errormsg)

        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, *args, **kwargs)

        return output


    def to_df(self, date_index=False):
        '''
        Export results to a pandas dataframe

        Args:
            date_index  (bool): if True, use the date as the index
        '''
        resdict = self.export_results(for_json=False)
        df = pd.DataFrame.from_dict(resdict)
        df['date'] = self.datevec
        new_columns = ['t','date'] + df.columns[1:-1].tolist() # Get column order
        df = df.reindex(columns=new_columns) # Reorder so 't' and 'date' are first
        if date_index:
            df = df.set_index('date')
        return df


    def to_excel(self, filename=None, skip_pars=None):
        '''
        Export parameters and results as Excel format

        Args:
            filename  (str): if None, return string; else, write to file
            skip_pars (list): if provided, a custom list parameters to exclude

        Returns:
            An sc.Spreadsheet with an Excel file, or writes the file to disk
        '''
        if skip_pars is None:
            skip_pars = ['variant_map', 'vaccine_map'] # These include non-string keys so fail at sc.flattendict()

        # Export results
        result_df = self.to_df(date_index=True)

        # Export parameters
        pars = {k:v for k,v in self.pars.items() if k not in skip_pars}
        par_df = pd.DataFrame.from_dict(sc.flattendict(pars, sep='_'), orient='index', columns=['Value'])
        par_df.index.name = 'Parameter'

        # Convert to spreadsheet
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Results')
            par_df.to_excel(writer, sheet_name='Parameters')
        spreadsheet.load()

        if filename is None:
            output = spreadsheet
        else:
            output = spreadsheet.save(filename)

        return output


    def shrink(self, skip_attrs=None, in_place=True):
        '''
        "Shrinks" the simulation by removing the people and other memory-intensive
        attributes (e.g., some interventions and analyzers), and returns a copy of
        the "shrunken" simulation. Used to reduce the memory required for RAM or
        for saved files.

        Args:
            skip_attrs (list): a list of attributes to skip (remove) in order to perform the shrinking; default "people"
            in_palce (bool): whether to perform the shrinking in place (default), or return a shrunken copy instead

        Returns:
            shrunken (Sim): a Sim object with the listed attributes removed
        '''
        from . import interventions as cvi # To avoid circular imports
        from . import analysis as cva

        # By default, skip people (~90% of memory), the popdict (which is usually empty anyway), and _orig_pars (which is just a backup)
        if skip_attrs is None:
            skip_attrs = ['popdict', 'people', '_orig_pars']

        # Create the new object, and copy original dict, skipping the skipped attributes
        if in_place:
            shrunken = self
            for attr in skip_attrs:
                setattr(self, attr, None)
        else:
            shrunken = object.__new__(self.__class__)
            shrunken.__dict__ = {k:(v if k not in skip_attrs else None) for k,v in self.__dict__.items()}

        # Shrink interventions and analyzers, with a lot of checking along the way
        for key in ['interventions', 'analyzers']:
            ias = sc.tolist(self.pars[key]) # List of interventions or analyzers
            shrunken_ias = []
            for ia in ias:
                if isinstance(ia, (cvi.Intervention, cva.Analyzer)):
                    shrunken_ia = ia.shrink(in_place=in_place)
                else:
                    shrunken_ia = ia
                shrunken_ias.append(shrunken_ia)
            self.pars[key] = shrunken_ias # Actually shrink, and re-store

        # Don't return if in place
        if in_place:
            return
        else:
            return shrunken


    def save(self, filename=None, keep_people=None, skip_attrs=None, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            kwargs: passed to sc.makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        **Example**::

            sim.save() # Saves to a .sim file
        '''

        # Set keep_people based on whether or not we're in the middle of a run
        if keep_people is None:
            if self.initialized and not self.results_ready:
                keep_people = True
            else:
                keep_people = False

        # Handle the filename
        if filename is None:
            filename = self.simfile
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Handle the shrinkage and save
        if skip_attrs or not keep_people:
            obj = self.shrink(skip_attrs=skip_attrs, in_place=False)
        else:
            obj = self
        cvm.save(filename=filename, obj=obj)

        return filename


    @staticmethod
    def load(filename, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            filename (str): the name or path of the file to load from
            kwargs: passed to cv.load()

        Returns:
            sim (Sim): the loaded simulation object

        **Example**::

            sim = cv.Sim.load('my-simulation.sim')
        '''
        sim = cvm.load(filename, *args, **kwargs)
        if not isinstance(sim, BaseSim): # pragma: no cover
            errormsg = f'Cannot load object of {type(sim)} as a Sim object'
            raise TypeError(errormsg)
        return sim


    def _get_ia(self, which, label=None, partial=False, as_list=False, as_inds=False, die=True, first=False):
        ''' Helper method for get_interventions() and get_analyzers(); see get_interventions() docstring '''

        # Handle inputs
        if which not in ['interventions', 'analyzers']: # pragma: no cover
            errormsg = f'This method is only defined for interventions and analyzers, not "{which}"'
            raise ValueError(errormsg)

        ia_list = sc.tolist(self.pars[which]) # List of interventions or analyzers
        n_ia = len(ia_list) # Number of interventions/analyzers

        if label == 'summary': # Print a summary of the interventions
            df = sc.dataframe(columns=['ind', 'label', 'type'])
            for ind,ia_obj in enumerate(ia_list):
                df = df.append(dict(ind=ind, label=str(ia_obj.label), type=type(ia_obj)))
            print(f'Summary of {which}:')
            print(df)
            return

        else: # Standard usage case
            position = 0 if first else -1 # Choose either the first or last element
            if label is None: # Get all interventions if no label is supplied, e.g. sim.get_interventions()
                label = np.arange(n_ia)
            if isinstance(label, np.ndarray): # Allow arrays to be provided
                label = label.tolist()
            labels = sc.tolist(label)

            # Calculate the matches
            matches = []
            match_inds = []
            for label in labels:
                if sc.isnumber(label):
                    matches.append(ia_list[label]) # This will raise an exception if an invalid index is given
                    label = n_ia + label if label<0 else label # Convert to a positive number
                    match_inds.append(label)
                elif sc.isstring(label) or isinstance(label, type):
                    for ind,ia_obj in enumerate(ia_list):
                        if sc.isstring(label) and ia_obj.label == label or (partial and (label in str(ia_obj.label))):
                            matches.append(ia_obj)
                            match_inds.append(ind)
                        elif isinstance(label, type) and (isinstance(ia_obj, label) or str(ia_obj.__class__) == str(label)):
                            matches.append(ia_obj)
                            match_inds.append(ind)
                else: # pragma: no cover
                    errormsg = f'Could not interpret label type "{type(label)}": should be str, int, list, or {which} class'
                    raise TypeError(errormsg)

            # Parse the output options
            if as_inds:
                output = match_inds
            elif as_list: # Used by get_interventions()
                output = matches
            else:
                if len(matches) == 0: # pragma: no cover
                    if die:
                        errormsg = f'No {which} matching "{label}" were found'
                        raise ValueError(errormsg)
                    else:
                        output = None
                else:
                    output = matches[position] # Return either the first or last match (usually), used by get_intervention()

            return output


    def get_interventions(self, label=None, partial=False, as_inds=False):
        '''
        Find the matching intervention(s) by label, index, or type. If None, return
        all interventions. If the label provided is "summary", then print a summary
        of the interventions (index, label, type).

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            as_inds (bool): if true, return matching indices instead of the actual interventions

        **Examples**::

            tp = cv.test_prob(symp_prob=0.1)
            cb1 = cv.change_beta(days=5, changes=0.3, label='NPI')
            cb2 = cv.change_beta(days=10, changes=0.3, label='Masks')
            sim = cv.Sim(interventions=[tp, cb1, cb2])
            cb1, cb2 = sim.get_interventions(cv.change_beta)
            tp, cb2 = sim.get_interventions([0,2])
            ind = sim.get_interventions(cv.change_beta, as_inds=True) # Returns [1,2]
            sim.get_interventions('summary') # Prints a summary
        '''
        return self._get_ia('interventions', label=label, partial=partial, as_inds=as_inds, as_list=True)


    def get_intervention(self, label=None, partial=False, first=False, die=True):
        '''
        Like get_interventions(), find the matching intervention(s) by label,
        index, or type. If more than one intervention matches, return the last
        by default. If no label is provided, return the last intervention in the list.

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            first (bool): if true, return first matching intervention (otherwise, return last)
            die (bool): whether to raise an exception if no intervention is found

        **Examples**::

            tp = cv.test_prob(symp_prob=0.1)
            cb = cv.change_beta(days=5, changes=0.3, label='NPI')
            sim = cv.Sim(interventions=[tp, cb])
            cb = sim.get_intervention('NPI')
            cb = sim.get_intervention('NP', partial=True)
            cb = sim.get_intervention(cv.change_beta)
            cb = sim.get_intervention(1)
            cb = sim.get_intervention()
            tp = sim.get_intervention(first=True)
        '''
        return self._get_ia('interventions', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)


    def get_analyzers(self, label=None, partial=False, as_inds=False):
        '''
        Same as get_interventions(), but for analyzers.
        '''
        return self._get_ia('analyzers', label=label, partial=partial, as_list=True, as_inds=as_inds)


    def get_analyzer(self, label=None, partial=False, first=False, die=True):
        '''
        Same as get_intervention(), but for analyzers.
        '''
        return self._get_ia('analyzers', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)


#%% Define people classes

class BasePeople(FlexPretty):
    '''
    A class to handle all the boilerplate for people -- note that as with the
    BaseSim vs Sim classes, everything interesting happens in the People class,
    whereas this class exists to handle the less interesting implementation details.
    '''

    def set_pars(self, pars=None):
        '''
        Re-link the parameters stored in the people object to the sim containing it,
        and perform some basic validation.
        '''
        orig_pars = self.__dict__.get('pars') # Get the current parameters using dict's get method
        if pars is None:
            if orig_pars is not None: # If it has existing parameters, use them
                pars = orig_pars
            else:
                pars = {}
        elif sc.isnumber(pars): # Interpret as a population size
            pars = {'pop_size':pars} # Ensure it's a dictionary

        # Copy from old parameters to new parameters
        if isinstance(orig_pars, dict):
            for k,v in orig_pars.items():
                if k not in pars:
                    pars[k] = v

        # Do minimal validation -- needed here since pop_size should be converted to an int when first set
        if 'pop_size' not in pars:
            errormsg = f'The parameter "pop_size" must be included in a population; keys supplied were:\n{sc.newlinejoin(pars.keys())}'
            raise sc.KeyNotFoundError(errormsg)
        pars['pop_size'] = int(pars['pop_size'])
        pars.setdefault('n_variants', 1)
        pars.setdefault('location', None)
        self.pars = pars # Actually store the pars
        return


    def validate(self, sim_pars=None, die=True, verbose=False):
        '''
        Perform validation on the People object.

        Args:
            sim_pars (dict): dictionary of parameters from the sim to ensure they match the current People object
            die (bool): whether to raise an exception if validation fails
            verbose (bool): detail to print
        '''

        # Check that parameters match
        if sim_pars is not None:
            mismatches = {}
            keys = ['pop_size', 'pop_type', 'location', 'pop_infected', 'frac_susceptible', 'n_variants'] # These are the keys used in generating the population
            for key in keys:
                sim_v = sim_pars.get(key)
                ppl_v = self.pars.get(key)
                if sim_v is not None and ppl_v is not None:
                    if sim_v != ppl_v:
                        mismatches[key] = sc.objdict(sim=sim_v, people=ppl_v)
            if len(mismatches):
                errormsg = 'Validation failed due to the following mismatches between the sim and the people parameters:\n'
                for k,v in mismatches.items():
                    errormsg += f'  {k}: sim={v.sim}, people={v.people}'
                raise ValueError(errormsg)

        # Check that the keys match
        contact_layer_keys = set(self.contacts.keys())
        layer_keys    = set(self.layer_keys())
        if contact_layer_keys != layer_keys:
            errormsg = f'Parameters layers {layer_keys} are not consistent with contact layers {contact_layer_keys}'
            raise ValueError(errormsg)

        # Check that the length of each array is consistent
        expected_len = len(self)
        expected_variants = self.pars['n_variants']
        for key in self.keys():
            if self[key].ndim == 1:
                actual_len = len(self[key])
            else: # If it's 2D, variants need to be checked separately
                actual_variants, actual_len = self[key].shape
                if actual_variants != expected_variants:
                    if verbose:
                        print(f'Resizing "{key}" from {actual_variants} to {expected_variants}')
                    self._resize_arrays(keys=key, new_size=(expected_variants, expected_len))
            if actual_len != expected_len: # pragma: no cover
                if die:
                    errormsg = f'Length of key "{key}" did not match population size ({actual_len} vs. {expected_len})'
                    raise IndexError(errormsg)
                else:
                    if verbose:
                        print(f'Resizing "{key}" from {actual_len} to {expected_len}')
                    self._resize_arrays(keys=key)

        # Check that the layers are valid
        for layer in self.contacts.values():
            layer.validate()

        return


    def _resize_arrays(self, new_size=None, keys=None):
        ''' Resize arrays if any mismatches are found '''

        # Handle None or tuple input (representing variants and pop_size)
        if new_size is None:
            new_size = len(self)
        pop_size = new_size if not isinstance(new_size, tuple) else new_size[1]
        self.pars['pop_size'] = pop_size

        # Reset sizes
        if keys is None:
            keys = self.keys()
        keys = sc.tolist(keys)
        for key in keys:
            self[key].resize(new_size, refcheck=False) # Don't worry about cross-references to the arrays

        return


    def lock(self):
        ''' Lock the people object to prevent keys from being added '''
        self._lock = True
        return


    def unlock(self):
        ''' Unlock the people object to allow keys to be added '''
        self._lock = False
        return


    def __getitem__(self, key):
        ''' Allow people['attr'] instead of getattr(people, 'attr')
            If the key is an integer, alias `people.person()` to return a `Person` instance
        '''
        try:
            return self.__dict__[key]
        except: # pragma: no cover
            if isinstance(key, int):
                return self.person(key)
            else:
                errormsg = f'Key "{key}" is not a valid attribute of people'
                raise AttributeError(errormsg)


    def __setitem__(self, key, value):
        ''' Ditto '''
        if self._lock and key not in self.__dict__: # pragma: no cover
            errormsg = f'Key "{key}" is not a current attribute of people, and the people object is locked; see people.unlock()'
            raise AttributeError(errormsg)
        self.__dict__[key] = value
        return


    def __len__(self):
        ''' This is just a scalar, but validate() and _resize_arrays() make sure it's right '''
        return int(self.pars['pop_size'])


    def __iter__(self):
        ''' Iterate over people '''
        for i in range(len(self)):
            yield self[i]


    def __add__(self, people2):
        ''' Combine two people arrays '''
        newpeople = sc.dcp(self)
        keys = list(self.keys())
        for key in keys:
            npval = newpeople[key]
            p2val = people2[key]
            if npval.ndim == 1:
                newpeople.set(key, np.concatenate([npval, p2val], axis=0), die=False) # Allow size mismatch
            elif npval.ndim == 2:
                newpeople.set(key, np.concatenate([npval, p2val], axis=1), die=False)
            else:
                errormsg = f'Not sure how to combine arrays of {npval.ndim} dimensions for {key}'
                raise NotImplementedError(errormsg)

        # Validate
        newpeople.pars['pop_size'] += people2.pars['pop_size']
        newpeople.validate()

        # Reassign UIDs so they're unique
        newpeople.set('uid', np.arange(len(newpeople)))

        return newpeople


    def __radd__(self, people2):
        ''' Allows sum() to work correctly '''
        if not people2: return self
        else:           return self.__add__(people2)


    def _brief(self):
        '''
        Return a one-line description of the people -- used internally and by repr();
        see people.brief() for the user version.
        '''
        try:
            layerstr = ', '.join([str(k) for k in self.layer_keys()])
            string   = f'People(n={len(self):0n}; layers: {layerstr})'
        except Exception as E: # pragma: no cover
            string = sc.objectid(self)
            string += f'Warning, multisim appears to be malformed:\n{str(E)}'
        return string


    def summarize(self, output=False):
        ''' Print a summary of the people -- same as brief '''
        return self.brief(output=output)


    def set(self, key, value, die=True):
        ''' Ensure sizes and dtypes match '''
        current = self[key]
        value = np.array(value, dtype=self._dtypes[key]) # Ensure it's the right type
        if die and len(value) != len(current): # pragma: no cover
            errormsg = f'Length of new array does not match current ({len(value)} vs. {len(current)})'
            raise IndexError(errormsg)
        self[key] = value
        return


    def get(self, key):
        ''' Convenience method -- key can be string or list of strings '''
        if isinstance(key, str):
            return self[key]
        elif isinstance(key, list):
            arr = np.zeros((len(self), len(key)))
            for k,ky in enumerate(key):
                arr[:,k] = self[ky]
            return arr


    def true(self, key):
        ''' Return indices matching the condition '''
        return self[key].nonzero()[0]


    def false(self, key):
        ''' Return indices not matching the condition '''
        return (~self[key]).nonzero()[0]


    def defined(self, key):
        ''' Return indices of people who are not-nan '''
        return (~np.isnan(self[key])).nonzero()[0]


    def undefined(self, key):
        ''' Return indices of people who are nan '''
        return np.isnan(self[key]).nonzero()[0]


    def count(self, key):
        ''' Count the number of people for a given key '''
        return np.count_nonzero(self[key])


    def count_by_variant(self, key, variant):
        ''' Count the number of people for a given key '''
        return np.count_nonzero(self[key][variant,:])


    def count_not(self, key):
        ''' Count the number of people who do not have a property for a given key '''
        return len(self[key]) - self.count(key)


    def keys(self):
        ''' Returns keys for all properties of the people object '''
        return self.meta.all_states[:]


    def person_keys(self):
        ''' Returns keys specific to a person (e.g., their age) '''
        return self.meta.person[:]


    def state_keys(self):
        ''' Returns keys for different states of a person (e.g., symptomatic) '''
        return self.meta.states[:]


    def date_keys(self):
        ''' Returns keys for different event dates (e.g., date a person became symptomatic) '''
        return self.meta.dates[:]


    def dur_keys(self):
        ''' Returns keys for different durations (e.g., the duration from exposed to infectious) '''
        return self.meta.durs[:]


    def layer_keys(self):
        ''' Get the available contact keys -- try contacts first, then beta_layer '''
        try:
            keys = list(self.contacts.keys())
        except: # If not fully initialized
            try:
                keys = list(self.pars['beta_layer'].keys())
            except:  # pragma: no cover # If not even partially initialized
                keys = []
        return keys


    def indices(self):
        ''' The indices of each people array '''
        return np.arange(len(self))


    def to_df(self):
        ''' Convert to a Pandas dataframe '''
        df = pd.DataFrame.from_dict({key:self[key] for key in self.keys()})
        return df


    def to_arr(self):
        ''' Return as numpy array '''
        arr = np.empty((len(self), len(self.keys())), dtype=cvd.default_float)
        for k,key in enumerate(self.keys()):
            if key == 'uid':
                arr[:,k] = np.arange(len(self))
            else:
                arr[:,k] = self[key]
        return arr


    def person(self, ind):
        ''' Method to create person from the people '''
        p = Person()
        for key in self.meta.all_states:
            data = self[key]
            if data.ndim == 1:
                val = data[ind]
            elif data.ndim == 2:
                val = data[:,ind]
            else:
                errormsg = f'Cannot extract data from {key}: unexpected dimensionality ({data.ndim})'
                raise ValueError(errormsg)
            setattr(p, key, val)

        contacts = {}
        for lkey, layer in self.contacts.items():
            contacts[lkey] = layer.find_contacts(ind)
        p.contacts = contacts

        return p


    def to_list(self):
        ''' Return all people as a list '''
        return list(self)


    def from_list(self, people, resize=True):
        ''' Convert a list of people back into a People object '''

        # Handle population size
        pop_size = len(people)
        if resize:
            self._resize_arrays(new_size=pop_size)

        # Iterate over people -- slow!
        for p,person in enumerate(people):
            for key in self.keys():
                self[key][p] = getattr(person, key)

        return


    def to_graph(self): # pragma: no cover
        '''
        Convert all people to a networkx MultiDiGraph, including all properties of
        the people (nodes) and contacts (edges).

        **Example**::

            import covasim as cv
            import networkx as nx
            sim = cv.Sim(pop_size=50, pop_type='hybrid', contacts=dict(h=3, s=10, w=10, c=5)).run()
            G = sim.people.to_graph()
            nodes = G.nodes(data=True)
            edges = G.edges(keys=True)
            node_colors = [n['age'] for i,n in nodes]
            layer_map = dict(h='#37b', s='#e11', w='#4a4', c='#a49')
            edge_colors = [layer_map[G[i][j][k]['layer']] for i,j,k in edges]
            edge_weights = [G[i][j][k]['beta']*5 for i,j,k in edges]
            nx.draw(G, node_color=node_colors, edge_color=edge_colors, width=edge_weights, alpha=0.5)
        '''
        import networkx as nx

        # Copy data from people into graph
        G = self.contacts.to_graph()
        for key in self.keys():
            data = {k:v for k,v in enumerate(self[key])}
            nx.set_node_attributes(G, data, name=key)

        # Include global layer weights
        for u,v,k in G.edges(keys=True):
            edge = G[u][v][k]
            edge['beta'] *= self.pars['beta_layer'][edge['layer']]

        return G


    def save(self, filename=None, force=False, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Note: by default this function raises an exception if trying to save a
        run or partially run People object, since the changes that happen during
        a run are usually irreversible.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            force (bool): whether to allow saving even of a run or partially-run People object
            kwargs: passed to ``sc.makefilepath()``

        Returns:
            filename (str): the validated absolute path to the saved file

        **Example**::

            sim = cv.Sim()
            sim.initialize()
            sim.people.save() # Saves to a .ppl file
        '''

        # Check if we're trying to save an already run People object
        if self.t > 0 and not force:
            errormsg = f'''
The People object has already been run (t = {self.t}), which is usually not the
correct state to save it in since it cannot be re-initialized. If this is intentional,
use sim.people.save(force=True). Otherwise, the correct approach is:

    sim = cv.Sim(...)
    sim.initialize() # Create the people object but do not run
    sim.people.save() # Save people immediately after initialization
    sim.run() # The People object is
'''
            raise RuntimeError(errormsg)

        # Handle the filename
        if filename is None:
            filename = 'covasim.ppl'
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename
        cvm.save(filename=filename, obj=self)

        return filename


    @staticmethod
    def load(filename, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            filename (str): the name or path of the file to load from
            args (list): passed to ``cv.load()``
            kwargs (dict): passed to ``cv.load()``

        Returns:
            people (People): the loaded people object

        **Example**::

            people = cv.people.load('my-people.ppl')
        '''
        people = cvm.load(filename, *args, **kwargs)
        if not isinstance(people, BasePeople): # pragma: no cover
            errormsg = f'Cannot load object of {type(people)} as a People object'
            raise TypeError(errormsg)
        return people



    def init_contacts(self, reset=False):
        ''' Initialize the contacts dataframe with the correct columns and data types '''

        # Create the contacts dictionary
        contacts = Contacts(layer_keys=self.layer_keys())

        if self.contacts is None or reset: # Reset all
            self.contacts = contacts
        else: # Only replace specified keys
            for key,layer in contacts.items():
                self.contacts[key] = layer
        return


    def add_contacts(self, contacts, lkey=None, beta=None):
        '''
        Add new contacts to the array. See also contacts.add_layer().
        '''

        # Validate the supplied contacts
        if isinstance(contacts, (Contacts, dict)): # If it's a Contacts object or a dict, we can use it directly
            if isinstance(contacts, dict) and lkey is not None: # Edge case: a dict for a single layer has been provided
                new_contacts = {}
                new_contacts[lkey] = contacts
            else:
                if 'p1' in contacts: # Avoid the mistake of passing a single layer
                    errormsg = 'To supply a single layer as a dict, you must supply an lkey as well'
                    raise ValueError(errormsg)
                new_contacts = contacts # Main use case
        elif isinstance(contacts, Layer):
            if lkey is None: # If no layer key is supplied, use the first layer
                lkey = self.layer_keys()[0]
            new_contacts = {}
            new_contacts[lkey] = contacts
        elif isinstance(contacts, list): # Assume it's a list of contacts by person, not an edgelist
            new_contacts = self.make_edgelist(contacts) # Assume contains key info
        else: # pragma: no cover
            errormsg = f'Cannot understand contacts of type {type(contacts)}; expecting dataframe, array, or dict'
            raise TypeError(errormsg)

        # Ensure the columns are right and add values if supplied
        for lkey, new_layer in new_contacts.items():
            n = len(new_layer['p1'])
            if 'beta' not in new_layer.keys() or len(new_layer['beta']) != n:
                if beta is None:
                    beta = 1.0
                beta = cvd.default_float(beta)
                new_layer['beta'] = np.ones(n, dtype=cvd.default_float)*beta

            # Create the layer if it doesn't yet exist
            if lkey not in self.contacts:
                self.contacts[lkey] = Layer(label=lkey)

            # Actually include them, and update properties if supplied
            for col in self.contacts[lkey].keys(): # Loop over the supplied columns
                self.contacts[lkey][col] = np.concatenate([self.contacts[lkey][col], new_layer[col]])
            self.contacts[lkey].validate()

        return


    def make_edgelist(self, contacts):
        '''
        Parse a list of people with a list of contacts per person and turn it
        into an edge list.
        '''

        # Handle layer keys
        lkeys = self.layer_keys()
        if len(contacts):
            contact_keys = contacts[0].keys() # Pull out the keys of this contact list
            lkeys += [key for key in contact_keys if key not in lkeys] # Extend the layer keys

        # Initialize the new contacts
        new_contacts = Contacts(layer_keys=lkeys)
        for lkey in lkeys:
            new_contacts[lkey]['p1']    = [] # Person 1 of the contact pair
            new_contacts[lkey]['p2']    = [] # Person 2 of the contact pair

        # Populate the new contacts
        for p,cdict in enumerate(contacts):
            for lkey,p_contacts in cdict.items():
                n = len(p_contacts) # Number of contacts
                new_contacts[lkey]['p1'].extend([p]*n) # e.g. [4, 4, 4, 4]
                new_contacts[lkey]['p2'].extend(p_contacts) # e.g. [243, 4538, 7,19]

        # Turn into a dataframe
        for lkey in lkeys:
            new_layer = Layer(label=lkey)
            for ckey,value in new_contacts[lkey].items():
                new_layer[ckey] = np.array(value, dtype=new_layer.meta[ckey])
            new_contacts[lkey] = new_layer

        return new_contacts


    @staticmethod
    def remove_duplicates(df):
        ''' Sort the dataframe and remove duplicates -- note, not extensively tested '''
        p1 = df[['p1', 'p2']].values.min(1) # Reassign p1 to be the lower-valued of the two contacts
        p2 = df[['p1', 'p2']].values.max(1) # Reassign p2 to be the higher-valued of the two contacts
        df['p1'] = p1
        df['p2'] = p2
        df.sort_values(['p1', 'p2'], inplace=True) # Sort by p1, then by p2
        df.drop_duplicates(['p1', 'p2'], inplace=True) # Remove duplicates
        df = df[df['p1'] != df['p2']] # Remove self connections
        df.reset_index(inplace=True, drop=True)
        return df


class Person(sc.prettyobj):
    '''
    Class for a single person. Note: this is largely deprecated since sim.people
    is now based on arrays rather than being a list of people.
    '''
    def __init__(self, pars=None, uid=None, age=-1, sex=-1, contacts=None):
        self.uid         = uid # This person's unique identifier
        self.age         = cvd.default_float(age) # Age of the person (in years)
        self.sex         = cvd.default_int(sex) # Female (0) or male (1)
        self.contacts    = contacts # Contacts
        # self.infected = [] #: Record the UIDs of all people this person infected
        # self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection
        return


class FlexDict(dict):
    '''
    A dict that allows more flexible element access: in addition to obj['a'],
    also allow obj[0]. Lightweight implementation of the Sciris odict class.
    '''

    def __getitem__(self, key):
        ''' Lightweight odict -- allow indexing by number, with low performance '''
        try:
            return super().__getitem__(key)
        except KeyError as KE:
            try: # Assume it's an integer
                dictkey = self.keys()[key]
                return self[dictkey]
            except:
                raise sc.KeyNotFoundError(KE) # Raise the original error

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())


class Contacts(FlexDict):
    '''
    A simple (for now) class for storing different contact layers.

    Args:
        data (dict): a dictionary that looks like a Contacts object
        layer_keys (list): if provided, create an empty Contacts object with these layers
        kwargs (dict): additional layer(s), merged with data

    New in version 3.1.2: swapped order of arguments
    '''
    def __init__(self, data=None, layer_keys=None, **kwargs):
        data = sc.mergedicts(data, kwargs)
        if layer_keys is not None:
            for lkey in layer_keys:
                self[lkey] = Layer(label=lkey)
        if data:
            for lkey,layer_data in data.items():
                self[lkey] = Layer(**layer_data)
        return

    def __repr__(self):
        ''' Use slightly customized repr'''
        keys_str = ', '.join([str(k) for k in self.keys()])
        output = f'Contacts({keys_str})\n'
        for key in self.keys():
            output += f'\n"{key}": '
            output += self[key].__repr__() + '\n'
        return output


    def __len__(self):
        ''' The length of the contacts is the length of all the layers '''
        output = 0
        for key in self.keys():
            try:
                output += len(self[key])
            except: # pragma: no cover
                pass
        return output


    def add_layer(self, **kwargs):
        '''
        Small method to add one or more layers to the contacts. Layers should
        be provided as keyword arguments.

        **Example**::

            hospitals_layer = cv.Layer(label='hosp')
            sim.people.contacts.add_layer(hospitals=hospitals_layer)
        '''
        for lkey,layer in kwargs.items():
            if not isinstance(layer, Layer):
                try:
                    layer = Layer(layer, label=lkey)
                except Exception as E:
                    exc = type(E)
                    errormsg = f'Could not parse {type(layer)} as layer: must be Layer or dict'
                    raise exc(errormsg) from E
            layer.validate()
            self[lkey] = layer
        return


    def pop_layer(self, *args):
        '''
        Remove the layer(s) from the contacts.

        **Example**::

            sim.people.contacts.pop_layer('hospitals')

        Note: while included here for convenience, this operation is equivalent
        to simply popping the key from the contacts dictionary.
        '''
        for lkey in args:
            self.pop(lkey)
        return


    def to_graph(self): # pragma: no cover
        '''
        Convert all layers to a networkx MultiDiGraph

        **Example**::

            import networkx as nx
            sim = cv.Sim(pop_size=50, pop_type='hybrid').run()
            G = sim.people.contacts.to_graph()
            nx.draw(G)
        '''
        import networkx as nx
        H = nx.MultiDiGraph()
        for lkey,layer in self.items():
            G = layer.to_graph()
            H = nx.compose(H, nx.MultiDiGraph(G))
        return H



class Layer(FlexDict):
    '''
    A small class holding a single layer of contact edges (connections) between people.

    The input is typically three arrays: person 1 of the connection, person 2 of
    the connection, and the weight of the connection. Connections are undirected;
    each person is both a source and sink.

    This class is usually not invoked directly by the user, but instead is called
    as part of the population creation.

    Args:
        p1 (array): an array of N connections, representing people on one side of the connection
        p2 (array): an array of people on the other side of the connection
        beta (array): an array of weights for each connection
        label (str): the name of the layer (optional)
        kwargs (dict): other keys copied directly into the layer

    Note that all arguments (except for label) must be arrays of the same length,
    although not all have to be supplied at the time of creation (they must all
    be the same at the time of initialization, though, or else validation will fail).

    **Examples**::

        # Generate an average of 10 contacts for 1000 people
        n = 10_000
        n_people = 1000
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        layer = cv.Layer(p1=p1, p2=p2, beta=beta, label='rand')
        layer = cv.Layer(dict(p1=p1, p2=p2, beta=beta), label='rand') # Alternate method

        # Convert one layer to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        layer2 = cv.Layer(**layer, index=index, self_conn=self_conn, label=layer.label)

    New in version 3.1.2: allow a single dictionary input
    '''

    def __init__(self, *args, label=None, **kwargs):
        self.meta = {
            'p1':    cvd.default_int,   # Person 1
            'p2':    cvd.default_int,   # Person 2
            'beta':  cvd.default_float, # Default transmissibility for this contact type
        }
        self.basekey = 'p1' # Assign a base key for calculating lengths and performing other operations
        self.label = label

        # Handle args
        kwargs = sc.mergedicts(*args, kwargs)

        # Initialize the keys of the layers
        for key,dtype in self.meta.items():
            self[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key,value in kwargs.items():
            self[key] = np.array(value, dtype=self.meta.get(key))

        # Set beta if not provided
        key = 'beta'
        if key not in kwargs.keys():
            self[key] = np.ones(len(self), dtype=self.meta[key])

        return


    def __len__(self):
        try:
            return len(self[self.basekey])
        except: # pragma: no cover
            return 0


    def __repr__(self):
        ''' Convert to a dataframe for printing '''
        namestr = self.__class__.__name__
        labelstr = f'"{self.label}"' if self.label else '<no label>'
        keys_str = ', '.join(self.keys())
        output = f'{namestr}({labelstr}, {keys_str})\n' # e.g. Layer("h", p1, p2, beta)
        output += self.to_df().__repr__()
        return output


    def __contains__(self, item):
        """
        Check if a person is present in a layer

        Args:
            item: Person index

        Returns: True if person index appears in any interactions

        """
        return (item in self['p1']) or (item in self['p2'])

    @property
    def members(self):
        """
        Return sorted array of all members
        """
        return np.unique([self['p1'], self['p2']])


    def meta_keys(self):
        ''' Return the keys for the layer's meta information -- i.e., p1, p2, beta '''
        return self.meta.keys()


    def validate(self, force=True):
        '''
        Check the integrity of the layer: right types, right lengths.

        If dtype is incorrect, try to convert automatically; if length is incorrect,
        do not.
        '''
        n = len(self[self.basekey])
        for key,dtype in self.meta.items():
            if dtype:
                actual = self[key].dtype
                expected = dtype
                if actual != expected:
                    self[key] = np.array(self[key], dtype=expected) # Probably harmless, so try to convert to correct type
            actual_n = len(self[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for layer key "{key}"; got {actual_n}' # We can't fix length mismatches
                raise TypeError(errormsg)
        return


    def pop_inds(self, inds):
        '''
        "Pop" the specified indices from the edgelist and return them as a dict.
        Returns in the right format to be used with layer.append().

        Args:
            inds (int, array, slice): the indices to be removed
        '''
        output = {}
        for key in self.meta_keys():
            output[key] = self[key][inds] # Copy to the output object
            self[key] = np.delete(self[key], inds) # Remove from the original
        return output


    def append(self, contacts):
        '''
        Append contacts to the current layer.

        Args:
            contacts (dict): a dictionary of arrays with keys p1,p2,beta, as returned from layer.pop_inds()
        '''
        for key in self.keys():
            new_arr = contacts[key]
            n_curr = len(self[key]) # Current number of contacts
            n_new = len(new_arr) # New contacts to add
            n_total = n_curr + n_new # New size
            self[key] = np.resize(self[key], n_total) # Resize to make room, preserving dtype
            self[key][n_curr:] = new_arr # Copy contacts into the layer
        return


    def to_df(self):
        ''' Convert to dataframe '''
        df = pd.DataFrame.from_dict(self)
        return df


    def from_df(self, df, keys=None):
        ''' Convert from a dataframe '''
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            self[key] = df[key].to_numpy()
        return self


    def to_graph(self): # pragma: no cover
        '''
        Convert to a networkx DiGraph

        **Example**::

            import networkx as nx
            sim = cv.Sim(pop_size=20, pop_type='hybrid').run()
            G = sim.people.contacts['h'].to_graph()
            nx.draw(G)
        '''
        import networkx as nx
        data = [np.array(self[k], dtype=dtype).tolist() for k,dtype in [('p1', int), ('p2', int), ('beta', float)]]
        G = nx.DiGraph()
        G.add_weighted_edges_from(zip(*data), weight='beta')
        nx.set_edge_attributes(G, self.label, name='layer')
        return G


    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all of the contacts
        associated with a subset of the people in this layer. Since contacts are bidirectional
        it's necessary to check both P1 and P2 for the target indices. The return type is a Set
        so that there is no duplication of indices (otherwise if the Layer has explicit
        symmetric interactions, they could appear multiple times). This is also for performance so
        that the calling code doesn't need to perform its own unique() operation. Note that
        this cannot be used for cases where multiple connections count differently than a single
        infection, e.g. exposure risk.

        Args:
            inds (array): indices of people whose contacts to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array): a set of indices for pairing partners

        Example: If there were a layer with
        - P1 = [1,2,3,4]
        - P2 = [2,3,1,4]
        Then find_contacts([1,3]) would return {1,2,3}
        """

        # Check types
        if not isinstance(inds, np.ndarray):
            inds = sc.toarray(inds)
        if inds.dtype != np.int64:  # pragma: no cover # This is int64 since indices often come from cv.true(), which returns int64
            inds = np.array(inds, dtype=np.int64)

        # Find the contacts
        contact_inds = cvu.find_contacts(self['p1'], self['p2'], inds)
        if as_array:
            contact_inds = np.fromiter(contact_inds, dtype=cvd.default_int)
            contact_inds.sort()  # Sorting ensures that the results are reproducible for a given seed as well as being identical to previous versions of Covasim

        return contact_inds


    def update(self, people, frac=1.0):
        '''
        Regenerate contacts on each timestep.

        This method gets called if the layer appears in ``sim.pars['dynam_layer']``.
        The Layer implements the update procedure so that derived classes can customize
        the update e.g. implementing over-dispersion/other distributions, random
        clusters, etc.

        Typically, this method also takes in the ``people`` object so that the
        update can depend on person attributes that may change over time (e.g.
        changing contacts for people that are severe/critical).

        Args:
            people (People): the Covasim People object, which is usually used to make new contacts
            frac (float): the fraction of contacts to update on each timestep
        '''
        # Choose how many contacts to make
        pop_size   = len(people) # Total number of people
        n_contacts = len(self) # Total number of contacts
        n_new = int(np.round(n_contacts*frac)) # Since these get looped over in both directions later
        inds = cvu.choose(n_contacts, n_new)

        # Create the contacts, not skipping self-connections
        self['p1'][inds]   = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=cvd.default_int) # Choose with replacement
        self['p2'][inds]   = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=cvd.default_int)
        self['beta'][inds] = np.ones(n_new, dtype=cvd.default_float)
        return

