import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class CSVLog:

    def __init__(self, file, columns=None, autoflush=True, verbose=False, overwrite=False):
        self.file = file
        self.autoflush = autoflush
        self.verbose = verbose
            
        if os.path.exists(file) and not overwrite:
            self.tell('Loading existing file from {}'.format(file))
            self.df = pd.read_csv(file, sep='\t')
            self.columns = sorted(self.df.columns.values.tolist())
        else:
            self.tell('File {} does not exist or overwrite=True. Creating new frame.'.format(file))
            assert columns is not None, 'columns cannot be None'
            self.columns = sorted(columns)
            dir = os.path.dirname(self.file)
            if dir and not os.path.exists(dir): os.makedirs(dir)
            self.df = pd.DataFrame(columns=self.columns)
        
        self.defaults = {}

    def already_calculated(self, **kwargs):
        df = self.df
        if df.shape[0] == 0:
            return False
        if len(kwargs) == 0:
            kwargs = self.defaults
        for key,val in kwargs.items():
            df = df.loc[df[key] == val]
            if df.shape[0] == 0:
                return False
        return True

    def set_default(self, param, value):
        self.defaults[param] = value


    def add_layered_row(self, **kwargs):

        #print("CSVLog::add_layered_row()")
        #print('--defaults--\n\t', {self.defaults.keys})
        
        # set defaults
        for key in self.defaults.keys():
            if key not in kwargs:
                print("key ", {key}, "not found, setting default")
                kwargs[key]=self.defaults[key]

        local_columns = sorted(list(kwargs.keys()))
        values = [kwargs[col_i] for col_i in local_columns]

        #print(self.columns)
        #print(local_columns)
        #print(values)

        s = pd.Series(values, index=self.columns)
        self.df = self.df.append(s, ignore_index=True)             # deprecated as of pandas 2.0
        #self.df = pd.concat([self.df, s], ignore_index=True)            
        if self.autoflush: self.flush()
        self.tell(kwargs)


    def add_simple_row(self, **kwargs):

        """
        print("cvs_log::add_row()")
        print("defaults:", {self.defaults.keys})
        """

        # set defaults
        for key in self.defaults.keys():
            if key not in kwargs:
                #print("key ", {key}, "not found, setting default")
                kwargs[key]=self.defaults[key]

        local_columns = sorted(list(kwargs.keys()))
        values = [kwargs[col_i] for col_i in local_columns]

        """
        print(self.columns)
        print(local_columns)
        print(values)
        """

        s = pd.Series(values, index=self.columns)
        self.df = self.df.append(s, ignore_index=True)             # deprecated as of pandas 2.0
        #self.df = pd.concat([self.df, s], ignore_index=True)            
        if self.autoflush: self.flush()
        self.tell(kwargs)

    def flush(self):
        self.df.to_csv(self.file, index=False, sep='\t')

    def tell(self, msg):
        if self.verbose: print(msg)



