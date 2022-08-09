import os
import pandas as pd

import bootstrap
import df_utils
import names


class stochastic_benchmark:
    def __init__(self, here=os.getcwd(), recover=True):
        # parameters should be user specified

        # Initialize saving/filename utilities
        self.here = names.path(here)

        # Check if checkpointed dataframes all exist -> if so, read all and exit

        # Otherwise begin recovery process
        # Read raw datafiles: expect that all files in exp_raw folder are pkls to go in
        # If parameters need to be recovered from filenames, should be specified here
        self.raw_data = df_utils.read_exp_raw(self.here)
        # Don't save this step since relatively cheap

        # Bootstrap
        if os.path.exists(self.here.bootstrap):
            self.bs_df = pd.read_pickle(self.here.bootstrap)
        else:
            # figure out how to store these parameters
            self.bs_df = bootstrap.Bootstrap(
                self.raw_data, group_on, bootstraps)
            self.bs_df.to_pick(self.here.bootstrap)

        # Stats stuff
