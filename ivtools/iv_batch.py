

import concurrent
import copy
from collections import defaultdict, OrderedDict

import pandas as pd
from pvlib import pvsystem
from sklearn import linear_model

from ivtools import iv_curve


def main():
    pass


class IVBatch(object):
    """
    Class for analyzing time-series IV curve data.
    """

    def __init__(self, df, dt_col, multi_sys=None):
        """Break up incoming data into dictionary of dataframes ({key: df, ...} for individual systems).

        The incoming data frame is assumed to have a column that uniquely identifies which system or module the IV curve
        belongs to.

        Args:
            df (pd.DataFrame): tabular, time-series IV data
            dt_col (str): column name of datetime
            multi_sys (str): column name that identifies unique systems (if multiple are in df)

        Returns:
            None
        """
        self.df_dict = OrderedDict()
        df[multi_sys] = df[multi_sys].astype(str)
        if multi_sys is not None:
            try:
                for sys_name, sys_data in df.groupby(multi_sys):
                    sys_data.index = pd.to_datetime(sys_data[dt_col])
                    sys_data.drop(dt_col, inplace=True, axis=1)
                    sys_data.drop(multi_sys, inplace=True, axis=1)
                    self.df_dict[sys_name] = sys_data.copy()
            except KeyError:
                raise KeyError('Must provide column name that indicates which row belongs to '
                               'which system as multi_sys argument.')
        else:
            df.index = pd.to_datetime(df[dt_col])
            self.df_dict['0'] = df.copy()

        self.masks = defaultdict(list)

    def add_system(self, key, df, overwrite=False):
        """Add system to object.

        Args:
            key (str): unique system identifier
            overwrite (bool, optional): permission to overwrite existing system

        Returns:
            None
        """
        key = str(key)
        if key in self.df_dict and not overwrite:
            raise KeyError('Key already exists.  Provide a unique system name.')
        self.df_dict[key] = df.copy()

    def remove_system(self, key):
        """Remove tracked system.

        Args:
            key (str): unique system identifier

        Returns:
            pd.DataFrame removed from monitoring
        """
        return self.pop(key)

    def pop(self, key):
        """Remove tracked system.

        Args:
            key (str): system identifier

        Returns:
            pd.DataFrame
        """
        key = str(key)
        try:
            return self.df_dict.pop(key)
        except KeyError:
            raise KeyError('Cannot find supplied key.')

    def clear_masks(self, sys_names=None, mask_names=None):
        """Clear masks for systems.  Note that this does not delete the columns in the data frames.  It
        simply 'untracks' the columns when the masks are applied.

        Args:
            sys_names (list): systems names to clear masks
            mask_names (list): list of mask names to clear

        Returns:
            None
        """
        if sys_names is None and mask_names is None:
            self.masks = defaultdict(list)
        elif sys_names is not None and mask_names is None:
            for sys_name in [str(name) for name in sys_names]:
                self.masks[sys_name] = []
        elif sys_names is None and mask_names is not None:
            for sys_name in self.list_systems():
                self.masks[sys_name] = [i for i in self.masks[sys_name] if i not in mask_names]
        else:
            for sys_name in [str(name) for name in sys_names]:
                self.masks[sys_name] = [i for i in self.masks[sys_name] if i not in mask_names]

    def add_mask(self, col, f, sys_names=None):
        """Add data mask/filter.  Mask names are stored in self.masks dictionary.

        Args:
            col (str): column name to apply filter to
            f (callable): function that takes a single, scalar argument
            sys_names (list): list of system identifiers

        Returns:
            None
        """
        if sys_names is None:
            sys_names = self.list_systems()
        for sys_name in sys_names:
            sys_df = self.df_dict[sys_name]
            mask = sys_df[col].apply(lambda x: f(x))
            assert mask.dtype == bool, 'Masks must be of type \'bool\'.'
            sys_df[col + '_mask'] = mask
            if col + '_mask' not in self.masks[sys_name]:
                self.masks[sys_name].append(col + '_mask')

    def list_masks(self, sys_name=None):
        """List mask column names.

        Args:
            sys_name (str, optional): name of specific system to get masks

        Returns:
            list: column names
        """
        if sys_name is None:
            return self.masks
        else:
            return self.masks[sys_name]

    def apply_masks(self, sys_name):
        """Apply all masks for a specific system.  Masks are applied using a mask_matrix.all(axis=1) call
        (i.e. all masks must be True).

        Args:
            sys_name (str): Key for self.df_dict

        Returns:
            np.array: bool mask
        """
        sys_df = self.df_dict[sys_name]
        masks = self.list_masks(sys_name=sys_name)
        return sys_df[masks].all(axis=1)

    def __iter__(self):
        """Enable iteration over systems stored in self.df_dict.

        Yields:
            str, pd.DataFrame: system name, system dataframe
        """
        for key, val in self.df_dict.items():
            yield key, val

    def list_systems(self):
        """Returns unique system names.

        Returns:
            list: dictionary keys unique to each system
        """
        return sorted(list(self.df_dict.keys()))

    def keys(self):
        """Alias for list_systems.  Provides an expected function name since accessing systems is done
        through a dictionary lookup.

        Returns:
            list: dictionary keys unique to each system
        """
        return self.list_systems()

    def get_system(self, key, cols=None):
        """Return invidiual system.

        Args:
            key (str): system identifier
            cols (list): list of specific column names to return

        Returns:
            pd.DataFrame
        """
        key = str(key)
        if cols is None:
            return self.df_dict[key]
        else:
            return self.df_dict[key][cols]

    def param_extraction(self, v_array_name='voltage_array', i_array_name='current_array', suffix='_extracted'):
        """Extract summary parameters from IV curve.

        Summary IV parameters are:
            ff: fill factor
            ipmax: current at maximum power point (mpp)
            ir: current at intersection of series and shunt resistance tangent lines
            isc: short-circuit current
            ix: current at 1/2 voc
            ixx: current at 1/2 (vpmax + voc)
            pmax: maximum power
            rs: series resistance
            rsh: shunt resistance
            voc: open-circuit voltage
            vpmax: voltage at mpp
            vr: voltage at intersection of series and shut resistance tanget lines

        Args:
            v_array_name (str, optional): column name for voltage array
            i_array_name (str, optional): column name for current array
            suffix (str, optional): string to add to end of new column names

        Returns:
            None
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            appended_dfs = \
                {executor.submit(_param_extraction, self.df_dict[sys_name], v_array_name=v_array_name,
                                 i_array_name=i_array_name, suffix=suffix): sys_name for sys_name in self.df_dict}
            for future in concurrent.futures.as_completed(appended_dfs):
                sys_name = appended_dfs[future]
                self.df_dict[sys_name] = future.result()

    def mismatch_identification(self, v_array_name='voltage_array', i_array_name='current_array'):
        """Locate mismatches in measured IV curves.

        Args:
            v_array_name (str, optional): column name for voltage array
            i_array_name (str, optional): column name for current array

        Returns:
            None
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            appended_dfs = \
                {executor.submit(_mismatch_identification, self.df_dict[sys_name], v_array_name=v_array_name,
                                 i_array_name=i_array_name): sys_name for sys_name in self.df_dict}
            for future in concurrent.futures.as_completed(appended_dfs):
                sys_name = appended_dfs[future]
                self.df_dict[sys_name] = future.result()

    def flag_outliers_iqr(self, col, sys_names=None):
        """Find outliers using 1.5 * inter-quartile range.  Masks will be added as columns with the name
        being col + '_iqr_mask' (col being a column in cols list).

        Args:
            sys_names (list): list of keys for accessing site-specific dfs
            col (str): column to make mask for

        Returns:
            None
        """
        sys_names = sys_names if sys_names is not None else self.list_systems()
        for sys_name in sys_names:
            sys_name = str(sys_name)
            sys_df = self.df_dict[sys_name]
            low = sys_df[col].quantile(.25)
            high = sys_df[col].quantile(.75)
            iqr = high - low
            low = low - (1.5 * iqr)
            high = high + (1.5 * iqr)
            mask = sys_df[col].between(low, high, inclusive=True)
            sys_df[col + '_iqr_mask'] = mask
            if col + '_iqr_mask' not in self.masks[sys_name]:
                self.masks[sys_name].append(col + '_iqr_mask')
            self.df_dict[sys_name] = sys_df.copy()

    def add_single_diode_params(self, params_per_sys, extracted_params=True):
        """Compute expected IV parameters using single diode equation.

        Args:
            params_per_sys (dict of dict): parameters needed for single-diode calculation from PVLIB; organized as follows:
                {'system_name':
                    {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME,
                     'n_mod': int},
                ...}
            extracted_params (bool, optional): Use extracted IV parameters (from measured IV curves) if possible.
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            appended_dfs = \
                {executor.submit(_add_single_diode_params, self.df_dict[sys_name],
                                 params_per_sys[sys_name], extracted_params): sys_name for sys_name in params_per_sys}
            for future in concurrent.futures.as_completed(appended_dfs):
                sys_name = appended_dfs[future]
                self.df_dict[sys_name] = future.result()

    def add_sapm_params(self, params_per_sys, extracted_params=True):
        """Compute expected IV parameters using Sandia Array Performance Model.

        Parameters will be added to the self.df_dict dataframes using the following columns:
            isc_sapm, voc_sapm, ipmax_sapm, vpmax_sapm, pmax_sapm, ix_sapm, ixx_sapm

        The measured values will also be normalized by single-diode parameters and stored in columns with names:
            isc_norm_sapm, voc_norm_sapm, ipmax_norm_sapm, vpmax_norm_sapm, pmax_norm_sapm, ix_norm_sapm, ixx_norm_sapm

        Args:
            params_per_sys (dict of dict): parameters needed for single-diode calculation from PVLIB; organized as follows:
                {'system_name': {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME,
                                 'n_mod': number of modules, 'aoi': angle of incidence}, ...}
            extracted_params (bool, optional): use extracted parameters (from param_extraction() method) or pre-supplied
                measurements.

        Returns:
            None
        """
        # This function isn't implemented yet.  SAPM requires more info than single diode and LFM.  Still deciding
        # on what can/should be supplied by user and would be computed internally.

        # The 'extra' inputs are:
        #     * zenith (sun angle) for relative airmass
        #     * pressure at site for converting relative airmass to absolute airmass
        #     * poa_direct, poa_diffuse, and aoi (angle of incidence) for effective irradiance
        raise NotImplementedError('Method coming soon.  Try add_single_diode_params() or add_lfm_params() for now.')

    def add_lfm_params(self, params_per_sys, v_array_name='voltage_array', i_array_name='current_array'):
        """Compute LFM parameters from IV curves and add columns to data frames.

        Args:
            v_array_name (str): name of column with voltages
            i_array_name (str): name of column with currents
            params_per_sys (dict): the dictionary should be organized as shown below
                {'system_name': {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME, 'n_mod': int}, ...}

        Returns:
            None
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            appended = \
                {executor.submit(_add_lfm_params, self.df_dict[sys_name], params_per_sys[sys_name],
                                 v_array_name=v_array_name, i_array_name=i_array_name): sys_name
                 for sys_name in params_per_sys}
            for future in concurrent.futures.as_completed(appended):
                sys_name = appended[future]
                self.df_dict[sys_name] = future.result()

    def add_single_diode_params_via_extraction(self, params_per_sys):
        """Compute expected IV parameters using single-diode equation.  Summary parameters (isc, voc, etc) will be
        extracted using methods implemented in iv_curve.py file.  This function is more expensive than the related
        add_single_diode_params(...).

        Single-diode parameters will be added to the self.df_dict dataframes using the following columns:
            isc_single_diode, voc_single_diode, ipmax_single_diode, vpmax_single_diode, pmax_single_diode,
            ix_single_diode, ixx_single_diode, rs_single_diode, rsh_single_diode, iratio_single_diode,
            vratio_single_diode, ir_single_diode, vr_single_diode

        The measured values will also be normalized by single-diode parameters and stored in columns with names
        listed above with the addition of 'norm' before single_diode.  For example, df[isc] / df[isc_single_diode]
        will be stored in a column named is_norm_single_diode.

        Args:
            params_per_sys (dict of dict): parameters needed for single-diode calculation from PVLIB:
                {sys_name:
                    {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME,
                     'n_mod': int},
                ...}

        Returns:
            None
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            appended = \
                {executor.submit(_add_single_diode_iv_curve, self.df_dict[sys_name], params_per_sys[sys_name]): sys_name
                 for sys_name in params_per_sys}
            for future in concurrent.futures.as_completed(appended):
                sys_name = appended[future]
                self.df_dict[sys_name] = future.result()

        self.param_extraction(v_array_name='voltage_array_single_diode',
                              i_array_name='current_array_single_diode', suffix='_single_diode')

        for sys_name in self.df_dict.keys():
            sys_df = self.df_dict[sys_name]
            sys_df['isc_norm_single_diode'] = sys_df['isc_extracted'] / sys_df['isc_single_diode']
            sys_df['voc_norm_single_diode'] = sys_df['voc_extracted'] / sys_df['voc_single_diode']
            sys_df['ipmax_norm_single_diode'] = sys_df['ipmax_extracted'] / sys_df['ipmax_single_diode']
            sys_df['vpmax_norm_single_diode'] = sys_df['vpmax_extracted'] / sys_df['vpmax_single_diode']
            sys_df['pmax_norm_single_diode'] = sys_df['pmax_extracted'] / sys_df['pmax_single_diode']
            sys_df['ix_norm_single_diode'] = sys_df['ix_extracted'] / sys_df['ix_single_diode']
            sys_df['ixx_norm_single_diode'] = sys_df['ixx_extracted'] / sys_df['ixx_single_diode']
            sys_df['rs_norm_single_diode'] = sys_df['rs_extracted'] / sys_df['rs_single_diode']
            sys_df['rsh_norm_single_diode'] = sys_df['rsh_extracted'] / sys_df['rsh_single_diode']
            sys_df['ff_norm_single_diode'] = sys_df['ff_extracted'] / sys_df['ff_single_diode']
            sys_df['ir_norm_single_diode'] = sys_df['ir_extracted'] / sys_df['ir_single_diode']
            sys_df['vr_norm_single_diode'] = sys_df['vr_extracted'] / sys_df['vr_single_diode']
            sys_df['iratio_norm_single_diode'] = sys_df['iratio_extracted'] / sys_df['iratio_single_diode']
            sys_df['vratio_norm_single_diode'] = sys_df['vratio_extracted'] / sys_df['vratio_single_diode']
            self.df_dict[sys_name] = sys_df.copy()

    def calc_degradation_ols(self, col, apply_masks=False):
        """Calculate yearly degradation rate of a given column of data using ordinary least squares.

        Args:
            col (str): column name
            apply_masks (bool, optional): whether or not to apply masks (from self.masks)

        Returns:
            dict of degradation parameters per system
            {'system_name':
                {'rd': degradation (100 * slople / intercept),
                 'm': slope of OLS line,
                 'b': intercept of OLS line,
                 'model': sklearn.linear_model.LinearRegression object},
            ...}
        """
        return_dict = {}
        for sys_name, sys_df in sorted(self.df_dict.items()):
            if apply_masks:
                sys_df = sys_df[self.apply_masks(sys_name)]
            xvals = (sys_df.index - sys_df.index[0]).total_seconds().values / (60 * 60 * 24 * 365)  # yearly degradation
            yvals = sys_df[col].values
            xvals = xvals.reshape(-1, 1)
            yvals = yvals.reshape(-1, 1)
            lr = linear_model.LinearRegression()
            lr.fit(xvals, yvals)
            m = lr.coef_[0][0]
            b = lr.intercept_[0]
            print('System {} degradation rate: {:.3f}'.format(sys_name, 100 * m / b))
            return_dict[sys_name] = {'rd': 100 * m / b, 'm': m, 'b': b, 'model': copy.deepcopy(lr)}
        return return_dict


def _add_single_diode_iv_curve(sys_df, sys_params):
    """Add IV curve generated by single diode model.

    Args:
        sys_df (pd.DataFrame): time-series data that must contain 'g_poa' and 'temp_air' parameters
        sys_params (dict): organized like so:
            {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME, 'n_mod': int}

    Returns:
        pd.DataFrame with new columns (voltage_array_single_diode and current_array_single_diode)
    """
    new_df = []
    if 'temp_cell' not in sys_df.keys():
        sys_df = _add_cell_module_temp(sys_df, sys_params['celltemp_model'])
    for i in range(len(sys_df)):
        ser = sys_df.iloc[i]
        # final two args are the band gap (assumed Si) and temperature dependence of bandgap at SRC
        il, i0, rs, rsh, nnsvth = pvsystem.calcparams_desoto(ser['g_poa'], ser['temp_cell'],
                                                             sys_params['mod']['alpha_sc'],
                                                             sys_params['mod'], 1.121, -0.0002677)
        params = pvsystem.singlediode(il, i0, rs, rsh, nnsvth, ivcurve_pnts=250)
        new_df.append({'voltage_array_single_diode': params['v'] * sys_params['n_mod'],
                       'current_array_single_diode': params['i']})

    new_df = pd.DataFrame(new_df, index=sys_df.index)

    sys_df['voltage_array_single_diode'] = new_df['voltage_array_single_diode']
    sys_df['current_array_single_diode'] = new_df['current_array_single_diode']

    return sys_df.copy()


def _param_extraction(sys_df, v_array_name='voltage_array', i_array_name='current_array', suffix='_extracted'):
    """Extract summary parameters from IV curve.

    Summary IV parameters are:
        ff: fill factor
        ipmax: current at maximum power point (mpp)
        ir: current at intersection of series and shunt resistance tangent lines
        isc: short-circuit current
        ix: current at 1/2 voc
        ixx: current at 1/2 (vpmax + voc)
        pmax: maximum power
        rs: series resistance
        rsh: shunt resistance
        voc: open-circuit voltage
        vpmax: voltage at mpp
        vr: voltage at intersection of series and shut resistance tanget lines

    Args:
        sys_df (pd.DataFrame): time-series data that must contain IV curves
        v_array_name (str, optional): column name for voltage array
        i_array_name (str, optional): column name for current array
        suffix (str, optional): string to add to end of new column names

    Returns:
        pd.DataFrame with additional columns added
    """
    params = []
    for i in range(len(sys_df)):
        try:
            ser = sys_df.iloc[i]
            x, y = ser[v_array_name], ser[i_array_name]
            curve = iv_curve.IVCurve(x, y)
            params.append(curve.extract_components())
        except ValueError:
            params.append({})

    components_df = pd.DataFrame(params, index=sys_df.index)
    # rename columns -- avoids some columns of initial df being overwritten
    components_df = components_df.rename({'ff': 'ff' + suffix, 'ipmax': 'ipmax' + suffix, 'ir': 'ir' + suffix,
                                          'isc': 'isc' + suffix, 'ix': 'ix' + suffix, 'ixx': 'ixx' + suffix,
                                          'pmax': 'pmax' + suffix, 'rs': 'rs' + suffix, 'rsh': 'rsh' + suffix,
                                          'voc': 'voc' + suffix, 'vpmax': 'vpmax' + suffix, 'vr': 'vr' + suffix,
                                          'iratio': 'iratio' + suffix, 'vratio': 'vratio' + suffix}, axis='columns')

    for key in components_df.keys():
        sys_df[key] = components_df[key]

    return sys_df.copy()


def _add_single_diode_params(sys_df, sys_params, extracted_params=True):
    """Compute expected IV parameters using single-diode equation.

    Single-diode parameters will be added to the self.df_dict dataframes using the following columns:
        isc_single_diode, voc_single_diode, ipmax_single_diode, vpmax_single_diode, pmax_single_diode,
        ix_single_diode, ixx_single_diode

    The measured values will also be normalized by single-diode parameters and stored in columns with names:
        isc_norm_single_diode, voc_norm_single_diode, ipmax_norm_single_diode, vpmax_norm_single_diode,
        pmax_norm_single_diode, ix_norm_single_diode, ixx_norm_single_diode

    Args:
        sys_df (pd.DataFrame): time-series data that must contain IV curves
        sys_params (dict): parameters needed for single-diode calculation from PVLIB; organized as follows:
            {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME, 'n_mod': int}

    Returns:
        pd.DataFrame with additional columns added
    """
    diode_params = []
    if 'temp_cell' not in sys_df.keys():
        sys_df = _add_cell_module_temp(sys_df, sys_params['celltemp_model'])
    for i in range(len(sys_df)):
        ser = sys_df.iloc[i]
        # final two args are the band gap (assumed Si) and temperature dependence of bandgap at SRC
        il, i0, rs, rsh, nnsvth = pvsystem.calcparams_desoto(ser['g_poa'], ser['temp_cell'],
                                                             sys_params['mod']['alpha_sc'],
                                                             sys_params['mod'], 1.121, -0.0002677)
        params = pvsystem.singlediode(il, i0, rs, rsh, nnsvth)
        diode_params.append(
            {'isc_single_diode': params['i_sc'], 'voc_single_diode': params['v_oc'] * sys_params['n_mod'],
             'ipmax_single_diode': params['i_mp'], 'vpmax_single_diode': params['v_mp'] * sys_params['n_mod'],
             'pmax_single_diode': params['p_mp'] * sys_params['n_mod'],
             'ix_single_diode': params['i_x'], 'ixx_single_diode': params['i_xx']}
        )

    param_df = pd.DataFrame(diode_params, index=sys_df.index)
    for key in param_df:
        sys_df[key] = param_df[key]

    if extracted_params:
        try:
            sys_df['isc_norm_single_diode'] = sys_df['isc_extracted'] / sys_df['isc_single_diode']
            sys_df['voc_norm_single_diode'] = sys_df['voc_extracted'] / sys_df['voc_single_diode']
            sys_df['ipmax_norm_single_diode'] = sys_df['ipmax_extracted'] / sys_df['ipmax_single_diode']
            sys_df['vpmax_norm_single_diode'] = sys_df['vpmax_extracted'] / sys_df['vpmax_single_diode']
            sys_df['pmax_norm_single_diode'] = sys_df['pmax_extracted'] / sys_df['pmax_single_diode']
            sys_df['ix_norm_single_diode'] = sys_df['ix_extracted'] / sys_df['ix_single_diode']
            sys_df['ixx_norm_single_diode'] = sys_df['ixx_extracted'] / sys_df['ixx_single_diode']
        except KeyError:
            raise KeyError('Call param_extraction() if extracted_params = True.')
    else:
        sys_df['isc_norm_single_diode'] = sys_df['isc'] / sys_df['isc_single_diode']
        sys_df['voc_norm_single_diode'] = sys_df['voc'] / sys_df['voc_single_diode']
        sys_df['ipmax_norm_single_diode'] = sys_df['ipmax'] / sys_df['ipmax_single_diode']
        sys_df['vpmax_norm_single_diode'] = sys_df['vpmax'] / sys_df['vpmax_single_diode']
        sys_df['pmax_norm_single_diode'] = sys_df['pmax'] / sys_df['pmax_single_diode']
        # data doesn't usually contain ix, ixx
        try:
            sys_df['ix_norm_single_diode'] = sys_df['ix'] / sys_df['ix_single_diode']
        except KeyError:
            pass
        try:
            sys_df['ixx_norm_single_diode'] = sys_df['ixx'] / sys_df['ixx_single_diode']
        except KeyError:
            pass

    return sys_df.copy()


def _mismatch_identification(sys_df, v_array_name='voltage_array', i_array_name='current_array'):
    """Locate mismatches in measured IV curves.

    Args:
        sys_df (pd.DataFrame): data that contain IV curve in columns v_array_name and i_array_name
        v_array_name (str, optional): column name for voltage array
        i_array_name (str, optional): column name for current array

    Returns:
        pd.DataFrame with additional columns added
    """
    mismatches = []
    for i in range(len(sys_df)):
        try:
            ser = sys_df.iloc[i]
            x, y = ser[v_array_name], ser[i_array_name]
            curve = iv_curve.IVCurve(x, y)
            mm = curve.locate_mismatch()
            mismatches.append({'has_mismatch': len(mm) > 0, 'n_mismatch': len(mm)})
        except ValueError:
            mismatches.append({})

    mismatches_df = pd.DataFrame(mismatches, index=sys_df.index)

    for key in mismatches_df:
        sys_df[key] = mismatches_df[key].fillna(False)

    return sys_df.copy()


def _add_lfm_params(sys_df, sys_params, v_array_name='voltage_array', i_array_name='current_array'):
    """Compute loss factor model (LFM) IV parameters.

    LFM parameters will be added to the self.df_dict dataframes using the following columns:
        nisct_lfm, nrsc_lfm, nimp_lfm, nroc_lfm, nvmp_lfm, nvoct_lfm, pimp_lfm, pvmp_lfm

    Args:
        sys_df (pd.DataFrame): time-series data that must contain IV curves
        sys_params (dict): parameters needed for single-diode calculation from PVLIB; organized as follows:
            {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME, 'n_mod': int}

    Returns:
        pd.DataFrame with additional columns added
    """
    lfm_params = []
    if 'temp_cell' not in sys_df.keys():
        sys_df = _add_cell_module_temp(sys_df, sys_params['mod'])
    for i in range(len(sys_df)):
        try:
            ser = sys_df.iloc[i]
            x, y = ser[v_array_name], ser[i_array_name]
            curve = iv_curve.IVCurve(x, y)
            lfm_params.append(
                    curve.calc_lfm_params(sys_params['mod'], sys_params['n_mod'], ser['g_poa'], ser['temp_cell'])
            )
        except ValueError:
            lfm_params.append({})

    param_df = pd.DataFrame(lfm_params, index=sys_df.index)

    columns = {'nisct': 'nisct_lfm', 'nrsc': 'nrsc_lfm', 'nimp': 'nimp_lfm', 'nroc': 'nroc_lfm', 'nvmp': 'nvmp_lfm',
               'nvoct': 'nvoct_lfm', 'pimp': 'ipmax_lfm', 'pvmp': 'vpmax_lfm', 'pmp': 'pmax_lfm'}

    param_df = param_df.rename(columns=columns)

    for key in param_df:
        sys_df[key] = param_df[key]

    return sys_df.copy()


def _add_sapm_params(sys_df, sys_params, v_array_name='voltage_array',
                     i_array_name='curent_array', extracted_params=True):
    """Compute IV parameters using Sandia Array Performance Model (SAPM).

    SAPM parameters will be added to the self.df_dict dataframes using the following columns:
        isc_sapm, voc_sapm, ipmax_sapm, vpmax_sapm, pmax_sapm, ix_sapm, ixx_sapm

    Args:
        sys_df (pd.DataFrame): time-series data that must contain IV curves
        sys_params (dict): parameters needed for single-diode calculation from PVLIB; organized as follows:
            {'mod': pvlib.pvsystem.retrieve_same(...).SYSTEM_NAME, 'n_mod': int}

    Returns:
        pd.DataFrame with additional columns added
    """
    sapm_params = []
    # for i in range(len(sys_df)):
        # ser = sys_df.iloc[i]
        # rel_airmass = pvlib.atmosphere.relativeairmass()
        # abs_airmass = pvlib.atmosphere.absoluteairmass(rel_airmass)
        # aoi = pvlib.irradiance.aoi()
        # effective_irradiance = pvsystem.sapm_effective_irradiance(direct, diffuse, abs_airmass, aoi, sys_params['mod'])
        # params = pvsystem.sapm(effective_irradiance, ser['temp'], params_per_sys['mod'])
        # sapm_params.append(
        #     {'isc_sapm': params['i_sc'], 'voc_sapm': params['v_oc'] * sys_params['n_mod'],
        #      'ipmax_sapm': params['i_mp'], 'vpmax_sapm': params['v_mp'] * sys_params['n_mod'],
        #      'pmax_sapm': params['p_mp'] * sys_params['n_mod'],
        #      'ix_sapm': params['i_x'], 'ixx_sapm': params['i_xx']}
        # )

    # param_df = pd.DataFrame(sapm_params, index=sys_df.index)
    # for key in param_df:
    #     sys_df[key] = param_df[key]
    #
    # if extracted_params:
    #     try:
    #         sys_df['isc_norm_sapm'] = sys_df['isc_extracted'] / sys_df['isc_sapm']
    #         sys_df['voc_norm_sapm'] = sys_df['voc_extracted'] / sys_df['voc_sapm']
    #         sys_df['ipmax_norm_sapm'] = sys_df['ipmax_extracted'] / sys_df['ipmax_sapm']
    #         sys_df['vpmax_norm_sapm'] = sys_df['vpmax_extracted'] / sys_df['vpmax_sapm']
    #         sys_df['pmax_norm_sapm'] = sys_df['pmax_extracted'] / sys_df['pmax_sapm']
    #         sys_df['ix_norm_sapm'] = sys_df['ix_extracted'] / sys_df['ix_sapm']
    #         sys_df['ixx_norm_sapm'] = sys_df['ixx_extracted'] / sys_df['ixx_sapm']
    #     except KeyError:
    #         raise KeyError('Call param_extraction() if extracted_params = True.')
    # else:
    #     sys_df['isc_norm_sapm'] = sys_df['isc'] / sys_df['isc_sapm']
    #     sys_df['voc_norm_sapm'] = sys_df['voc'] / sys_df['voc_sapm']
    #     sys_df['ipmax_norm_sapm'] = sys_df['ipmax'] / sys_df['ipmax_sapm']
    #     sys_df['vpmax_norm_sapm'] = sys_df['vpmax'] / sys_df['vpmax_sapm']
    #     sys_df['pmax_norm_sapm'] = sys_df['pmax'] / sys_df['pmax_sapm']
    #     # measured data doesn't usually contain ix or ixx but try to normalize anyway
    #     try:
    #         sys_df['ix_norm_sapm'] = sys_df['ix'] / sys_df['ix_sapm']
    #     except KeyError:
    #         pass
    #     try:
    #         sys_df['ixx_norm_sapm'] = sys_df['ixx'] / sys_df['ixx_sapm']
    #     except KeyError:
    #         pass
    #
    # self.df_dict[sys_name] = sys_df.copy()


def _add_cell_module_temp(sys_df, mod):
    if 'wind_speed' not in sys_df.keys():
        sys_df['wind_speed'] = 0

    vals = pvsystem.sapm_celltemp(sys_df['g_poa'], sys_df['wind_speed'], sys_df['temp_air'], mod)

    for key in vals.keys():
        sys_df[key] = vals[key]

    return sys_df


if __name__ == '__main__':
    main()
