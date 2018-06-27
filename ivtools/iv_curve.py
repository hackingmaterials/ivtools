

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn import linear_model

from ivtools import utils
from ivtools import detect_peaks


def main():
    pass


class IVCurve(object):

    """
    This class is used for processing, analyzing, and extracting information for current-voltage (IV) curves.
    Current work is focused on building out the parameter extraction for modeling methods (LFM, SAPM).
    """

    def __init__(self, voltage, current, name=None, low_voltage_cutoff=25, min_pts=25):
        """Initialize object.

        Args:
            voltage (np.array): voltage measurements
            current (np.array): current measurements
            name (str): name for sample
            low_voltage_cutoff (float): voltages near Isc can be very noisy - ignore any voltages < this value.
            min_pts (int): require a minimum number of points to exist for analysis - this is after removing low
                voltage points.

        Returns:
            None
        """
        # voltage = np.asarray(voltage).flatten().astype(float)
        # current = np.asarray(current).flatten().astype(float)
        mask = voltage > low_voltage_cutoff
        x = voltage[mask].astype(float)
        y = current[mask].astype(float)

        # assure adequate amount of data points
        if len(x) < min_pts or len(y) < min_pts:
            raise ValueError('Not enough points in IV curve.')

        # sorting IV curve by voltage (needed for interpolation)
        xy = np.column_stack((x, y))
        xy = xy[xy[:, 0].argsort()]
        x = xy[:, 0]
        y = xy[:, 1]

        self.v_ = x
        self.i_ = y

        if name is not None:
            self.name_ = name
        else:
            self.name_ = 0

        self.interp_func_ = None

    def is_smooth(self, pct_change=.1, start=10, end=10):
        """Check how smooth the interpolated curve is.  A curve is adequately smooth if the maximum percent change
        between adjacent points is < pct_change.  This could/should be made more robust.

        Args:
            pct_change (float): maximum allowed percent change to determine smoothness
            start (int): check smoothness of curve from array[start:] (ignore first 'n' points)
            end (int): check smoothness of curve up to array[:-end] (ignore last 'n' points)

        Returns:
            bool if max pct change is < pct_change
        """
        v, i = self.smooth()
        v = v[start:-end]
        i = i[start:-end]
        return np.max(np.abs(np.diff(i) / i[:-1])) < pct_change

    def extract_components(self, isc_lim=.1, voc_lim=.1, mode='all'):
        """Extract Common IV curve characteristics.  Optionally returns parameters needed for modeling with the
        Loss Factor Model and the Sanida Array Performance Model.

        Isc and Voc are found by performing a linear regression on endpoints of the data.
        Rs and Rsh are determined used slopes from the above regression.

        The characterisitics are:
            Isc: short-circuit current
            Voc: open-circuit voltage
            Ipmax: current at maximum power
            Vpmax: voltage at maximum power
            Rs: series resistance
            Rsh: shunt resistance

        Additional values for LFM:
            Vr: voltage at intersection of Rsh and Rs tangent line
            Ir: current at intersection of Rsh and Rs tangent line

        Additional values for SAPM:
            Ix: current at 1/2 Voc
            Ixx: current at 1/2 (Voc + Vpmax)

        Args:
            isc_lim (int, float): percent or number of data points to calculate Isc and Rsh with
            voc_lim (int, float): percent or number of data points to calculate Voc and Rs with
            mode (str): Return extra parameters for specific performance models (lfm, sapm).

        Returns:
            dict with key and value pairs are described above.
        """
        v, i = self.smooth()

        # maximum power point
        pmax = np.max((v * i))
        mpp_idx = np.argmax((v * i))
        vpmax = v[mpp_idx]
        ipmax = i[mpp_idx]

        # isc and rsh
        if type(isc_lim) == float:
            isc_size = int(len(i) * isc_lim)
        else:
            isc_size = isc_lim
        isc_lm = linear_model.LinearRegression().fit(v[:isc_size].reshape(-1, 1), i[:isc_size].reshape(-1, 1))
        isc = isc_lm.predict(np.asarray([0]).reshape(-1, 1))[0][0]
        rsh = isc_lm.coef_[0][0] * -1

        # voc and rs
        if type(voc_lim) == float:
            voc_size = int(len(v) * voc_lim)
        else:
            voc_size = voc_lim
        voc_lm = linear_model.LinearRegression().fit(i[::-1][:voc_size].reshape(-1, 1),
                                                     v[::-1][:voc_size].reshape(-1, 1))
        voc = voc_lm.predict(np.asarray([0]).reshape(-1, 1))[0][0]
        rs = voc_lm.coef_[0][0] * -1

        # fill factor
        ff = (ipmax * vpmax) / (isc * voc) * 100

        params = {'pmax': pmax,
                  'vpmax': vpmax,
                  'ipmax': ipmax,
                  'voc': voc,
                  'isc': isc,
                  'rs': rs,
                  'rsh': rsh,
                  'ff': ff,
                  'iratio': ipmax / isc,
                  'vratio': vpmax / voc}

        # LFM
        if mode.lower() in ('all', 'lfm'):
            isc_lm_m, isc_lm_b = isc_lm.coef_[0], isc_lm.intercept_[0]
            voc_lm_m, voc_lm_b = voc_lm.coef_[0], voc_lm.intercept_[0]

            vr = ((isc_lm_b * voc_lm_m) + voc_lm_b) / (1 - (isc_lm_m * voc_lm_m))
            vr = vr[0]
            ir = isc_lm.predict(np.asarray([vr]).reshape(-1, 1))[0][0]

            params['vr'] = vr
            params['ir'] = ir

        # SAPM
        if mode.lower() in ('all', 'sapm'):
            ix = self.smooth(np.asarray([.5 * voc]))[1]
            ixx = self.smooth(np.asarray([.5 * (voc + vpmax)]))[1]
            params['ix'] = ix[0]
            params['ixx'] = ixx[0]

        # if with_models:
        #     params['isc_lm'] = isc_lm
        #     params['voc_lm'] = voc_lm

        return params

    def calc_lfm_params(self, mod, n_mod, irrad, temp):
        """Caluclate parameters for Loss Factor Model (LFM).  Measured parameters will be extracted from IV curve while
        reference values must be supplied.

        Args:
            mod (pvsystem): PVLIB pvsystem
            n_mod (int): number of modules
            irrad (float): poa irradiance (W/m2)
            temp (float): cell temperatur (C)

        Returns:
            dict of parameters for modeling PV system using the LFM (see above docs)
        """
        # if self.irrad_ is None or self.temp_ is None:
        risc = mod['I_sc_ref']
        ripmax = mod['I_mp_ref']
        rvoc = mod['V_oc_ref'] * n_mod
        rvpmax = mod['V_mp_ref'] * n_mod
        alpha = mod['alpha_sc']
        beta = mod['beta_oc']

        components = self.extract_components(mode='all')
        lfm_params = dict()
        lfm_params['nisct'] = components['isc'] / risc / irrad * (1 + alpha * (25 - temp))
        lfm_params['nrsc'] = components['ir'] / components['isc']
        lfm_params['nimp'] = components['ipmax'] * risc / components['ir'] / ripmax
        lfm_params['nroc'] = components['vr'] / components['voc']
        lfm_params['nvmp'] = components['vpmax'] * rvoc / components['vr'] / rvpmax
        lfm_params['nvoct'] = components['voc'] / rvoc * (1 + beta * (25 - temp))
        lfm_params['pimp'] = lfm_params['nisct'] * lfm_params['nrsc'] * lfm_params['nimp'] * ripmax * \
                             (irrad / (1 + alpha * (25 - temp)))
        lfm_params['pvmp'] = lfm_params['nvmp'] * lfm_params['nroc'] * lfm_params['nvoct'] * rvpmax / \
                             (1 + beta * (25 - temp))
        lfm_params['pmp'] = lfm_params['pimp'] * lfm_params['pvmp']
        return lfm_params

    # def normalize_curve(self, set_member=False, by='isc_voc', spline_kwargs={}):
    #     """Normalize IV curve by Isc/Voc or by MPP.  Data will be re-interpolated if set_member is True.

    #     Parameters
    #     ----------
    #     set_member: bool
    #         Reset member variables self.v_, self.i_ with normalized values.
    #     by: str
    #         Must be either 'isc_voc' or 'mpp'.
    #     spline_kwargs: dict
    #         Parameters for smoothing spline.

    #     Returns
    #     -------
    #     v: np.asarray
    #         Voltages
    #     i: np.asarray
    #         Currents
    #     """
    #     assert False, 'Method needs work/reimplementation.'
    #     components = self.extract_components()
    #     if by == 'isc_voc':
    #         i_scaler = components['isc']
    #         v_scaler = components['voc']
    #     elif by == 'mpp':
    #         i_scaler = components['ipmax']
    #         v_scaler = components['vpmax']
    #     else:
    #         raise ValueError("Argument 'by' must be either 'isc_voc' or 'mpp'.")

    #     v, i = self.smooth()

    #     v = v / v_scaler
    #     i = i / i_scaler

    #     if set_member:
    #         self.v_ = v
    #         self.i_ = i
    #         if len(spline_kwargs) > 0:
    #             self.interpolate(**spline_kwargs)
    #         else:
    #             self.interpolate()

    #     return v, i

    def interpolate(self, pct_change_allowed=0.1, spline_kwargs={'s': 0.025}):
        """Get interpolation function.

        Args:
            pct_change_allowed (float): smoothing parameter - only allow points with less than this amount of percent
                change between adjacent points
            spline_kwargs (dict): keywords for spline (see scipy.interpolate.splrep)

        Returns:
            self
        """
        pct_change = np.abs(np.diff(self.i_)) / self.i_[:-1]
        pct_change = np.insert(pct_change, 0, 0)
        v = self.v_[pct_change < pct_change_allowed]
        i = self.i_[pct_change < pct_change_allowed]
        self.interp_func_ = interpolate.splrep(v, i, **spline_kwargs)
        return self

    def smooth(self, v=None, raw_v=False, der=0, npts=250):
        """Return smoothed IV curve.  If self.interpolate() is not called first, default interpolation will be
        used (see interpolate method).

        Optionally pass specific voltage(s) (v=your_array), use the measured voltage (raw_v=True), or generate
        npts voltage values between endpoints of measured voltages.

        Args:
            v (np.array): specific voltage value(s) to calculate current for - skipped if None
            raw_v (bool): use measured voltage values (self.v_) or (if False) generate points between first
                and last elements of self.v_
            der (int): order of derivative
            npts (int): number of points to use in IV curve - ignored if raw_v is True

        Returns:
            (np.array, np.array) which is v_smooth, i_smooth (vectors of smoothed voltage and current values)
        """
        if self.interp_func_ is None:
            self.interpolate()

        if raw_v:
            v_smooth = self.v_
        elif v is not None:
            v_smooth = v
        else:
            v_smooth = np.linspace(self.v_[0], self.v_[-1], npts)
        i_smooth = interpolate.splev(v_smooth, self.interp_func_, der=der)

        return v_smooth, i_smooth

    # def locate_mismatch(self, min_peak=-.02, dv_neighbor_cutoff=0,
    #                     low_v_cutoff=0, vis=False, vis_with_deriv=False, verbose=False,
    #                     smooth_window=5, smooth_cutoff=0.005):  # smooth_cutoff=1e-4):
    #     """Find and return points of mismatch in IV curve.
    #
    #     Mismatches will be identified by finding peaks near zero in the dI/dV curve.
    #
    #
    #     This funciton is overly complex and should be revisited.
    #
    #     Args:
    #         min_peak (float): minimum value for a peak
    #         dv_neighbor_cutoff (int): this cutoff is a smoothing parameter for points that may be in the same
    #             mismatch region - if the index of a point is <= to it's neighbors, they are the 'same' mismatch
    #         low_v_cutoff (float): remove mismatching very close to 0V
    #         vis (bool): generate visualization
    #         vis_with_deriv (bool): include derivative in visualization
    #         verbose (bool): return extra information
    #         smooth_window (int): window size to check smoothness of peaks - window defined as:
    #             [i-smooth_window:i+smooth_window]
    #         smooth_cutoff (float): if smoothness of derivative near peaks is < smooth_cutoff, the peak is ingored
    #
    #     Returns:
    #         np.array where mismatching is deceted organized as follows:
    #             [[voltage, current],
    #              [...    , ...    ]]
    #         dict with following key/values (returned if verbose = True):
    #             v (np.array): evenly spaced voltages used to smooth curve
    #             i (np.array): smoothed current values from interpolation
    #             di_dv (np.array): smoothed first derivative from interpolation
    #             d2i_dv2 (np.array): smoothed second derivative from interpolation
    #     """
    #     # get smoothed curve to work with
    #     v, i = self.smooth()
    #     _, di_dv = self.smooth(der=1)
    #     _, d2i_dv2 = self.smooth(der=2)
    #
    #     # mismatches = np.isclose(di_dv, 0, atol=atol) & np.less(d2i_dv2, 0)
    #     mismatches = utils.get_local_extrema(di_dv, cutoff=min_peak)
    #
    #     # isolate regions of mismatch - consider peaks close together as same peak
    #     indices = mismatches
    #     vdiffs = np.insert(np.diff(v[indices]), 0, 0)
    #     subgroups = []
    #     group = []
    #     for ind, vdiff in zip(indices, vdiffs):
    #         if vdiff <= dv_neighbor_cutoff:
    #             group.append(ind)
    #         else:
    #             subgroups.append(group[:])
    #             group = [ind]
    #     subgroups.append(group)
    #
    #     # construct list of mismatch points
    #     mismatch_pts = []
    #     for sg in subgroups:
    #         if not sg:
    #             continue
    #         max_deriv_idx = np.argmax(di_dv[sg]) + sg[0]
    #         # window = di_dv[max_deriv_idx - smooth_window: max_deriv_idx + smooth_window]
    #         window = i[max_deriv_idx - smooth_window: max_deriv_idx + smooth_window]
    #         # if np.sqrt(np.mean(np.diff(window)**2)) < smooth_cutoff:
    #         # if np.mean(np.diff(window)**2) < smooth_cutoff:
    #         if (np.std(window) / np.mean(window)) < smooth_cutoff:
    #             continue
    #         if v[max_deriv_idx] > low_v_cutoff:
    #             mismatch_pts.append((v[max_deriv_idx], i[max_deriv_idx]))
    #     mismatch_pts = np.asarray(mismatch_pts)
    #
    #     if vis:
    #         self.mismatch_vis_(v, i, di_dv, d2i_dv2, mismatch_pts, indices, vis_with_deriv=vis_with_deriv)
    #
    #     if verbose:
    #         return mismatch_pts, {'v': v, 'i': i, 'di_dv': di_dv, 'd2i_dv2': d2i_dv2}
    #     else:
    #         return mismatch_pts

    def locate_mismatch(self, mph=-.02, mpd=1, threshold=0.00001, edge='rising',
                        kpsh=False, valley=False, show=False, ax=None):
        """Find and return points of mismatch in IV curve.

        Mismatches will be identified by finding peaks near zero in the dI/dV curve.

        This funciton is overly complex and should be revisited.

        Args:

        Returns:
            np.array with indices where mismatching exists.
        """
        # get smoothed curve to work with
        v, i = self.smooth()
        _, di_dv = self.smooth(der=1)
        _, d2i_dv2 = self.smooth(der=2)

        mismatches = detect_peaks.detect_peaks(di_dv, mph=mph, mpd=mpd, threshold=threshold,
                                               edge=edge, kpsh=kpsh, valley=valley)

        return mismatches

    # def normalized_mismatch(self):
    #     """Normalize mismatch points by Isc/Voc.

    #     Returns
    #     -------
    #     normed_mismatch: list of tuple
    #         Normalized value pairs.
    #     """
    #     raise NotImplementedError('Normalization will be done using LFM parameters.')
    #     mismatch = self.locate_mismatch()
    #     components = self.extract_components()
    #     v = np.asarray([a[0] for a in mismatch])
    #     i = np.asarray([a[1] for a in mismatch])
    #     v = v / components['voc']
    #     i = i / components['isc']
    #     normed_mismatch = [(vv, ii) for vv, ii in zip(v, i)]
    #     return normed_mismatch

    def mismatch_vis_(self, v, i, di_dv, d2i_dv2, mismatch_pts, indices, vis_with_deriv=False):
        """Visualize mismatch points and local maxima of dI/dV.  Most conveniently used
        via IVCurve.locate_mismatch(..., vis=True, ...).

        Args:
            v (np.array): voltages
            i (np.array): currents
            di_dv (np.array): first derivative of IV curve
            d2i_dv2 (np.array): second derivative of IV curve
            mismatch_pts (np.array): approximate mismatch locations
            indices (np.array): local maxima voltage indices
            vis_with_deriv (bool): plot derivative information

        Returns:
            None
        """
        if vis_with_deriv:
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        else:
            fig, axes = plt.subplots()

        if vis_with_deriv:
            ax = axes[0]
        else:
            ax = axes

        ax.plot(self.v_, self.i_, label='Raw data')
        ax.plot(v, i, label='Smoothed data')
        if len(mismatch_pts) > 0:
            ax.scatter(mismatch_pts[:, 0], mismatch_pts[:, 1], label='Approx. mismatch',
                       marker='o', facecolor='none', edgecolor='k', linewidth=2,
                       s=100, alpha=1, zorder=-1)
        # ax.scatter(mismatch_x_pts, mismatch_y_pts, label='approx mismatch', marker='+', c='k',
        #            linewidth=2, s=300,alpha=1, zorder=-1)
        _ = ax.legend(loc='lower left')
        _ = ax.set_title('IV profile (sample {})'.format(self.name_), fontsize=18)
        _ = ax.set_xlabel('Voltage / V', fontsize=15)
        _ = ax.set_ylabel('Current / A', fontsize=15)

        fig.savefig('{}.eps'.format(self.name_), format='eps', dpi=1000)

        if vis_with_deriv:
            ax = axes[1]
            ax.plot(v, di_dv, label=r'$\frac{ d\mathrm{I} }{ d\mathrm{V} }$')
            # ax.plot(v, d2i_dv2, label=r'$\frac{ d^2\mathrm{I} }{ d\mathrm{V^2} }$')
            ax.scatter(v[indices], di_dv[indices], label='local maxima', alpha=1, marker='o', facecolor='none',
                       edgecolor='k', linewidth=2, s=400, zorder=-1)
            _ = ax.legend(loc='lower left')
            _ = ax.set_title('Derivative IV profile (sample {})'.format(self.name_))
            _ = ax.set_xlabel('Voltage / V')
            _ = ax.set_ylabel('Current / A')

        _ = fig.tight_layout()

    def plot(self, ax=None):
        """Generate IV curve plot.  Will show raw data points and smoothed function.  MPP will be marked.

        Args:
            ax (matplotlib.axex.Axes): existing ax to plot on (new one created if None)

        Returns:
            None
        """
        v, i = self.smooth()
        components = self.extract_components(mode='all')

        if ax is None:
            fig, ax = plt.subplots()
            fig.tight_layout()

        ax.plot(self.v_, self.i_, label='raw data')
        ax.plot(v, i, label='smoothed data')
        ax.scatter(components['vpmax'], components['ipmax'], label='MPP', c='k', zorder=100)
        ax.plot([0, components['vpmax']], [components['ipmax'], components['ipmax']], linestyle='--', c='k', alpha=.5)
        ax.plot([components['vpmax'], components['vpmax']], [0, components['ipmax']], linestyle='--', c='k', alpha=.5)

        ax.legend()
        ax.set_title('IV profile (sample {})'.format(self.name_))
        ax.set_xlabel('Voltage / V')
        ax.set_ylabel('Current / A')


if __name__ == '__main__':
    main()
