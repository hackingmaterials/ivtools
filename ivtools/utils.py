
import numpy as np
import pandas as pd
import scipy.linalg as la

import matplotlib.pyplot as plt


def main():
    pass


def read_iv_csv(fname, round_30min=False):
    """Read IV-curve file from Sandia.

    Data will also be cleaned to remove 'Unnamed' columns.  The current and voltage values will be
    compressed into their own cell ('current_series' and 'voltage_series').   These values are
    listed in the 'Unnamed' columns that are being removed.

    Args:
        fname (str): path to file

    Returns:
        pd.DataFrame of IV data with 'Unnamed' columns removed.
    """
    df = pd.read_csv(fname)
    ser_list = []
    for index, ser in df.iterrows():
        if np.isfinite(ser['points']):
            ser['current_array'] = ser.values[12: 12 + int(ser['points'])]
            ser['voltage_array'] = ser.values[12 + int(ser['points']): 12 + (2 * int(ser['points']))]
            ser_list.append(ser[['datetime', 'irrad', 'temp', 'string', 'isc', 'voc', 'pmax',
                                 'ipmax', 'vpmax', 'ff', 'points', 'current_array', 'voltage_array']])
#         except ValueError:  # 'points' is NaN, breaks array slicing, skip entry
#             continue
    df = pd.DataFrame(ser_list)
    try:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['time'] = df['datetime'].apply(lambda x: x.time())
        if round_30min:
            df['datetime_rounded'] = df['datetime'].dt.round('30min')
    except ValueError:
        pass
    return df

def read_iv_csv_updated(fname, round_30min=False):
    """Read IV-curve file from Sandia.

    Data will also be cleaned to remove 'Unnamed' columns.  The current and voltage values will be
    compressed into their own cell ('current_series' and 'voltage_series').   These values are
    listed in the 'Unnamed' columns that are being removed.

    Args:
        fname (str): path to file

    Returns:
        pd.DataFrame of IV data with 'Unnamed' columns removed.
    """
    df = pd.read_csv(fname)
    ser_list = []
    df['datetime'] = df['timestamp']
    for index, ser in df.iterrows():
        if np.isfinite(ser['points']):
            # print(ser['I'])
            # print(type(ser['I']))
            try:
                curr = [float(i) for i in ser['I'].strip().split(',')]
                volt = [float(i) for i in ser['V'].strip().split(',')]
            except AttributeError:
                curr = np.zeros((1,))
                volt = np.zeros((1,))
            # print(curr)
            # print(type(curr))
            ser['current_array'] = np.asarray(curr)
            ser['voltage_array'] = np.asarray(volt)
            # ser['current_array'] = ser.values[12: 12 + int(ser['points'])]
            # ser['voltage_array'] = ser.values[12 + int(ser['points']): 12 + (2 * int(ser['points']))]
            ser_list.append(ser[['datetime', 'irrad', 'temp', 'string', 'isc', 'voc', 'pmax',
                                 'ipmax', 'vpmax', 'ff', 'points', 'current_array', 'voltage_array']])
#         except ValueError:  # 'points' is NaN, breaks array slicing, skip entry
#             continue
    df = pd.DataFrame(ser_list)
    try:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['time'] = df['datetime'].apply(lambda x: x.time())
        if round_30min:
            df['datetime_rounded'] = df['datetime'].dt.round('30min')
    except ValueError:
        pass
    return df


def plot_iv_info(ser, ax=None, ax2=None, savefig=None):
    if ax is None:
        fig, ax = plt.subplots()

    iv = ax.plot(ser['voltage_array'], ser['current_array'], label='IV')
    mpp = ax.scatter(ser['vpmax'], ser['ipmax'], label='MPP(IV)', zorder=100)

    ff_horizontal = ax.plot([0, ser['vpmax']], [ser['ipmax'], ser['ipmax']], linestyle='--', c='k', alpha=.5)
    ff_vertical = ax.plot([ser['vpmax'], ser['vpmax']], [0, ser['ipmax']], linestyle='--', c='k', alpha=.5)

    ax.set_xlabel('Voltage / V')
    ax.set_ylabel('Current / A')
    ax.set_title(ser['datetime'])

#     i_ratio = ax.plot([0, ser['vpmax']], [ser['isc'], ser['ipmax']], linestyle='--', label='I$_\mathrm{ratio}$')
#     v_ratio = ax.plot([ser['vpmax'], ser['voc']], [ser['ipmax'], 0], linestyle='--', label='V$_\mathrm{ratio}$')

    if ax2 is None:
        ax2 = ax.twinx()

    p_vs_v = ax2.plot(ser['voltage_array'], ser['voltage_array'] * ser['current_array'], c='C1', label='PV')
    mpp_p_vs_v = ax2.scatter(ser['vpmax'], ser['pmax'], c='C1', label='MPP(PV)')

    ax2.set_ylabel('Power / W')

    lines = iv + p_vs_v # + i_ratio + v_ratio
    labels = [l.get_label() for l in lines]
    legend = ax.legend(lines, labels, loc=('lower center'))

    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    return ax


def get_local_extrema(array, cutoff=None, mode='maxima'):
    """Simple peak finding algorithm.  Peaks are considered to be
    places where a point is greater than both neighbors.

    Args:
        array (np.array): 1D array of values
        cutoff (float): minimum peak or maximum valley cutoff (ie peak >= cutoff or valley <= cutoff).
        mode (str): 'maxima' finds peaks while 'minima' finds valleys.

    Returns
        1D array of indices where extrema occur.
    """
    window = 3
    rolling_windows = la.hankel(np.arange(0, len(array) - window + 1),
                                np.arange(len(array) - window, len(array)))
    if mode == 'maxima':
        if cutoff is None:
            cutoff = -np.inf
        local_extrema = np.argmax(array[rolling_windows], axis=1) == 1
        indices = rolling_windows[local_extrema][:, rolling_windows.shape[1] // 2]
        cutoff_indices = np.argwhere(array >= cutoff)
        indices = np.intersect1d(indices, cutoff_indices)
    elif mode == 'minima':
        if cutoff is None:
            cutoff = np.inf
        local_extrema = np.argmin(array[rolling_windows], axis=1) == 1
        indices = rolling_windows[local_extrema][:, 1]
        cutoff_indices = np.argwhere(array <= cutoff)
        indices = np.intersect1d(indices, cutoff_indices)
    else:
        raise ValueError('Mode must be either maxima or minima.')

    return indices


if __name__ == '__main__':
    main()
