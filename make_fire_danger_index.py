#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
"""
Calculate the Forest Fire Danger Index or Grass Fire Danger Index.

    FFDI is defined as:

    ffdi = 1.25 * df * exp [ (t - rh)/30.0 + 0.0234 * ws]

    where:
    - t = Temperature (ºC)
    - rh = relative humidity (%)
    - ws = wind speed (km hr-1)
    - df = drought factor

    GFDI is defined as:

    gfdi = np.minimum(f * pow(10, i1 + i2 + i3 + i4), 500)

    where:
    - f = fuel load factor: pow(fuel_load, 1.027) / 4.587 fuel_load default to 4.5 t/ha
    - i1 = curing index: -0.004096 * pow(100 - curing, 1.536) curing default to 100%
    - i2 = temp index: 0,01201 * T
    - i3 = wind (km/h) index: 0.2789 * np.sqrt(WindSpd_kmph), where WindSpd_kmph = UnitConverter.kn2kmh(WindSpd)
    - i4 = RH index: -0.09577 * np.sqrt(RH)


Simple usage:

$ source config/<env>.conf
(this contains a bunch of python environment stuff - you could
set all this up yourself if you prefer)

$ python modules/calc_fire_danger_index.py -it <input_t_path> -irh <input_rh_path> -iws <input_ws_path> -idf <input_df_path> -vt <t_var> -vrh <rh_var> -vws <ws_var> -vdf <df_var> -o <output_path>

e.g.

h python modules/calc_fire_danger_index.py.py -it daq5_tmean_20180729_emn.nc -irh daq5_rh_09_20180729_emn.nc -iws daq5_wind_speed_09_20180729_emn.nc -idf daq5_df_09_20180729_emn.nc -vt tmean -vrh rh -vws wind_speed -vdf df -o daq5_ffdi_20180729_emn.nc

To see the other optional args, use:

$ python modules/calc_fire_danger_index.py.py -h

(c) 2016 Commonwealth of Australia
    Australian Bureau of Meteorology, R&D
    All Rights Reserved
"""

__author__ = 'Catherine de Burgh-Day and Naomi Benger (GFDI addition)'

import logging
import numpy as np
import argparse

try:
    import modules.product_processing_utilities_common as cmn
except ImportError:
    import product_processing_utilities_common as cmn

DEFAULT_AXIS = 0
DEFAULT_MAPAXIS = 0
DEFAULT_WSC_VARIABLE_NAME = None
DEFAULT_WSC_INPUT_DATA = None

DEFAULT_LOG_FILE_NAME = None
DEFAULT_STREAM_LOGGER_LEVEL = 'INFO'
DEFAULT_LOGFILE_LOGGER_LEVEL = 'DEBUG'

DEFAULT_METHOD_VARS = 'keep'
DEFAULT_METHOD_DIMS = 'keep'
DEFAULT_METHOD_GATTRS = 'omit'
DEFAULT_SRC_VARS = ()
DEFAULT_SRC_DIMS = ()
DEFAULT_SRC_GATTRS = ()
DEFAULT_ATTR_SOURCE_FILE = None
DEFAULT_NEW_ATTRS = {}

logger = logging.getLogger(__name__)


def fetch_data(
        input_data,
        variable_name,
    ):
    '''
    Function fetch the data attributes and dimensions, and massage
    the data into the desired shape.

    Args:
        input_data (str): The data file.
        variable_name (str): The name of the variable.

    Returns:
        data_dict (dict): A dictionary containing the data, the
            attributes associated with the data (as a dictionary),
            the names of dimensions associated with the data (as
            a tuple of stings), the name of the varible the data
            was read from, and the datatype of the data.
    '''
    
        logger = logging.getLogger(__name__)

    data_dict, _, _ = cmn._get_data(
        input_data,
        variable_name,
        None,
        None,
    )

    if data_dict['data'] is None:
        bad_varname_err_msg = (
            "Error: Data dict is None! Maybe you "
            "passed in the wrong variable name(s)? "
            "Variable name that returned None is {}."
        ).format(variable_name)
        logger.error(bad_varname_err_msg)
        raise ValueError(bad_varname_err_msg)

    return data_dict

#def fdi(t_data, rh_data, ws_data, df_data, curing=100, mode='gfdi'):
def fdi(t_data, rh_data, ws_data, df_data, curing, mode):
    '''
    Calculate the Fire Danger Index.
    Either the Forest Fire Danger Index (FFDI) (mode='ffdi') or
    Grass Fire Danger Index (GFDI) (mode='gfdi').

    FFDI is defined as:

    ffdi = 1.25 * df * exp [ (t - rh)/30.0 + 0.0234 * ws]

    where:
    - t = Temperature (ºC)
    - rh = relative humidity (%)
    - ws = wind speed (km hr-1)
    - df = drought factor

    GFDI is defined as:

    reference: https://www.bnhcrc.com.au/sites/default/files/managed/downloads/fire_danger_indices_report_v1.1.pdf

    gfdi = np.minimum(f * pow(10, i1 + i2 + i3 + i4), 500)

    where:
    - f = fuel load factor: pow(fuel_load, 1.027) / 4.587 fuel_load default to 4.5 t/ha --> f defaults to 1.039828449636768
    - i1 = curing index: -0.004096 * pow(100 - curing, 1.536) curing default to 100%
    - i2 = temp index: 0.01201 * T
    - i3 = wind (km/h) index: 0.2789 * np.sqrt(WindSpd_kmph), where WindSpd_kmph = UnitConverter.kn2kmh(WindSpd)
    - i4 = RH index: -0.09577 * np.sqrt(RH)

    The input `t_data`, `rh_data`, `ws_data` and `df_data`
    arrays must be the same shape.

    Args:
        t_data (numpy.ndarray): The temperature data.
            Must be the same shape as the other input
            data arrays.
        rh_data (numpy.ndarray): The relative humidity
            data. Must be the same shape as the other
            input data arrays.
        ws_data (numpy.ndarray): The wind speed data.
            Must be the same shape as the other input
            data arrays.
        df_data (numpy.ndarray): The drought factor data.
            Must be the same shape as the other input
            data arrays.
        mode (Optional[str]): Either ffdi or gfdi.
            Whether to compute the forest fire danger
            index or the grass fire dnager index.

    Returns:
        fdi_data (np.ndarray): The (forest or grass) fire
            danger index ([F/G]FDI) array. Will be the same
            shape as the input data arrays.
    '''
    logger = logging.getLogger(__name__)

    ok_modes = ('ffdi', 'gfdi')
    shapes = [arr.shape for arr in [t_data, rh_data, ws_data, df_data]]
    bad_shapes_err = (
        "Error: the shapes of the input data do not match! "
        "Shapes are: t_data: {};  rh_data: {};  ws_data: {};  "
        "df_data: {}."
    ).format(*shapes)
    bad_mode_err = (
        "Error: mode must be one of {} but it is {}"
    ).format(ok_modes, mode)

    lmode = mode.lower()

    if (len(set(shapes)) > 1):
        logger.error(bad_shapes_err)
        raise AttributeError(bad_shapes_err)

    if (lmode not in ok_modes):
        logger.error(bad_mode_err)
        raise AttributeError(bad_mode_err)

    # Make a new mask for fdi by combining the input data masks
    t_mask = np.ma.getmaskarray(t_data)
    rh_mask = np.ma.getmaskarray(rh_data)
    ws_mask = np.ma.getmaskarray(ws_data)
    df_mask = np.ma.getmaskarray(df_data)
    fdi_mask = np.ma.mask_or(
        np.ma.mask_or(t_mask, rh_mask),
        np.ma.mask_or(ws_mask, df_mask),
    )

    
    if (lmode == 'ffdi'):
        # FFDI consisent with AWAP
        fdi_data =  np.exp(0.0234 * ws_data - 0.0345 * rh_data + 0.0338 * t_data + 0.243147) * (df_data**0.987)
    elif (lmode == 'gfdi'):
        #Initialising GFDI components
        #defaulting fuel load to 4.5 t/ha
        f = 1.02169322
        #defaulting curing to 100% as set in function input
        i1 = -0.004096 * pow(100 - curing, 1.536)
        i2 = 0.01201 * t_data
        i3 = 0.2789 * np.sqrt(ws_data)
        i4 = -0.09577 * np.sqrt(rh_data)
        fdi_data = np.minimum(f * pow(10, i1 + i2 + i3 + i4), 500)
    else:
        logger.error(bad_mode_err)
        raise AttributeError(bad_mode_err)

    # Apply the fdi mask
    fdi_data = np.ma.array(fdi_data, mask=fdi_mask)

    return fdi_data


def main(args):

    cmn._set_logging(
        stream_logger_level=args.stream_logger_level,
        logfile_logger_level=args.logfile_logger_level,
        log_file_name=args.log_file_name,
    )
    logger = logging.getLogger(__name__)

    ok_modes = ('ffdi', 'gfdi')
    if (args.mode.lower() not in ok_modes):
        bad_mode_err = (
            "Error: mode must be one of {} but it is {}"
        ).format(ok_modes, args.mode)
        logger.error(bad_mode_err)
        raise AttributeError(bad_mode_err)

    # Read in t, rh, ws and df data dicts
    t_data_dicts = [fetch_data(
            tdata, variable_name=vt,
    ) for vt, tdata in zip(args.t_variable_name, args.input_t_data)]
    rh_data_dicts = [fetch_data(
            rhdata, variable_name=vrh,
    ) for vrh, rhdata in zip(args.rh_variable_name, args.input_rh_data)]
    ws_data_dicts = [fetch_data(
            wsdata, variable_name=ws,
    ) for ws, wsdata in zip(args.ws_variable_name, args.input_ws_data)]
    df_data_dicts = [fetch_data(
            dfdata, variable_name=df,
    ) for df, dfdata in zip(args.df_variable_name, args.input_df_data)]

    # Read in the curing value
#    curing = float(args.input_curing_data)
    curing = 100.

    # Grab one of each dict for extracting metadata etc.
    one_dicts_list = [
        t_data_dicts[0],
        rh_data_dicts[0],
        ws_data_dicts[0],
        df_data_dicts[0],
    ]

    # Convert units to those expected for FDI if needed
    t_units = 'degC'
    t_data_dicts = [cmn._check_and_convert_units(
        t_dict, t_units, args.axis
    ) for t_dict in t_data_dicts]
    ws_units = 'km hr-1'
    ws_data_dicts = [cmn._check_and_convert_units(
        ws_dict, ws_units, args.axis
    ) for ws_dict in ws_data_dicts]

    # Take averages of t, rh, ws and df data read in
    t_data = np.ma.array(
        [t_dd['data'] for t_dd in t_data_dicts]
    )
    mean_t_data = np.ma.average(t_data, axis=0)
    rh_data = np.ma.array(
        [rh_dd['data'] for rh_dd in rh_data_dicts]
    )
    mean_rh_data = np.ma.average(rh_data, axis=0)
    ws_data = np.ma.array(
        [ws_dd['data'] for ws_dd in ws_data_dicts]
    )
    mean_ws_data = np.ma.average(ws_data, axis=0)
    df_data = np.ma.array(
        [df_dd['data'] for df_dd in df_data_dicts]
    )
    mean_df_data = np.ma.average(df_data, axis=0)

    # Compute (f/g)fdi from t, rh, ws and df....
    fdi_data = fdi(
        mean_t_data, mean_rh_data, mean_ws_data, mean_df_data,
        curing, mode=args.mode
    )

    # Try to find a fillvalue and missing value to copy...
    newvalinfo = cmn._find_fillvalue_and_missingvalue(
        one_dicts_list, newfillval=1.0e20, fillvalname='_FillValue',
        newmissval=1.0e20, missvalname='missing_value'
    )
    newfillval, fillvalname, newmissval, missvalname = newvalinfo

    # Set up stuff for the different modes
    mode_info = {
        'ffdi': {
            'name': 'ffdi',
            'long_name': 'forest fire danger index',
        },
        'gfdi': {
            'name': 'gfdi',
            'long_name': 'grass fire danger index',
        },
    }

    # Set up attrs dict
    fdi_attrs = {
        'long_name': mode_info[args.mode.lower()]['long_name'],
        'units': "none",
        fillvalname: newfillval,
        'missing_value': newmissval,
    }

    # Set up output dict including attrs dict
    fdi_data_dict = {
        'data': fdi_data,
        'datatype': t_data_dicts[0]['datatype'],
        'dims':  t_data_dicts[0]['dims'],
        'attrs': fdi_attrs,
        'name': mode_info[args.mode.lower()]['name'],
        'rec_dim': t_data_dicts[0]['rec_dim'],
    }

    # Save fdi to netcdf
    cmn._save_data(
        args.output_data,
        [fdi_data_dict],
        attr_source_file=args.attr_source_file,
        method_vars=args.method_vars,
        method_dims=args.method_dims,
        method_gattrs=args.method_gattrs,
        src_vars=tuple(args.src_vars),
        src_dims=tuple(args.src_dims),
        src_gattrs=tuple(args.src_gattrs),
        new_attrs=args.new_attr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = (
            'Compute a (Forest/Grass) Fire Danger Index ([F/G]FDI) '
            'from temperature, relative humidity, wind speed and '
            'drought factor.'
        )
    )
    parser.add_argument(
        '-it', '--input-t-data', dest='input_t_data', nargs='*',
        type=str, help=(
            'the input temperature data files. They must all '
            'have the same dimensions. The number of input files '
            'must match the number of associated variable names. The '
            '(unweighted) mean will be taken of these to get the '
            'final T used to compute the FDI. This allows for '
            'using a mean temperature across several daily values for '
            'example.'
        ),
    )
    parser.add_argument(
        '-irh', '--input-rh-data', dest='input_rh_data', nargs='*',
        type=str, help=(
            'the input relative humidity data files. They must all '
            'have the same dimensions. The number of input files '
            'must match the number of associated variable names. The '
            '(unweighted) mean will be taken of these to get the '
            'final RH used to compute the FDI. This allows for '
            'using a mean RH across several daily values for '
            'example.'
        ),
    )
    parser.add_argument(
        '-iws', '--input-ws-data', dest='input_ws_data', nargs='*',
        type=str, help=(
            'the input wind speed data files. They must all '
            'have the same dimensions. The number of input files '
            'must match the number of associated variable names. The '
            '(unweighted) mean will be taken of these to get the '
            'final wind speed used to compute the FDI. This allows for '
            'using a mean wind speed across several daily values for '
            'example.'
        ),
    )
    parser.add_argument(
        '-idf', '--input-df-data', dest='input_df_data', nargs='*',
        type=str, help=(
            'the input drought factor data files. They must all '
            'have the same dimensions. The number of input files '
            'must match the number of associated variable names. The '
            '(unweighted) mean will be taken of these to get the '
            'final DF used to compute the FDI. This allows for '
            'using a mean DF across several daily values for '
            'example.'
        ),
    )

    parser.add_argument(
        '-o', '--output-data', dest='output_data', type=str,
        help='the output rh data file'
    )

    parser.add_argument(
        '-o_ws', '--output-wind_speed', dest='output_wind_speed', type=str,
        help='the wind speed output file'
    )

    parser.add_argument(
        '-vt', '--t-variable-name', dest='t_variable_name',
        type=str, nargs='*',
        help=(
            'the name of the temperature data variables in '
            'the input files. The number of variable names '
            'must match the number of associated input files. '
        ),
    )
    parser.add_argument(
        '-vrh', '--rh-variable-name', dest='rh_variable_name',
        type=str, nargs='*',
        help=(
            'the name of the relative humidity data variables in '
            'the input files. The number of variable names '
            'must match the number of associated input files. '
        ),
    )
    parser.add_argument(
        '-vws', '--ws-variable-name', dest='ws_variable_name',
        type=str, nargs='*',
        help=(
            'the name of the wind speed data variables in '
            'the input files. The number of variable names '
            'must match the number of associated input files. '
        ),
    )
    parser.add_argument(
        '-vdf', '--df-variable-name', dest='df_variable_name',
        type=str, nargs='*',
        help=(
            'the name of the drought factor data variables in '
            'the input files. The number of variable names '
            'must match the number of associated input files. '
        ),
    )
   parser.add_argument(
        '-iwsc', '--wsc-input-data', dest='wsc_input_data',
        type=str, help=(
            'an optional wind speed calibration file, which if '
            'provided will be used to recalibrate the wind speed '
            'data prior to computing the FDI. Default is None (no '
            'additional calibration). If this is set, '
            '`wsc_variable_name` must also be set.'
        ),
        default=DEFAULT_WSC_INPUT_DATA,
    )
    parser.add_argument(
        '-vwsc', '--wsc-variable-name', dest='wsc_variable_name',
        type=str, help=(
            'the variable name of the mapping data in an optional '
            'wind speed calibration file, which if provided will be '
            'used to recalibrate the wind speed data prior to '
            'computing the FDI. Default is None. If a calibration '
            'file is provided this option must also be set.'
        ),
        default=DEFAULT_WSC_VARIABLE_NAME,
    )
    parser.add_argument(
        '-m', '--mode', dest='mode', type=str,
        help=(
            "the mode to run in - whether to compute the forest "
            "fire danger index or the grass fire danger index. "
        ),
    )
    parser.add_argument(
        '-x', '--axis', dest='axis', type=str,
        help='the axis of time in the data',
        default=DEFAULT_AXIS
    )
    parser.add_argument(
        '-xm', '--mapaxis', dest='mapaxis', type=str,
        help=(
            'the axis of the mapping dimension in the recalibration mapping '
            'array, if one is provided. If no mapping array is provided this '
            'is ignored. Default is 0.'
        ),
        default=DEFAULT_MAPAXIS
    )
    parser.add_argument(
        '-AF', '--attr-source-file', dest='attr_source_file', type=str,
        help=(
            "the source file to get old variables, dims "
            "and attributes out of"
        ),
        default=DEFAULT_ATTR_SOURCE_FILE,
    )
    parser.add_argument(
        '-MV', '--method-vars', dest='method_vars', type=str,
        help=(
            "the method for grabbing vars from the input netcdf file "
            "(either `'keep'` (default), or `'omit'`)."
            "For more details on the way data (vars, dims and global attributes)"
            "are copied (or not) from a source netCDF file,"
            "look at the doc string in _create_nc in"
        ),
        default=DEFAULT_METHOD_VARS,
    )
    parser.add_argument(
        '-MD', '--method-dims', dest='method_dims', type=str,
        help=(
            "the method for grabbing dims from the input netcdf file "
            "(either `'keep'` (default), or `'omit'`)."
            "For more details on the way data (vars, dims and global attributes)"
            "are copied (or not) from a source netCDF file,"
            "look at the doc string in _create_nc in"
        ),
        default=DEFAULT_METHOD_DIMS,
    )
    parser.add_argument(
        '-MG', '--method-gattrs', dest='method_gattrs', type=str,
        help=(
            "the method for grabbing global attributes from the input netcdf file "
            "(either `'keep'`, or `'omit' (default)`)."
            "For more details on the way data (vars, dims and global attributes)"
            "are copied (or not) from a source netCDF file,"
            "look at the doc string in _create_nc in"
        ),
        default=DEFAULT_METHOD_GATTRS,
    )
    parser.add_argument(
        '-SV', '--src_vars', dest='src_vars', type=str, nargs='*',
        help=(
            "the variables in the the input file to apply `'method'` "
            "to (default is `()` - i.e. none of the vars). If the "
            "vars share dimensions with the modified varible, then "
            "the dimension size must be unchanged or the modified "
            "varible must use a new dimension"
            "For more details on the way data (vars, dims and global attributes)"
            "are copied (or not) from a source netCDF file,"
            "look at the doc string in _create_nc in"
        ),
        default=DEFAULT_SRC_VARS,
    )
    parser.add_argument(
        '-SD', '--src_dims', dest='src_dims', type=str, nargs='*',
        help=(
            "the dimensions in the source input file to apply `'method_dims'` "
            "to (default is `()` - i.e. none of the dims)."
            "product_processing_utilities_common.py"
            "For more details on the way data (vars, dims and global attributes)"
            "are copied (or not) from a source netCDF file,"
            "look at the doc string in _create_nc in"
        ),
        default=DEFAULT_SRC_DIMS,
    )
    parser.add_argument(
        '-SG', '--src_gattrs', dest='src_gattrs', type=str, nargs='*',
        help=(
            "the global attributes in the source input file to apply `'method_gattrs'` "
            "to (default is `()` - i.e. keep all the global attributes)."
            "For more details on the way data (vars, dims and global attributes)"
            "are copied (or not) from a source netCDF file,"
            "look at the doc string in _create_nc in"
        ),
        default=DEFAULT_SRC_GATTRS,
    )
    parser.add_argument(
        '-A', '--new-attr', dest='new_attr', nargs='*',
        action = type(
            '', (argparse.Action, ), dict(__call__ = cmn._parse_attrs_dict_list)
        ),
        help=(
            "the attribute name, the attribute value, and the method for "
            "adding the attibute to the file if it already exists (one of "
            "'skip' to not add it, 'append' to add it to the existing "
            "attribute, and 'overwrite' to replace the existing attribute) "
            "separated by commas. Can call this arg multiple times to add in "
            "multiple new attributes. "
        ),
        default=DEFAULT_NEW_ATTRS,
    )
    parser.add_argument(
        '-lo', '--logfile',
        dest='log_file_name', type=str,
        help=(
            "the file to write the python logs out to."
            "If unset, no logfile is produced"
        ),
        default=DEFAULT_LOG_FILE_NAME
    )
    parser.add_argument(
        '-slog', '--stream-logger-level',
        dest='stream_logger_level', type=str,
        help=(
            "the logger level for the stdout and stderr"
        ),
        default=DEFAULT_STREAM_LOGGER_LEVEL
    )
    parser.add_argument(
        '-flog', '--logfile-logger-level',
        dest='logfile_logger_level', type=str,
        help=(
            "the logger level for the logfle. If no logfile "
            "is being produced this is ignored."
        ),
        default=DEFAULT_LOGFILE_LOGGER_LEVEL
    )
    args = parser.parse_args()

    main(args)

      
      


