# -*- coding: utf-8 -*-
"""Module for generating ScopeSim source object(s) for UVEX from user input
   Taking into account telescope orientation, background, etc."""
import astropy
import numpy as np
import astropy.table
import synphot.units
from astropy.coordinates import SkyCoord
from synphot import SourceSpectrum, BlackBodyNorm1D
from astropy.io import fits
import warnings

class UVEXInput:
    '''
    There are currently three main types of data stored in the UVEXInput class.

    First, there are point sources with an associated spectra.
    These are given in the table point_sources_from_spectra.
    For each source we have a position (ra, dec) and a reference to a spectrum.
    This reference is a string which matches a key in the spectra dictionary.
    Each point source also comes with a scale factor which multiplies its referenced spectrum

    Second, there are point sources with constant spectra.
    This table is a convenience as any spectra could be added to the dictionary of spectra
    and referenced by a point source in point_sources_from_spectra.
    These are given in point_sources_from_magnitude.
    Their position (ra, dec) as well as their magnitude (mag) are included in this table

    Third, there are FITS files containing images.
    These can be found in the patches table, which also contains position information (ra, dec) for each FITS image.
    The table contains paths (path column) and also spectral references (ref) which are strings that match spectra dictionary keys.
    We assume that the data is in the primary HDU

    Object attributes after validation functions:

        1. observation_coordinates: This is a SkyCoord object for the observation coordinates
        2. observation_time: An astropy.time.Time object for the observation time
        3. use_background: bool for if we should generate a background
        4. point_sources_from_spectrum: Astropy QTable with headers: "ra", "dec", "ref", "scale". "ra" and "dec" have units.
        "ref" is a scalar value which matches a key in the spectra dictionary. Scale is a unitless quantity which scales the referenced spectrum.
        scale exists so that multiple point sources can reference the same spectrum up to differing multiplicative factors.
        5. point_sources_from_magnitude: Astropy QTable with headers: "ra", "dec", "mag". "ra" and "dec" and "mag" all have units.
        These point sources should be converted to SourceSpectrum objects using the synphot ConstFlux1D model as opposed to the astropy Const1D
        6. patches: Astropy QTable with headers: "ra", "dec", "ref", "path", where "ra" and "dec" have units. Each value of "ref" matches a key in
        spectra. "path" values are paths to valid FITS files. "ra" and "def" give the location of the center of the FITS image.
        All FITS headers are ignored (except the basic ones) and we take the position information from the table "ra" and "dec" values
        7. spectra: A dictionary with string keys that match "ref" values in astropy tables. The values are synphot SourceSpectrum objects.

        TODO: I can add an input for extinction when it is decided how the user will specify it
    '''
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self,
                 observation_coordinates: SkyCoord,
                 observation_time: astropy.time.Time,
                 use_background: bool,
                 point_sources_from_spectra: astropy.table.QTable = None,
                 point_sources_from_magnitude: astropy.table.QTable = None,
                 patches: astropy.table.QTable = None,
                 spectra: astropy.table.QTable = None):

        """
        :param observation_coordinates: The SkyCoord object describing where the observation takes place.
        :param observation_time: An astropy.time.Time object describing when the observation takes place.
        :param use_background: A boolean value indicating whether to generate a background from the observation coordinates and time.
        :param point_sources_from_spectra: An astropy QTable of point sources with spectra given in the spectra parameter.
        The columns must be "ra", "dec", "ref" (string reference to the spectrum in spectra), and scale which is a scalar value multiplying the referenced spectrum.
        :param point_sources_from_magnitude: An astropy QTable of point sources generated with the SourceObject Constant spectrum.
        Required columns are "ra", "dec", "magnitude"
        :param spectra: An astropy QTable of spectra with columns "ref" and "spectra".
        Elements in the "ref" column are strings matching those in the point_sources_from_spectrum table.
        Elements in the spectra column are synphot SourceSpectrum objects.
        :param patches: An astropy QTable for fits file inputs. The columns must be "ra", "dec", "ref", "path".
        "ra", "dec" give the center of the image, "ref" is a string matching the "ref" column in the spectra parameter.


        """

        if not isinstance(use_background, bool):
            raise TypeError("use_background must be a boolean value.")

        #Ensure that the astropy QTables inputted are in the expected format
        point_refs = self.table_validation(point_sources_from_spectra, #table
                                           ["ra", "dec", "ref", "scale"], #expected headers
                                           [float, float, str, float], #expected column data types
                                           [0,1], #header indexes which correspond to columns that must have units
                                           " point_sources_from_spectrum table", #table name to show in error messages
                                           2) #header index of data column to return to compare

        patch_refs = self.table_validation(patches,
                                           ["ra", "dec", "ref", "path"],
                                           [float, float, str, str],
                                           [0,1],
                                           " patches table",
                                           2)

        self.table_validation(point_sources_from_magnitude,
                              ["ra", "dec", "mag"],
                              [float, float, float],
                              [0, 1, 2],
                              " point_sources_from_magnitude table",
                              None)

        self.validate_observation_coordinates(observation_coordinates, observation_time)
        self.validate_fits(patches.columns["path"].data.tolist())
        spectrum_refs = self.validate_spectra(spectra)
        # ensure that any reference to a spectra ocuring in the patches or point source table corresponds to a spectrum in spectra
        self.cross_check(point_refs, patch_refs, spectrum_refs)

        warnings.warn("FITS headers are not being used. "
                      "Position and spectral information for each FITS image will be taken from the spectra astropy table.")
        warnings.warn("Magnitudes given in units other than PHOTLAM may not give constant spectra once converted.")

        self.observation_coordinates = observation_coordinates
        self.observation_time = observation_time
        self.use_background = use_background
        self.point_sources_from_spectra = point_sources_from_spectra
        self.point_sources_from_magnitude = point_sources_from_magnitude
        self.spectra = spectra
        self.patches = patches

        print("Done initializing UVEXInput object \n"
              f"coordinates: {self.observation_coordinates} \n"
              f"time: {self.observation_time} \n"
              f"{len(self.point_sources_from_spectra[0])} point sources with specified spectra \n"
              f"{len(self.point_sources_from_magnitude[0])} point sources with constant spectra\n"
              f"spectra: {self.spectra.keys()}\n"
              f"patches: {self.patches.columns["path"].data} \n ")




    @staticmethod
    def table_validation(table: astropy.table.QTable,
                         headers: list[str],
                         types: list[type],
                         headers_with_units: list[int],
                         debug_name: str,
                         ref_index):
        """
        Checks:
        1. table is None or an Astropy QTable with no masking and units
        2. table headers match "headers"
        3. types of elements in columns are "types"
        4. headers specified by headers_with_units have an associated unit

        :param table: AstropyQTable
        :param headers: list of header strings such as "ra" or "ref" that the table should have
        :param types: list of types such as "str" or "float" that the table columns should be
        :param headers_with_units: indices of headers which need a unit in the table units attribute
        :param debug_name: name of the table as will show up in error messages
        :param ref_index: index of the ref column
        :return: list of values from ref_index column
        """

        if table is None:
            return []

        if not isinstance(table, astropy.table.QTable):
            raise TypeError(f"{debug_name} must be an astropy QTable")

        if table.masked == True:
            raise Exception(f"Masked Tables not supported, {debug_name} has attribute masked = True")

        if not table.colnames == headers:
            raise Exception(
                f"{debug_name} should have {headers} as column headers, it currently has {table.colnames} as headers")

        if len(headers_with_units) > 0:
            try:
                units = table.units
            except AttributeError:
                raise AttributeError(f"{debug_name}  QTable does not have unit attribute, {[headers[i] for i in headers_with_units]} should have units")
            if isinstance(units, list):
                units = dict(zip(table.colnames, units))

        for header_index in headers_with_units:
            try:
                unit = units[headers[header_index]]
            except KeyError:
                raise KeyError(f"{debug_name} QTable must have unit specified for {headers[header_index]} columns, currently the units are {units}")
            if not isinstance(unit,astropy.units.Unit):
                raise TypeError(f"{debug_name} QTable must have unit specified for {headers[header_index]} columns")


        for (header, type) in zip(headers, types):
            column = table.columns[header]
            for (i, element) in enumerate(column.data.tolist()):
                if not isinstance(element, type):
                    raise Exception(f"Element {i} in column {header} of {debug_name} should have type {type} instead of {type(element)}")
                if not element:
                    raise Exception(f"Invalid element {i} in column {header}")


        if ref_index:
            return table.columns[headers[ref_index]].data.tolist()
        else:
            return []

    @staticmethod
    def validate_fits(fits_paths: list[str]):
        """
        Checks:
        1. Each path opens as a fits file
        :param fits_paths: list of paths to FITS files
        """

        for path in fits_paths:
            try:
                with fits.open(path) as hdul:
                    hdul = fits.open(path)


                    hdul.verify()
            except:
                raise Exception(f"{path} is not a valid fits file")
        return

    @staticmethod
    def validate_observation_coordinates(observation_coordinates: SkyCoord, observation_time: astropy.time.Time):
        """
        Checks:
        1. observation_coordinates is an astropy SkyCoord type
        2. observation_time is an astropy Time object
        3. if the obstime of the observation_coordinates has been set then it matches observation_time
        4. observation_time has only one time value
        4. there is exactly one RA and one DEC value (since SkyCoord can be initialized with arrays of coords)
        """
        if not isinstance(observation_coordinates, astropy.coordinates.SkyCoord):
            raise TypeError("observation_coordinates must be a SkyCoord")
        if not isinstance(observation_time, astropy.time.Time):
            raise TypeError("observation_time must be an astropy.time.Time value")
        # check if observation coordinates has an obstime (which is optional for SkyCoord)
        if isinstance(observation_coordinates.obstime, astropy.time.Time):
            #if so this time must match observation time
            if not observation_coordinates.obstime == observation_time:
                raise Exception("If observation_coordinates is initialized with an obstime it must match observation_time")
        #ensure there is only one time
        if not np.array(observation_time.value).size == 1:
            raise Exception("Time must have exactly one time value")
        #check that there is exactly one value each for RA and DEC
        if not len(observation_coordinates.ra) == 1:
            raise Exception("observation_coordinates must have exactly one RA value")
        if not len(observation_coordinates.dec) == 1:
            raise Exception("observation_coordinates must have exactly one DEC value")
        return

    @staticmethod
    def validate_spectra(spectra: dict[str, SourceSpectrum]):
        """
        Checks:
        1. spectra is None or a dict[str, SourceSpectrum]
        2. References are not empty strings

        :param spectra: dictionary (key: string, value: SourceSpectrum object) containing all spectra referenced in
        input tables
        :return: list of ref values
        """

        if spectra is None:
            return []
        #check if spectra is a dictionary
        if not isinstance(spectra, dict):
            raise TypeError(
                "spectra must be None or a dictionary of SourceSpectrum objects with the keys equal to ref values in input tables")
        #check each key is a valid string and each value is a SourceSpectrum object
        for key in spectra.keys():
            if not isinstance(key, str):
                raise TypeError("keys of spectra dictionary must be strings")
            if not key:
                raise Exception(f"Key must not be an empty string")
            if not isinstance(spectra[key], SourceSpectrum):
                raise TypeError("values of spectra must be astropy SourceSpectrum objects")
        return spectra.keys()




    @staticmethod
    def cross_check(point_refs: list[str], patch_refs: list[str], spectra_refs: list[str]):
        """
        Checks:
        1. All spectra referenced in the point_sources_from_spectrum table exist as references in spectra.
        2. All spectra referenced in the patches table exist as references in spectra.
        3. Warn for unused spectra.
        :param point_refs: List of spectrum references in point_sources_from_spectrum.
        :param patch_refs: List of spectrum references in patches.
        :param spectrum_refs: List of spectrum keys in spectra
        :return:
        """
        # make sure that all the spectra referenced in the point source table are in the list of spectra
        if not set(point_refs).issubset(set(spectra_refs)):
            raise Exception(
                "Every spectrum reference in point_sources must have a corresponding key in the spectra dictionary \n"
                f"From point_sources table: {set(point_refs)} \n"
                f"From spectra: {set(spectra_refs)} \n")

        # make sure that all the spectra referenced in the FITS table are in the list of spectra
        if not set(patch_refs).issubset(set(spectra_refs)):
            raise Exception(
                "Every spectrum reference in a FITS file must have a corresponding key in the spectra dictionary \n"
                f"From FITS files: {set(patch_refs)} \n"
                f"From spectra: {set(spectra_refs)} \n")

        # warn if some of the spectra given are not used in either the FITS files or the point source table
        if not (set(point_refs).union(set(patch_refs))).issubset(spectra_refs):
            raise Warning(f"Unused spectra: {(set(spectra_refs) - set(patch_refs)) - set(point_refs)}")








def examples():



    coordinates = [SkyCoord([10], [20], unit="deg"), #perfect
                   SkyCoord([10,103], [20,23], unit="deg",obstime='2001-01-02T12:34:56'),#more than one ra/dec
                   SkyCoord([10], [20], unit="deg",obstime='2011-01-02T12:34:56')] #wrong time

    background = [True, #yes
                  "not a bool ):"] #no

    time = [astropy.time.Time('2001-01-02T12:34:56'), #yes
            '2001-01-02T12:34:56'] #no

    #perfect
    constant_good_table = astropy.table.QTable(names=["ra", "dec", "mag"], dtype=('f4', 'f4','f4'))
    constant_good_table.add_row((2.0, 3.0, 0.5))
    constant_good_table.add_row((6.0, 2.0, 0.9))
    constant_good_table.units = [astropy.units.deg, astropy.units.deg, synphot.units.PHOTLAM]



    #no units
    constant_bad_table1 = astropy.table.QTable(names=["ra", "dec", "mag"], dtype=('f4', 'f4', 'f4'))
    constant_bad_table1.add_row((2.0, 3.0, 0.5))
    constant_bad_table1.add_row((6.0, 2.0, 0.9))

    #not enough units
    constant_bad_table2 = astropy.table.QTable(names=["ra", "dec", "mag"], dtype=('f4', 'f4', 'f4'))
    constant_bad_table2.add_row((2.0, 3.0, 0.5))
    constant_bad_table2.add_row((6.0, 2.0, 0.9))
    constant_bad_table2.units = [astropy.units.deg, astropy.units.deg]

    constant_tables = [constant_good_table, constant_bad_table1, constant_bad_table2]

    #perfect
    good_table = astropy.table.QTable(names=["ra", "dec", "ref", "scale"],dtype=('f4', 'f4', 'str','f4'))
    good_table.add_row((2.0,3.0,"spectra1",0.5))
    good_table.add_row((6.0, 2.0, "spectra2", 0.9))
    good_table.units = [astropy.units.deg, astropy.units.deg]

    # extra header
    bad_table_1 = astropy.table.QTable(names=["ra", "dec", "ref", "scale", "extra"],dtype=('f4', 'f4', 'S2','f4','f4'))
    bad_table_1.add_row((2.0, 3.0, "spectra1", 0.5,0.4))
    bad_table_1.add_row((6.0, 2.0, "spectra2", 0.9,0.7))
    bad_table_1.units = [astropy.units.deg, astropy.units.deg]

    # wrong headers
    bad_table_2 = astropy.table.QTable(names=["rah", "deck", "ref", "weight"],dtype=('f4', 'f4', 'S2','f4'))
    bad_table_2.add_row((2.0, 3.0, "spectra1", 0.5))
    bad_table_2.add_row((6.0, 2.0, "spectra2", 0.9))
    bad_table_2.units = [astropy.units.deg, astropy.units.deg]

    # empty ref string
    bad_table_3 = astropy.table.QTable(names=["ra", "dec", "ref", "scale"],dtype=('f4', 'f4', 'S2','f4'))
    bad_table_3.add_row((2.0, 3.0, " ", 0.5))
    bad_table_3.add_row((6.0, 2.0, "spectra1", 0.9))
    bad_table_3.units = [astropy.units.deg, astropy.units.deg]


    # wrong type for ref
    bad_table_5 = astropy.table.QTable(names=["ra", "dec", "ref", "scale"])
    bad_table_5.add_row((2.0, 3.0, 2, 0.5))
    bad_table_5.add_row((6.0, 2.0, 5, 0.9))
    bad_table_5.units = [astropy.units.deg, astropy.units.deg]

    tables = [good_table, bad_table_1, bad_table_2, bad_table_3, bad_table_5]

    good_spectra = {"spectra1": SourceSpectrum(BlackBodyNorm1D, temperature=6000),
                    "spectra2": SourceSpectrum(BlackBodyNorm1D, temperature=400)}


    bad_spectra_1 = {
        "spectra1": SourceSpectrum(BlackBodyNorm1D, temperature=6000),
    }

    bad_spectra_2 = {
        "spectra1": SourceSpectrum(BlackBodyNorm1D, temperature=6000),
        "spectra2": "not a source spectrum object"
    }

    bad_spectra_3 = {
        "spectra1": SourceSpectrum(BlackBodyNorm1D, temperature=6000),
        "spectra1": SourceSpectrum(BlackBodyNorm1D, temperature=3000)
    }

    spectra = [good_spectra,bad_spectra_1,bad_spectra_2,bad_spectra_3]


    hdu = fits.PrimaryHDU(data=[1,2,3])
    hdu.writeto("good.fits", overwrite=True)

    with open('badfits.txt', 'w') as f:
        f.write('not a fits',)

    patch_paths = [["good.fits"], ["bad.fits"], ["nonexistant.fits"]]


    good_patches = astropy.table.QTable(names=["ra", "dec", "ref", "path"], dtype=["f4", "f4", "str", "str"])
    good_patches.add_row(("20.0","3.0", "spectra1", "good.fits"))
    good_patches.units = [astropy.units.deg, astropy.units.deg, astropy.units.deg]

    bad_patches2 = astropy.table.QTable(names=["ra", "dec", "ref", "path"], dtype=["f4", "f4", "str", "str"])
    bad_patches2.add_row(("20.0", "3.0", "spectra1", "badfits.txt"))
    bad_patches2.units = [astropy.units.deg, astropy.units.deg, astropy.units.deg]

    bad_patches1 = astropy.table.QTable(names=["ra", "dec", "ref", "path"], dtype=["f4", "f4", "str", "str"])
    bad_patches1.add_row(("20.0", "3.0", "spectra1", "nonexistant.fits"))
    bad_patches1.units = [astropy.units.deg, astropy.units.deg, astropy.units.deg]

    patches = [good_patches,bad_patches1,bad_patches2]


    uvex = UVEXInput.__new__(UVEXInput)


    #for each of coordinates, time, tables, constant_tables, patches, and spectra, the 0th index passes and all else fails
    UVEXInput.__init__(uvex, coordinates[0], time[0], False, tables[0],  constant_tables[0],patches[0],spectra[0])




examples()

