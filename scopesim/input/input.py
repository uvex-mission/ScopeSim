# -*- coding: utf-8 -*-
"""Module for generating ScopeSim source object(s) for UVEX from user input
   Taking into account telescope orientation, background, etc."""


logger = get_logger(__name__)


class UVEXInput:
    """
    Class for generating a ScopeSim compatible input object for UVEX

    Parameters
    ----------
    input : ...
    
    coord : SkyCoord
    
    date : Astropy Time

    Attributes
    ----------
    source

    Methods
    -------
    add_source

    Examples
    --------
    TBD

    """
    def __init__(input='', coord='', date=''):
        # Validate that input/coord/date are all in an accepted form
        # Check date and coord
        self._validate_input(input) # Check input (its own function)
        # We should be able to operate without an input as well for a blank sky background
        
        # Determine roll from coord and date
        # Define the UVEX field of view
        
        # Create a ScopeSim Source object for the background
        # - Generate WCS for the header
        # - We should do this even if we want to set sky background to zero
        #   just so we have a well-defined field of view and WCS to add more sources to
        # Initialize a ScopeSim source object containing the background object
        
        # If input exists:
        # - Transform the input to the UVEX FoV coordinates
        # - Apply extinction (flag for whether or not it is already applied)
        # - Create a ScopeSim Source object for the input
        # - Add to the ScopeSim source object
        
        
    def _validate_input(self, input):
        # Check that input is in an accepted form
        # Input: (Table with [ra,dec,mag], spectrum)
        #     or (2D image, spectrum) or (Table with [ra,dec,2Dimage?], spectrum)
        # where spectrum is a full spectrum or a string indicating shape ('flat','blackbody','powerlaw') plus an addition parameter (temperature, index)
        # Basically, split out 'input' into as many parameters as we need to be looking for
        # and figure out the logic to make the check
        # ScopeSim templates are separated into spatial and spectral elements
        # so it's okay if we divide them up early on
        
        # Perform any unit conversions -> magnitudes to AB mag system
        
        # Potentially also return input in some more generic form easier to work with?

    def add_source(self, input):
        # Takes an additional input and adds it to the ScopeSim Source object
        self._validate_input(input)
        
        # Generate a Source object and add to source as in init
    
    def get_scopesim_source(self):
        # Getter for the overall ScopeSim source object once finished adding stuff to it
