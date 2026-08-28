"""Vega <-> AB magnitude conversions.

Sign convention, applied uniformly to every entry in `OFFSETS`:

    offset = m_Vega - m_AB

so that m_AB = m_Vega - offset and m_Vega = m_AB + offset.
"""

from dataclasses import dataclass
import difflib
import numpy as np

FREI_GUNN_1994 = "Frei & Gunn 1994, AJ, 108, 1476 (Table 2)"
BIANCHI_SHIAO_2020 = "Bianchi & Shiao 2020, ApJS, 250, 36 (Table 2)"

@dataclass(frozen=True)
class Offset:
    """m_Vega - m_AB for a single bandpass."""
    value: float
    reference: str
    uncertainty: float | None = None

OFFSETS: dict[tuple[str, str], Offset] = {
    ("johnson", "b"): Offset(0.163, FREI_GUNN_1994, 0.004),
    ("johnson", "v"): Offset(0.044, FREI_GUNN_1994, 0.004),
    ("gaia", "g"): Offset(-0.092, BIANCHI_SHIAO_2020),
    ("gaia", "bp"): Offset(-0.066, BIANCHI_SHIAO_2020),
    ("gaia", "rp"): Offset(-0.362, BIANCHI_SHIAO_2020)
}

_FILTER_SYNONYMS = {"g_bp": "bp", "gbp": "bp", "g_rp": "rp", "grp": "rp"}

def lookup(band: str) -> Offset:
    """Return the tabulated Vega-minus-AB offset entry for a named bandpass.

    The band string is matched case-insensitively and is expected to take
    the form "<system> <filter>", e.g. "Gaia G_BP". Alternative
    filter spellings listed in `_FILTER_SYNONYMS` are accepted.

    Parameters
    ----------
    band : str
        String identifier for the photometric system and band.
    
    Returns
    -------
    offset : Offset
        Table entry holding the Vega-minus-AB offset, its literature
        reference, and its uncertainty where one is published.
    
    Raises
    ------
    ValueError
        If no table entry exists for `band`.
    """
    system, _, filt = band.strip().lower().partition(" ")
    key = (system, _FILTER_SYNONYMS.get(filt, filt))
    try:
        return OFFSETS[key]
    except KeyError:
        known = [f"{s} {f}" for s, f in sorted(OFFSETS)]
        close = difflib.get_close_matches(f'{key[0]} {key[1]}', known, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(
            f"No Vega-AB offset for {band!r}.{hint} "
            f"Known bands: {', '.join(known)}"
        ) from None

def vega_to_ab(mvega, band: str):
    """Convert apparent magnitudes form the Vega system to the AB system.

    Parameters
    ----------
    mvega : numpy.ndarray
        Apparent magnitude in the Vega photometric system.
    band : str
        String identifier for the photometric system and band.
 
    Returns
    -------
    mab : numpy.ndarray
        Apparent magnitude in the AB photometric system.
    """
    return np.asarray(mvega, dtype=float) - lookup(band).value

def ab_to_vega(mab, band: str):
    """Ditto, but for AB-to-Vega conversion."""
    return np.asarray(mab, dtype=float) + lookup(band).value