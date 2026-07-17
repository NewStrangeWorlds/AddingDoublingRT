References
==========

Foundational method
-------------------

* Plass, G. N., Kattawar, G. W., & Catchings, F. E. (1973).
  *Matrix operator theory of radiative transfer. 1: Rayleigh scattering.*
  Applied Optics, 12(2), 314–329.
* Hansen, J. E. (1971).
  *Multiple scattering of polarized light in planetary atmospheres. Part II.
  Sunlight reflected by terrestrial water clouds.*
  Journal of the Atmospheric Sciences, 28(8), 1400–1426.


Doubling extensions
-------------------

* Wiscombe, W. J. (1975).
  *On initialization, error and flux conservation in the doubling method.*
  Journal of Quantitative Spectroscopy and Radiative Transfer, 16, 637–658.
* Wiscombe, W. J. (1976).
  *Extension of the doubling method to inhomogeneous sources.*
  Journal of Quantitative Spectroscopy and Radiative Transfer, 16, 477–489.
* Wiscombe, W. J. (1977).
  *The delta-M method: Rapid yet accurate radiative flux calculations for
  strongly asymmetric phase functions.*
  Journal of the Atmospheric Sciences, 34(9), 1408–1422.


Linearisation and remote sensing
--------------------------------

The analytic temperature-Jacobian propagation follows the linearisation of the
interaction principle; the accompanying literature (in the repository's
``literature/`` directory) develops the linearised / doubling–adding machinery
used here:

* Spurr, R., & Christi, M.
  *Linearization of the interaction principle: analytic Jacobians.*
  (``literature/Spurr_Christi.pdf``.)
* Hasekamp, O. P., & Landgraf, J. (2001).
  *Ozone profile retrieval from backscattered ultraviolet radiances: The
  inverse problem solved by regularization.*
  (``literature/Hasekamp_Landgraf_2001.pdf``.)
* Hasekamp, O. P., & Landgraf, J. (2005).
  *Linearization of vector radiative transfer with respect to aerosol
  properties and its use in satellite remote sensing.*
  (``literature/Hasekamp_2005.pdf``.)
* Liu, Q., & Weng, F. (2006).
  *Advanced doubling–adding method for radiative transfer in planetary
  atmospheres.*  Journal of the Atmospheric Sciences, 63(12), 3459–3465.
  (``literature/Liu_Weng_2006.pdf``.)


Project documents
-----------------

The ``tex/`` directory of the repository contains the internal project notes
this documentation is based on:

* ``adding_doubling_description.tex`` — the mathematical description of the
  forward solver (:doc:`theory/index`).
* ``temperature_jacobian_plan.tex`` — the analytic temperature-Jacobian
  implementation plan (:doc:`theory/jacobian`).
* ``heating_jacobian_depth_limit.tex`` — analysis of the depth limit and
  conditioning of the heating-rate Jacobian (:ref:`heating-depth-limit`).
