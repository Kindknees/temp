# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LightSim2grid, LightSim2grid implements a c++ backend targeting the Grid2Op platform.

import faulthandler
faulthandler.enable()

__version__ = "0.7.3"

__all__ = ["newtonpf", "SolverType", "ErrorType", "solver"]

# import directly from c++ module
from lightsim2grid.solver import SolverType
from lightsim2grid.solver import ErrorType

try:
    from lightsim2grid.lightSimBackend import LightSimBackend
    __all__.append("LightSimBackend")
except ImportError:
    # grid2op is not installed, the Backend will not be available
    pass

try:
    from lightsim2grid.physical_law_checker import PhysicalLawChecker
    __all__.append("PhysicalLawChecker")
except ImportError as exc_:
    # grid2op is not installed, the PhysicalLawChecker will not be available
    pass

try:
    from lightsim2grid.timeSerie import TimeSerie
    __all__.append("TimeSerie")
    __all__.append("timeSerie")
except ImportError as exc_:
    # grid2op is not installed, the TimeSeries module will not be available
    pass

try:
    from lightsim2grid.securityAnalysis import SecurityAnalysis
    __all__.append("SecurityAnalysis")
    __all__.append("securityAnalysis")
except ImportError as exc_:
    # grid2op is not installed, the SecurtiyAnalysis module will not be available
    pass
