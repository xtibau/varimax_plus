# varimax<sup>+</sup>
Version 0.1


Varimax<sup>+</sup> is a variation of the Varimax algorithm (Kaiser 1958<sup>[1](#kaiser)</sup>) that uses bootstrap in conjunction with a hypothesis test to set to 0 those values of the resulting weights that are likely to be 0. Additionally, the software allows you to apply regular varimax, or other rotations like equamax or parsimax. 
# Installation
You can install varimax<sup>+</sup> using `pip`:

`pip install git+https://github.com/xtibau/varimax_plus.git#egg=varimax_plus`

# Contributions
Any contribution is more than welcome. If you want to collaborate in improving the algorithm, do not hesitate to contact me.  Improvements can be made by adding some tutorials with cool data, improving the algorithm that sorts the weights, or any other cool idea that you may have. 

# Code Example

```python
# For varimax^+
import numpy as np
from varimax_plus.varimax_plus import VarimaxPlus, Varimax

var_dict = {
    "truncate_by": 'max_comps',
    "max_comps": 15,
    "fraction_explained_variance": 0.9,
    "boot_axis": 0,
    "boot_rep": 100,  # The number of final datasets
    "boot_samples": 100,  # The number of samples in each dataset
    "alpha_level": 0.01,  # The \alpha level of the hypothesis t-test
    "verbose": "INFO"  # May be also DEBUG or None. 
}

data = np.random.rand(100, 50)  # Samples x Vars -> Samples x Components

# Instantiate the object
var_p = VarimaxPlus(data=data, **var_dict)

# Call it
var_res = var_p()

# Get the weights
weights = var_res["weights"]

# Get the components
components = data @ weights

# For regular Varimax

varimax_dict = {
    "truncate_by": 'max_comps',
    "max_comps": 15,
    "fraction_explained_variance": 0.9,
    "verbose": True
}
        
varimax = Varimax(data, **varimax_dict)
var_results = varimax()

# Get the weights
weights = var_res["weights"]

# Get the components
components = data @ weights
```

# License
Varimax<sup>+</sup> is a Free Software project under the GNU General Public License v3, which means all its code is available for everyone to download, examine, use, modify, and distribute, subject to the usual restrictions attached to any GPL software. If you are not familiar with the GPL, see the license.txt file for more details on license terms and other legal issues. 

# References

<a name="kaiser">1</a>: Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. _Psychometrika_, 23(3), 187-200.