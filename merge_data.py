import sys
sys.path.append('/Users/utkarshagrawal/Documents/Postdoc/U_1_haar')
from merge_utils import merge

measurement_type = 'proj'
assert measurement_type in ['weak','proj'], print('measurement_type wrong')

setup = 'purification/with_scrambling'
evolution_type = 'matchgate'
scrambling_type = 'matchgate'

merge(measurement_type,setup)