# Tests: check inference common behavior using saved artifacts or temporary inputs.

import pytest
from src.inference.common import risk_level


# Test stage: risk band contract.
@pytest.mark.parametrize('width', [0.5, 0.1])
def test_risk_band_contract(width):
    assert risk_level(-width - 0.01, 0, width) == 'LOW'
    assert risk_level(-width, 0, width) == 'MEDIUM'
    assert risk_level(0, 0, width) == 'MEDIUM'
    assert risk_level(width, 0, width) == 'HIGH'
