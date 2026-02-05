"""
Sample policy database for validation.
In production, this would be replaced with a real database query.
"""
from schemas.models import PolicyData, PolicyCoverage
from typing import Optional


# Sample policy database
POLICY_DATABASE = {
    "POL-2024-001": PolicyData(
        policy_number="POL-2024-001",
        coverage_types=["collision", "comprehensive", "liability"],
        max_payout=500000.0,
        exclusions=["intentional_damage", "racing", "drunk_driving"],
        coverages=[
            PolicyCoverage(coverage_type="collision", max_limit=300000.0, deductible=5000.0),
            PolicyCoverage(coverage_type="comprehensive", max_limit=400000.0, deductible=3000.0),
            PolicyCoverage(coverage_type="liability", max_limit=500000.0, deductible=0.0)
        ]
    ),
    "POL-2024-002": PolicyData(
        policy_number="POL-2024-002",
        coverage_types=["collision", "liability"],
        max_payout=300000.0,
        exclusions=["intentional_damage", "racing", "drunk_driving", "natural_disaster"],
        coverages=[
            PolicyCoverage(coverage_type="collision", max_limit=250000.0, deductible=7000.0),
            PolicyCoverage(coverage_type="liability", max_limit=300000.0, deductible=0.0)
        ]
    ),
    "POL-2024-003": PolicyData(
        policy_number="POL-2024-003",
        coverage_types=["comprehensive", "collision", "liability", "theft"],
        max_payout=750000.0,
        exclusions=["intentional_damage", "racing"],
        coverages=[
            PolicyCoverage(coverage_type="collision", max_limit=500000.0, deductible=3000.0),
            PolicyCoverage(coverage_type="comprehensive", max_limit=600000.0, deductible=2000.0),
            PolicyCoverage(coverage_type="liability", max_limit=750000.0, deductible=0.0),
            PolicyCoverage(coverage_type="theft", max_limit=500000.0, deductible=5000.0)
        ]
    )
}


def get_policy(policy_number: str) -> Optional[PolicyData]:
    """
    Retrieve policy data by policy number.
    
    Args:
        policy_number: Policy identifier
        
    Returns:
        PolicyData if found, None otherwise
    """
    return POLICY_DATABASE.get(policy_number)


def get_default_policy() -> PolicyData:
    """
    Get a default policy for testing purposes.
    
    Returns:
        Default PolicyData
    """
    return PolicyData(
        policy_number="DEFAULT",
        coverage_types=["collision", "comprehensive"],
        max_payout=500000.0,
        exclusions=["intentional_damage", "racing", "drunk_driving"],
        coverages=[
            PolicyCoverage(coverage_type="collision", max_limit=300000.0, deductible=5000.0),
            PolicyCoverage(coverage_type="comprehensive", max_limit=400000.0, deductible=3000.0)
        ]
    )
