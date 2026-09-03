MODEL_PROVISION_EXT_FEATURE = 4
HIERARCHICAL_FL_ORCHESTRATION_FEATURE = 3


def feature_mask(*feature_numbers: int) -> str:
    value = 0
    for feature_number in feature_numbers:
        if feature_number <= 0:
            raise ValueError("3GPP feature numbers must be positive")
        value |= 1 << (feature_number - 1)
    return format(value, "X")


def feature_intersection(requested: str, supported: str) -> str:
    try:
        requested_value = int(requested or "0", 16)
        supported_value = int(supported or "0", 16)
    except ValueError as error:
        raise ValueError("supported features must be hexadecimal bitmasks") from error
    value = requested_value & supported_value
    return format(value, "X") if value else ""


def includes_feature(value: str, feature_number: int) -> bool:
    try:
        parsed = int(value or "0", 16)
    except ValueError:
        return False
    bit = 1 << (feature_number - 1)
    return parsed & bit == bit
