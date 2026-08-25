import math
from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from py_mtlf.models import SpecAlignedModel


class StrictSpecModel(SpecAlignedModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class VolumeMeasurement(StrictSpecModel):
    total_volume: int = Field(alias="totalVolume", ge=0)
    ul_volume: int = Field(alias="ulVolume", ge=0)
    dl_volume: int = Field(alias="dlVolume", ge=0)
    total_nb_of_packets: int = Field(alias="totalNbOfPackets", ge=0)
    ul_nb_of_packets: int = Field(alias="ulNbOfPackets", ge=0)
    dl_nb_of_packets: int = Field(alias="dlNbOfPackets", ge=0)


class ThroughputMeasurement(StrictSpecModel):
    ul_throughput: str = Field(alias="ulThroughput")
    dl_throughput: str = Field(alias="dlThroughput")
    ul_packet_throughput: str = Field(alias="ulPacketThroughput")
    dl_packet_throughput: str = Field(alias="dlPacketThroughput")

    @field_validator("ul_throughput", "dl_throughput")
    @classmethod
    def validate_bit_rate(cls, value: str) -> str:
        return _validate_rate(value, {"bps", "Kbps", "Mbps", "Gbps", "Tbps"})

    @field_validator("ul_packet_throughput", "dl_packet_throughput")
    @classmethod
    def validate_packet_rate(cls, value: str) -> str:
        return _validate_rate(value, {"pps", "kpps", "Mpps", "Gpps", "Tpps"})


class UserDataUsageMeasurement(StrictSpecModel):
    volume_measurement: VolumeMeasurement = Field(alias="volumeMeasurement")
    throughput_measurement: ThroughputMeasurement = Field(alias="throughputMeasurement")


class NotificationItem(StrictSpecModel):
    event_type: Literal["USER_DATA_USAGE_MEASURES"] = Field(alias="eventType")
    ue_ipv4_addr: str = Field(default="", alias="ueIpv4Addr")
    ue_ipv6_prefix: str = Field(default="", alias="ueIpv6Prefix")
    ue_mac_addr: str = Field(default="", alias="ueMacAddr")
    dnn: str = ""
    supi: str = ""
    snssai: dict | None = None
    time_stamp: datetime = Field(alias="timeStamp")
    start_time: datetime | None = Field(default=None, alias="startTime")
    user_data_usage_measurements: tuple[UserDataUsageMeasurement, ...] = Field(
        min_length=1,
        alias="userDataUsageMeasurements",
    )

    @field_validator("time_stamp", "start_time")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("UPF notification timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def require_ue_address(self) -> "NotificationItem":
        if not (self.ue_ipv4_addr or self.ue_ipv6_prefix or self.ue_mac_addr):
            raise ValueError("a UE IPv4 address, IPv6 prefix or MAC address is required")
        return self


class NotificationData(StrictSpecModel):
    correlation_id: str = Field(min_length=1, alias="correlationId")
    notification_items: tuple[NotificationItem, ...] = Field(
        min_length=1,
        alias="notificationItems",
    )
    achieved_sampling_ratio: int | None = Field(default=None, alias="achievedSampRatio")


def _validate_rate(value: str, allowed_units: set[str]) -> str:
    parts = value.strip().split()
    if len(parts) != 2 or parts[1] not in allowed_units:
        raise ValueError("throughput measurement has an unsupported unit")
    try:
        number = float(parts[0])
    except ValueError as error:
        raise ValueError("throughput measurement must be numeric") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError("throughput measurement must be finite and non-negative")
    return value.strip()
