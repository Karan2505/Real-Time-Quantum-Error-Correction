"""AURA-Link: the real-time QEC transport (PHY, framing, time sync, ingress gateway)."""

from .frames import FrameType, Header, encode_frame, decode_header, VERSION, HEADER_LEN
from .phy import PhyConfig, Impairments, GilbertElliott, PhyTx, PhyRx, SerialChannel
from .ingress import ArrivalTimestampTagger, FrameParser, IngressElasticFifo, SequenceTracker, IngressSink
from .timesync import TimeSyncConfig, TimeSyncMaster, TimeSyncSlave
from .endpoint import LinkConfig, ClockConfig, LinkState, ControlEndpoint, IngressGateway
from .link import SimLink, build_sim_link
from .records import SyndromeRecord, CorrectionRecord, ErasureRecord

__all__ = [n for n in dir() if not n.startswith("_")]
