"""Exception hierarchy.  Every error raised by AURA-QEC derives from AuraError."""


class AuraError(Exception):
    """Base class for all AURA-QEC errors."""


class ConfigError(AuraError):
    """Invalid or inconsistent configuration."""


class CodeError(AuraError):
    """An error-correcting code failed validation."""


class CompileError(AuraError):
    """The QEC compiler could not produce a valid program."""


class TimingViolation(AuraError):
    """Two hardware operations overlapped on one qubit, or a deadline was missed
    in a way the hardware cannot tolerate."""


class HardwareFault(AuraError):
    """A virtual hardware component entered an unrecoverable state."""


class CalibrationError(AuraError):
    """A calibration experiment or fit failed."""


class LinkError(AuraError):
    """AURA-Link transport failure (session, PHY, flow control)."""


class ProtocolError(LinkError):
    """A frame violated the AURA-Link wire protocol."""
