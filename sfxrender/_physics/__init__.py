"""Private perceptual physics building blocks for procedural Foley."""

from .closure import render_terminal_closure
from .electromechanical import render_electromechanical_hum
from .geometry import (
    BoundaryCondition,
    LumpedBody,
    RectangularPlate,
    plate_mode_shape,
    plate_modes,
    rectangular_plate_modes,
)
from .materials import MATERIALS, MechanicalMaterial
from .modes import Mode, ModeSet, decay_from_q, filter_audio_modes, q_from_decay
from .physical_impacts import (
    ImpactResponse,
    render_coupled_impact,
    render_force_response,
    render_physical_impact,
)
from .radiation import radiation_efficiency
from .resonator import ModalResonatorBank, modal_response
from .rng import RandomStream, component_rng, event_rng
from .rotating import render_rotating_machine
from .sliding import render_sliding_source
from .stochastic import StochasticEvent, sample_stochastic_events
from .strikes import (
    StrikeEvent,
    StruckResonatorPreset,
    build_contact_force_bus,
    render_strike_train,
)
from .thin_material import (
    ThinMaterialPreset,
    generate_thin_material_preset,
    render_thin_material_source,
)
from .turbulence import render_turbulence_layer

__all__ = [
    "MATERIALS",
    "BoundaryCondition",
    "ImpactResponse",
    "LumpedBody",
    "MechanicalMaterial",
    "ModalResonatorBank",
    "Mode",
    "ModeSet",
    "RandomStream",
    "RectangularPlate",
    "StochasticEvent",
    "StrikeEvent",
    "StruckResonatorPreset",
    "ThinMaterialPreset",
    "build_contact_force_bus",
    "component_rng",
    "decay_from_q",
    "event_rng",
    "filter_audio_modes",
    "generate_thin_material_preset",
    "modal_response",
    "plate_mode_shape",
    "plate_modes",
    "q_from_decay",
    "radiation_efficiency",
    "rectangular_plate_modes",
    "render_coupled_impact",
    "render_electromechanical_hum",
    "render_force_response",
    "render_physical_impact",
    "render_rotating_machine",
    "render_sliding_source",
    "render_strike_train",
    "render_terminal_closure",
    "render_thin_material_source",
    "render_turbulence_layer",
    "sample_stochastic_events",
]
