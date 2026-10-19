import pytest
import yaml

from dog_crash_sim.config import (BREED_SIZES, ScenarioConfig, commented_template, config_from_dict,
                                  dog_mass_and_height, params_table)
from dog_crash_sim.presets import PRESETS, preset


def test_default_config_is_valid():
    cfg = config_from_dict({})
    assert cfg == ScenarioConfig()


def test_template_roundtrip():
    cfg = config_from_dict({"collision": {"impact_speed_kph": 40.0}, "trunk_cargo": {"items": []}})
    text = commented_template(cfg)
    assert "# Speed of the subject vehicle" in text
    assert config_from_dict(yaml.safe_load(text)) == cfg


def test_all_six_categories_are_present():
    names = {c for c in ScenarioConfig.__dataclass_fields__}
    for cat in ("collision", "vehicle", "rear_passenger", "restraints", "trunk_cargo", "environment"):
        assert cat in names
    table = params_table()
    for var in ("impact_speed_kph", "impact_direction", "impact_angle_deg", "overlap", "partner.type",
                "mass_kg", "structure", "crush_zones_front", "rear_compartment", "breed_size", "seating_position",
                "posture", "seat_belt_used", "belt", "pretensioner", "load_limiter", "airbags", "seat",
                "head_restraints", "items", "packing", "restraint_method", "road_friction", "braking", "motion"):
        assert var in table, var


def test_unknown_parameter_and_bad_choice_raise():
    with pytest.raises(KeyError):
        config_from_dict({"collision": {"impact_sped_kph": 30}})
    with pytest.raises(ValueError):
        config_from_dict({"rear_passenger": {"posture": "flying"}})
    with pytest.raises(ValueError):
        config_from_dict({"collision": {"overlap": 1.5}})


def test_vehicle_preset_merge_keeps_overrides():
    cfg = config_from_dict({"vehicle": {"preset": "midsize_suv", "mass_kg": 2222.0}})
    assert cfg.vehicle.body_style == "suv"
    assert cfg.vehicle.mass_kg == 2222.0
    assert cfg.vehicle.rear_compartment.cushion_height_m == pytest.approx(0.78)


def test_dog_size_resolution():
    for name, (m, h) in BREED_SIZES.items():
        cfg = config_from_dict({"rear_passenger": {"breed_size": name}})
        assert dog_mass_and_height(cfg.rear_passenger) == (m, h)
    cfg = config_from_dict({"rear_passenger": {"breed_size": "custom", "mass_kg": 30.0}})
    mass, height = dog_mass_and_height(cfg.rear_passenger)
    assert mass == 30.0 and 0.5 < height < 0.7


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_validate(name):
    cfg = config_from_dict(preset(name))
    assert cfg.name == name
