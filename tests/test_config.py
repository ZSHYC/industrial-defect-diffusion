from industrial_defect.config import (
    CATEGORY_PROMPTS,
    defect_types_for_category,
    get_category_config,
    supported_categories,
)


def test_supported_categories() -> None:
    assert supported_categories() == ["leather", "tile", "wood"]


def test_defect_types_for_category() -> None:
    assert defect_types_for_category("tile") == ["crack", "glue_strip", "gray_stroke", "oil", "rough"]
    assert defect_types_for_category("wood") == ["color", "hole", "liquid", "scratch", "combined"]
    assert defect_types_for_category("wood", evaluation_order=True) == ["color", "combined", "hole", "liquid", "scratch"]
    assert defect_types_for_category("leather") == ["color", "cut", "fold", "glue", "poke"]


def test_prompts_cover_defect_types() -> None:
    for category in supported_categories():
        config = get_category_config(category)
        assert set(config.prompts) == set(config.defect_types)
        for defect_type, prompt in CATEGORY_PROMPTS[category].items():
            assert defect_type in config.defect_types
            assert "inspection" in prompt

