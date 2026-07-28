"""EXTRACT step: type synonyms, near-dupes, invalid lat/lng."""

from worker.extract import extract_facilities


def test_type_synonym_mapping():
    raw = [
        {
            "facility_name": "Plant A",
            "facility_address": "1 Industrial Rd, Detroit, MI",
            "facility_type": "factory",
            "confidence": "high",
        },
        {
            "facility_name": "DC West",
            "facility_address": "2 Warehouse Blvd, Reno, NV",
            "facility_type": "warehouse",
            "confidence": "medium",
        },
        {
            "facility_name": "Mine 9",
            "facility_address": "Pit Road, Atacama, Chile",
            "facility_type": "smelter",
            "confidence": "low",
        },
    ]
    out = extract_facilities(raw)
    assert len(out) == 3
    assert out[0].facility_type.value == "manufacturing"
    assert out[1].facility_type.value == "logistics"
    assert out[2].facility_type.value == "raw_material"


def test_expanded_type_synonyms_not_dropped():
    raw = [
        {
            "facility_name": "West Coast Terminal",
            "facility_address": "1 Port Rd, Long Beach, CA",
            "facility_type": "freight terminal",
            "confidence": "medium",
        },
        {
            "facility_name": "Highlands Tannery",
            "facility_address": "5 Tannery Ln, Fez, Morocco",
            "facility_type": "tannery",
            "confidence": "high",
        },
        {
            "facility_name": "Acme Contract Plant",
            "facility_address": "9 Contract Rd, Ho Chi Minh City, Vietnam",
            "facility_type": "contract manufacturer",
            "confidence": "medium",
        },
    ]
    out = extract_facilities(raw)
    assert len(out) == 3
    assert out[0].facility_type.value == "logistics"
    assert out[1].facility_type.value == "raw_material"
    assert out[2].facility_type.value == "manufacturing"


def test_unknown_type_dropped():
    raw = [
        {
            "facility_name": "HQ",
            "facility_address": "1 Main St",
            "facility_type": "sales_office",
            "confidence": "high",
        }
    ]
    assert extract_facilities(raw) == []


def test_near_duplicate_collapse():
    raw = [
        {
            "facility_name": "Plant A",
            "facility_address": "100 Industrial Parkway, Austin, TX 78701",
            "facility_type": "manufacturing",
            "confidence": "high",
        },
        {
            "facility_name": "Plant A Alt",
            "facility_address": "100 Industrial Parkway, Austin, TX 78701 USA",
            "facility_type": "manufacturing",
            "confidence": "medium",
        },
    ]
    out = extract_facilities(raw)
    assert len(out) == 1


def test_invalid_lat_lng_nulled():
    raw = [
        {
            "facility_name": "Plant B",
            "facility_address": "Somewhere",
            "facility_type": "manufacturing",
            "confidence": "high",
            "latitude": 999.0,
            "longitude": -200.0,
        }
    ]
    out = extract_facilities(raw)
    assert len(out) == 1
    assert out[0].latitude is None
    assert out[0].longitude is None


def test_product_stamped_onto_facilities():
    raw = [
        {
            "facility_name": "Plant A",
            "facility_address": "1 Industrial Rd, Detroit, MI",
            "facility_type": "manufacturing",
            "confidence": "high",
        }
    ]
    out = extract_facilities(raw, product="brake pads")
    assert len(out) == 1
    assert out[0].product == "brake pads"


def test_product_defaults_empty_when_omitted():
    raw = [
        {
            "facility_name": "Plant A",
            "facility_address": "1 Industrial Rd, Detroit, MI",
            "facility_type": "manufacturing",
            "confidence": "high",
        }
    ]
    out = extract_facilities(raw)
    assert out[0].product == ""


def test_empty_name_or_address_dropped():
    raw = [
        {
            "facility_name": "",
            "facility_address": "1 Main",
            "facility_type": "manufacturing",
            "confidence": "high",
        },
        {
            "facility_name": "OK",
            "facility_address": "  ",
            "facility_type": "manufacturing",
            "confidence": "high",
        },
    ]
    assert extract_facilities(raw) == []
