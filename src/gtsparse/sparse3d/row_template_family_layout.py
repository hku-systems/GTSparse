from __future__ import annotations

SEGMENT_FAMILY_NAMES = ("center", "single", "pair", "quad", "sept", "full27")

FAMILY_CENTER = 0
FAMILY_SINGLE = 1
FAMILY_PAIR = 2
FAMILY_QUAD = 3
FAMILY_SEPT = 4
FAMILY_FULL27 = 5

FAMILY_CENTER_START = 0
FAMILY_SINGLE_START = 1
FAMILY_PAIR_START = 27
FAMILY_QUAD_START = 40
FAMILY_SEPT_START = 47
FAMILY_FULL27_START = 51

FAMILY_CENTER_COUNT = 1
FAMILY_SINGLE_COUNT = 26
FAMILY_PAIR_COUNT = 13
FAMILY_QUAD_COUNT = 7
FAMILY_SEPT_COUNT = 4
FAMILY_FULL27_COUNT = 1

FAMILY_CENTER_END = FAMILY_SINGLE_START - 1
FAMILY_SINGLE_END = FAMILY_PAIR_START - 1
FAMILY_PAIR_END = FAMILY_QUAD_START - 1
FAMILY_QUAD_END = FAMILY_SEPT_START - 1
FAMILY_SEPT_END = FAMILY_FULL27_START - 1
FAMILY_FULL27_END = FAMILY_FULL27_START + FAMILY_FULL27_COUNT - 1

FAMILY_CENTER_PAYLOAD_WIDTH = 2
FAMILY_SINGLE_PAYLOAD_WIDTH = 3
FAMILY_PAIR_PAYLOAD_WIDTH = 4
FAMILY_QUAD_PAYLOAD_WIDTH = 6
FAMILY_SEPT_PAYLOAD_WIDTH = 9
FAMILY_FULL27_PAYLOAD_WIDTH = 28

FAMILY_CENTER_RING_STRIDE = 0
FAMILY_SINGLE_RING_STRIDE = 1
FAMILY_PAIR_RING_STRIDE = 2
FAMILY_QUAD_RING_STRIDE = 4
FAMILY_SEPT_RING_STRIDE = 7
FAMILY_FULL27_RING_STRIDE = 0

SEGMENT_TEMPLATE_COUNTS = (
    FAMILY_CENTER_COUNT,
    FAMILY_SINGLE_COUNT,
    FAMILY_PAIR_COUNT,
    FAMILY_QUAD_COUNT,
    FAMILY_SEPT_COUNT,
    FAMILY_FULL27_COUNT,
)
SEGMENT_PAYLOAD_WIDTHS = (
    FAMILY_CENTER_PAYLOAD_WIDTH,
    FAMILY_SINGLE_PAYLOAD_WIDTH,
    FAMILY_PAIR_PAYLOAD_WIDTH,
    FAMILY_QUAD_PAYLOAD_WIDTH,
    FAMILY_SEPT_PAYLOAD_WIDTH,
    FAMILY_FULL27_PAYLOAD_WIDTH,
)

SEGMENT_TEMPLATE_STARTS = (
    FAMILY_CENTER_START,
    FAMILY_SINGLE_START,
    FAMILY_PAIR_START,
    FAMILY_QUAD_START,
    FAMILY_SEPT_START,
    FAMILY_FULL27_START,
)

NUM_SEGMENT_TEMPLATES = 52
STRIDE_TEMPLATE_END = FAMILY_PAIR_END

TEMPLATE_TO_FAMILY = tuple(
    [FAMILY_CENTER] * FAMILY_CENTER_COUNT
    + [FAMILY_SINGLE] * FAMILY_SINGLE_COUNT
    + [FAMILY_PAIR] * FAMILY_PAIR_COUNT
    + [FAMILY_QUAD] * FAMILY_QUAD_COUNT
    + [FAMILY_SEPT] * FAMILY_SEPT_COUNT
    + [FAMILY_FULL27] * FAMILY_FULL27_COUNT
)
TEMPLATE_LOCAL_INDEX = tuple(
    [0]
    + list(range(FAMILY_SINGLE_COUNT))
    + list(range(FAMILY_PAIR_COUNT))
    + list(range(FAMILY_QUAD_COUNT))
    + list(range(FAMILY_SEPT_COUNT))
    + [0]
)


def family_from_template_id(template_id: int) -> int:
    return int(TEMPLATE_TO_FAMILY[int(template_id)])


def local_index_from_template_id(template_id: int) -> int:
    return int(TEMPLATE_LOCAL_INDEX[int(template_id)])


def payload_width_from_template_id(template_id: int) -> int:
    family_id = family_from_template_id(int(template_id))
    return int(SEGMENT_PAYLOAD_WIDTHS[family_id])


def family_base_from_template_id(template_id: int) -> int:
    template_id = int(template_id)
    if template_id <= FAMILY_CENTER_END:
        return 0
    if template_id <= FAMILY_SINGLE_END:
        return template_id - FAMILY_SINGLE_START
    if template_id <= FAMILY_PAIR_END:
        return FAMILY_PAIR_RING_STRIDE * (template_id - FAMILY_PAIR_START)
    if template_id <= FAMILY_QUAD_END:
        return FAMILY_QUAD_RING_STRIDE * (template_id - FAMILY_QUAD_START)
    if template_id <= FAMILY_SEPT_END:
        return FAMILY_SEPT_RING_STRIDE * (template_id - FAMILY_SEPT_START)
    return 0


def family_count_from_template_id(template_id: int) -> int:
    template_id = int(template_id)
    if template_id <= FAMILY_CENTER_END:
        return 0
    if template_id <= FAMILY_SINGLE_END:
        return 1
    if template_id <= FAMILY_PAIR_END:
        return 2
    if template_id <= FAMILY_QUAD_END:
        return 4
    if template_id <= FAMILY_SEPT_END:
        return 7
    return 26
