#pragma once

namespace gtsparse_row_template_center_last {

constexpr int kCenterOffset = 13;
constexpr int kNumLogicalOffsets = 27;
constexpr int kMaxPayloadSlots = 27;
constexpr int kBM = 128;
constexpr int kBN = 64;
constexpr int kBK = 32;
constexpr int kThreads = 128;

enum TemplateId : int {
    kTemplateCenter = 0,
    kTemplateSkip2Keep0 = 1,
    kTemplateSkip2Keep1 = 2,
    kTemplateSkip2Keep2 = 3,
    kTemplateSkip1Hole0 = 4,
    kTemplateSkip1Hole1 = 5,
    kTemplateSkip1Hole2 = 6,
    kTemplateFull27 = 7,
    kNumTemplates = 8,
};

enum TemplateFamilyKind : int {
    kFamilyCenter = 0,
    kFamilySkip2Every3 = 1,
    kFamilySkip1Every3 = 2,
    kFamilyFull27 = 3,
};

constexpr int kFamilyW1Templates = 1;
constexpr int kFamilyW9Templates = 3;
constexpr int kFamilyW18Templates = 3;
constexpr int kFamilyW27Templates = 1;

constexpr int kPayloadWidthW1 = 1;
constexpr int kPayloadWidthW9 = 10;
constexpr int kPayloadWidthW18 = 19;
constexpr int kPayloadWidthW27 = 27;

constexpr int kSlotCountW1 = 1;
constexpr int kSlotCountW9Min = 9;
constexpr int kSlotCountW9Max = 10;
constexpr int kSlotCountW18Min = 18;
constexpr int kSlotCountW18Max = 19;
constexpr int kSlotCountW27 = 27;

static constexpr int kLogicalToActual[kNumLogicalOffsets] = {
    0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4,
    22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26,
    13,
};

static constexpr int kTemplateSlotCount[kNumTemplates] = {
    kSlotCountW1,
    10,
    9,
    10,
    18,
    19,
    18,
    kSlotCountW27,
};

static constexpr int kTemplateInitialBias[kNumTemplates] = {
    26,
    0,
    1,
    2,
    1,
    0,
    0,
    0,
};

static constexpr int kTemplateFamily[kNumTemplates] = {
    kFamilyCenter,
    kFamilySkip2Every3,
    kFamilySkip2Every3,
    kFamilySkip2Every3,
    kFamilySkip1Every3,
    kFamilySkip1Every3,
    kFamilySkip1Every3,
    kFamilyFull27,
};

static constexpr int kTemplateLocalIndex[kNumTemplates] = {
    0,
    0,
    1,
    2,
    0,
    1,
    2,
    0,
};

static constexpr int kTemplatePayloadActualOffset[kNumTemplates][kMaxPayloadSlots] = {
    {13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 6, 1, 4, 7, 2, 5, 8, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 12, 15, 10, 16, 11, 14, 17, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {18, 21, 24, 19, 22, 25, 20, 23, 26, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 18, 12, 21, 15, 24, 10, 19, 22, 16, 25, 11, 20, 14, 23, 17, 26, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 18, 3, 21, 6, 24, 1, 19, 4, 22, 7, 25, 2, 20, 5, 23, 8, 26, 13, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 3, 12, 6, 15, 1, 10, 4, 7, 16, 2, 11, 5, 14, 8, 17, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4, 22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26, 13},
};

// Classification reject masks are defined in the builder's ring27-inplace
// match order. Bit 13 is the center slot and is intentionally left as 0 for
// all periodic templates, so center never prevents a periodic-family match.
static constexpr unsigned int kTemplateRejectMask[kNumTemplates] = {
    0x07ffdfffu, // center
    0x06db4db6u, // skip2_keep0
    0x05b6db6du, // skip2_keep1
    0x036d96dbu, // skip2_keep2
    0x01249249u, // skip1_hole0
    0x02490492u, // skip1_hole1
    0x04924924u, // skip1_hole2
    0x00000000u, // full27
};

// The CUDA builders classify directly from the warp ballot over actual offset
// ids [0, 26]. These masks are the same periodic families expressed in that
// actual offset bit layout, with the center bit left as 0.
static constexpr unsigned int kTemplateRejectMaskActual[kNumTemplates] = {
    0x07ffdfffu, // center
    0x07ffde00u, // skip2_keep0
    0x07fc01ffu, // skip2_keep1
    0x0003dfffu, // skip2_keep2
    0x000001ffu, // skip1_hole0
    0x0003de00u, // skip1_hole1
    0x07fc0000u, // skip1_hole2
    0x00000000u, // full27
};

}  // namespace gtsparse_row_template_center_last
