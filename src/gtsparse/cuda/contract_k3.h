#pragma once

namespace gtsparse_kernel3 {

constexpr int kNumLogicalOffsets = 3;
constexpr int kMaxPayloadSlots = 3;
constexpr int kBM = 128;

enum TemplateId : int {
    kTemplateOffset0 = 0,
    kTemplateOffset1 = 1,
    kTemplateOffset2 = 2,
    kTemplateOffsets01 = 3,
    kTemplateOffsets02 = 4,
    kTemplateOffsets12 = 5,
    kTemplateFull3 = 6,
    kNumTemplates = 7,
};

constexpr int kFamilyW1Templates = 3;
constexpr int kFamilyW2Templates = 3;
constexpr int kFamilyW3Templates = 1;
constexpr int kPayloadWidthW1 = 1;
constexpr int kPayloadWidthW2 = 2;
constexpr int kPayloadWidthW3 = 3;

static constexpr int kTemplateSlotCount[kNumTemplates] = {
    1, 1, 1, 2, 2, 2, 3,
};

static constexpr int kTemplateLocalIndex[kNumTemplates] = {
    0, 1, 2, 0, 1, 2, 0,
};

static constexpr int kTemplatePayloadOffset[kNumTemplates][kMaxPayloadSlots] = {
    {0, -1, -1},
    {1, -1, -1},
    {2, -1, -1},
    {0, 1, -1},
    {0, 2, -1},
    {1, 2, -1},
    {0, 1, 2},
};

}  // namespace gtsparse_kernel3
