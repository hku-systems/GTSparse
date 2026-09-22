#pragma once

namespace gtsparse_kernel9 {

constexpr int kNumLogicalOffsets = 9;
constexpr int kMaxPayloadSlots = 9;
constexpr int kBM = 128;

enum TemplateId : int {
    kTemplateCenter = 0,
    kTemplateSkip2Keep0 = 1,
    kTemplateSkip2Keep1 = 2,
    kTemplateSkip2Keep2 = 3,
    kTemplateSkip1Hole0 = 4,
    kTemplateSkip1Hole1 = 5,
    kTemplateSkip1Hole2 = 6,
    kTemplateFull9 = 7,
    kNumTemplates = 8,
};

constexpr int kFamilyW1Templates = 1;
constexpr int kFamilyW4Templates = 3;
constexpr int kFamilyW7Templates = 3;
constexpr int kFamilyW9Templates = 1;
constexpr int kPayloadWidthW1 = 1;
constexpr int kPayloadWidthW4 = 4;
constexpr int kPayloadWidthW7 = 7;
constexpr int kPayloadWidthW9 = 9;

static constexpr int kTemplateSlotCount[kNumTemplates] = {
    1, 4, 3, 4, 6, 7, 6, 9,
};

static constexpr int kTemplateLocalIndex[kNumTemplates] = {
    0, 0, 1, 2, 0, 1, 2, 0,
};

static constexpr int kTemplatePayloadOffset[kNumTemplates][kMaxPayloadSlots] = {
    {4, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 4, 6, -1, -1, -1, -1, -1},
    {1, 4, 7, -1, -1, -1, -1, -1, -1},
    {2, 4, 5, 8, -1, -1, -1, -1, -1},
    {1, 2, 4, 5, 7, 8, -1, -1, -1},
    {0, 2, 3, 4, 5, 6, 8, -1, -1},
    {0, 1, 3, 4, 6, 7, -1, -1, -1},
    {0, 1, 2, 3, 4, 5, 6, 7, 8},
};

}  // namespace gtsparse_kernel9
