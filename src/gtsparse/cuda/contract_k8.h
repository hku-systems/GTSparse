#pragma once

namespace gtsparse_kernel8 {

constexpr int kNumLogicalOffsets = 8;
constexpr int kBM = 128;
constexpr int kTemplateEmpty = 0;
constexpr int kTemplateSingletonBegin = 1;
constexpr int kTemplateFaceBegin = 9;
constexpr int kTemplateFull8 = 15;
constexpr int kNumTemplates = 16;
constexpr int kFamilyW1Templates = 8;
constexpr int kFamilyW2Templates = 0;
constexpr int kFamilyW4Templates = 6;
constexpr int kFamilyW8Templates = 1;

static constexpr int kTemplatePayloadOffset[kNumTemplates][8] = {
    {-1,-1,-1,-1,-1,-1,-1,-1},
    {0,-1,-1,-1,-1,-1,-1,-1}, {1,-1,-1,-1,-1,-1,-1,-1},
    {2,-1,-1,-1,-1,-1,-1,-1}, {3,-1,-1,-1,-1,-1,-1,-1},
    {4,-1,-1,-1,-1,-1,-1,-1}, {5,-1,-1,-1,-1,-1,-1,-1},
    {6,-1,-1,-1,-1,-1,-1,-1}, {7,-1,-1,-1,-1,-1,-1,-1},
    {0,1,2,3,-1,-1,-1,-1}, {4,5,6,7,-1,-1,-1,-1},
    {0,1,4,5,-1,-1,-1,-1}, {2,3,6,7,-1,-1,-1,-1},
    {0,2,4,6,-1,-1,-1,-1}, {1,3,5,7,-1,-1,-1,-1},
    {0,1,2,3,4,5,6,7},
};

}  // namespace gtsparse_kernel8
